"""One-off archive-facts backfill (Story 6.4).

Fills ``video_stats`` for the videos already indexed, so the landing-page
"status do acervo" shows full numbers from day one, and — for the same videos
— repairs the Qdrant payloads in place so results re-gain the episode/date
they were indexed without (the result card renders both from the payload).
Payload-only: no re-embedding, and each field is written only where it is
empty, so a value the index already knows is never overwritten.

Resumable and idempotent: a video whose row already carries duration, episode
and date is skipped, and a fetched cache entry is never re-fetched.

Fact resolution (fills NULLs only; duration precedence: cache > transcripts):
  ① metadata cache entry (arrives via the ingest pipeline)
  ② duration fallback: local transcripts cache ``transcripts/<id>.srt`` —
     ``max(start+duration)`` over the JSON segment list (the reference-script
     technique, zero network)
  ③ with ``--fetch-missing`` only: one yt-dlp ``--dump-single-json`` fetch,
     which fires whenever ANY fact is still missing (duration, episode or
     date — old pre-enrichment videos carry none) and upgrades the cache file
     with everything it learns (once ever; known fields are never regressed)
  ④ placeholder-title poison repair: a bot-gated yt-dlp can exit 0 with a
     degraded dump whose title is ``youtube video #<id>``, which seeded
     id-digit junk episodes into caches, rows and payloads. An episode whose
     digits occur inside the video id in a durationless cache entry is treated
     as MISSING (never as knowledge) and healed by ③; the payload value is
     overwritten, not preserved.

No fetch happens by default, so a run over ~3.5k videos is seconds of local
JSON parsing and never exposes a datacenter IP to YouTube bot detection.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from mc import config
from mc.config import COLLECTION_NAME
from mc.ingest.metadata import (
    _default_fetch,
    _normalize,
    duration_int,
    episode_int,
)
from mc.state import StateStore

logger = logging.getLogger(__name__)

#: Seconds to sleep between yt-dlp fetches when --fetch-missing is in effect.
_FETCH_BACKOFF_S = 5.0


@dataclass
class StatsBackfillReport:
    videos_considered: int = 0
    videos_backfilled: int = 0  # gained a non-null duration this run
    videos_skipped: int = 0  # already complete — resumable no-op
    still_missing: int = 0  # duration still None after the run
    fetched: int = 0  # videos whose missing facts arrived via a yt-dlp fetch
    index_updated: int = 0  # videos whose payload gained a missing fact

    def summary(self, stats: dict) -> str:
        seconds = int(stats.get("seconds") or 0)
        line = (
            f"stats-backfill: {self.videos_backfilled} backfilled, "
            f"{self.videos_skipped} already complete, "
            f"{self.still_missing} still missing duration; "
            f"{stats['videos']} videos, {seconds} s "
            f"({seconds // 60} min, {seconds // 3600} h, {seconds // 86400} d)"
        )
        if self.index_updated:
            line += f"; {self.index_updated} index payloads repaired"
        return line


def _transcript_duration(segments: list[dict]) -> int | None:
    """``int(max(start + duration))`` over the transcript segments (the
    reference-script technique). None when there are no usable segments."""
    ends: list[float] = []
    for seg in segments:
        try:
            ends.append(float(seg["start"]) + float(seg["duration"]))
        except (KeyError, TypeError, ValueError):
            continue
    return int(max(ends)) if ends else None


def _load_transcript(transcripts_dir: Path, video_id: str) -> list[dict] | None:
    """The cached transcript segment list, or None when absent/unreadable."""
    path = transcripts_dir / f"{video_id}.srt"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("transcript for %s unreadable (%s)", video_id, exc)
        return None
    return data if isinstance(data, list) else None


def _load_cache(cache_dir: Path, video_id: str) -> dict:
    path = cache_dir / f"{video_id}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _poisoned_episode(video_id: str, cache: dict) -> bool:
    """True when the cached episode is a placeholder-title artifact.

    A bot-gated yt-dlp can exit 0 with a degraded dump whose title is the
    placeholder ``youtube video #<video_id>``; the episode parser then read
    digits from the video id itself ("xg301415xZU" → 301415, "--5z_jFh-_g"
    → 5), and MAX(episode) crowned a nonsense "último episódio". Marker: the
    entry carries no duration (degraded dumps never write the key — real
    fetches always do) AND the episode's digits occur inside the video id."""
    if "duration" in cache:
        return False
    episode = episode_int(cache.get("episode"))
    return episode is not None and str(episode) in video_id


def _cache_ids(cache_dir: Path) -> set[str]:
    if not cache_dir.exists():
        return set()
    return {p.stem for p in cache_dir.glob("*.json")}


def _resolve_facts(
    video_id: str,
    cache: dict,
    cache_dir: Path,
    transcripts_dir: Path,
    *,
    fetch_missing: bool,
    fetch: Callable[[str], dict] | None,
    sleeper: Callable[[float], None],
) -> tuple[dict, bool]:
    """Return (facts, fetched) with facts = {duration, episode, date}.

    Duration precedence: cache > transcripts > fetch; episode/date come only
    from the cache (or the fetch). A fetch — only when ``fetch_missing`` and
    SOMETHING is still missing, be it the duration or the episode/date of an
    old pre-enrichment video — upgrades the cache file with everything it
    learns, so each video is fetched at most once ever. Fields the cache
    already knows are kept, never regressed, by that upgrade.
    """
    duration = duration_int(cache.get("duration"))
    if duration is None:
        segments = _load_transcript(transcripts_dir, video_id)
        if segments is not None:
            duration = _transcript_duration(segments)
    facts = {
        "duration": duration,
        "episode": episode_int(cache.get("episode")),
        "date": cache.get("date") or None,
    }
    if not fetch_missing or fetch is None or all(value is not None for value in facts.values()):
        return facts, False

    try:
        raw = fetch(video_id)
    except Exception as exc:  # noqa: BLE001 — skip, never fatal
        logger.warning("fetch failed for %s: %s", video_id, exc)
        return facts, False
    sleeper(_FETCH_BACKOFF_S)
    meta = _normalize(raw)
    merged = {**cache, **{k: v for k, v in meta.items() if v is not None}}
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{video_id}.json").write_text(json.dumps(merged), encoding="utf-8")

    fetched_duration = duration_int(merged.get("duration"))
    return {
        "duration": fetched_duration if fetched_duration is not None else duration,
        "episode": episode_int(merged.get("episode")),
        "date": merged.get("date") or None,
    }, True


def _repair_payload(
    client: QdrantClient,
    collection: str,
    video_id: str,
    facts: dict,
    overwrite: set[str] | None = None,
) -> int:
    """Fill this video's payloads where a fact is missing (Story 6.4 index repair).

    One ``set_payload`` per field — payload-only, no re-embedding — each scoped
    by a filter that requires the ``video_id`` AND that the field is still
    empty. A value the index already carries is therefore never overwritten,
    which keeps the whole operation idempotent and safe to re-run. Fields named
    in ``overwrite`` drop that is-empty guard: a poisoned value (junk parsed
    from a placeholder title) must be REPLACED even though the field is not
    empty. Returns how many fields were offered (0 → nothing to do)."""
    overwrite = overwrite or set()
    offered = 0
    for field, value in facts.items():
        if value is None:
            continue
        must = [models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
        if field not in overwrite:
            must.append(models.IsEmptyCondition(is_empty=models.PayloadField(key=field)))
        client.set_payload(
            collection_name=collection,
            payload={field: value},
            points=models.Filter(must=must),
        )
        offered += 1
    return offered


def run_stats_backfill(
    *,
    fetch_missing: bool = False,
    limit: int | None = None,
    state: StateStore | None = None,
    cache_dir: Path | None = None,
    transcripts_dir: Path | None = None,
    fetch: Callable[[str], dict] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
    repair_index: bool = True,
) -> dict:
    """Backfill the archive's per-video facts and repair the Qdrant payloads.

    Video universe = ``state.processed`` ∪ metadata-cache ids. Per video,
    episode/date come from the cache and duration follows the precedence above.
    Each fact the row still lacks is written to ``video_stats``, and — when a
    ``client`` is given and ``repair_index`` is set — patched into that video's
    payloads (payload-only; only where the field is still empty). ``limit``
    bounds the number of videos considered this run. Returns a report dict; the
    operator-facing summary is printed by the CLI.
    """
    state = state or StateStore(config.INGEST_DB_PATH)
    cache_dir = Path(cache_dir) if cache_dir is not None else config.META_CACHE_DIR
    transcripts_dir = (
        Path(transcripts_dir) if transcripts_dir is not None else config.TRANSCRIPTS_DIR
    )
    fetcher = fetch if fetch is not None else (_default_fetch if fetch_missing else None)

    report = StatsBackfillReport()
    universe = sorted(set(state.processed_ids()) | _cache_ids(cache_dir))

    for video_id in universe:
        if limit is not None and report.videos_considered >= limit:
            break
        report.videos_considered += 1

        cache = _load_cache(cache_dir, video_id)
        # A poisoned episode is not knowledge: neutralize it in both the cache
        # view and the row view, so the resolution re-fetches it and the index
        # repair REPLACES the junk instead of preserving it.
        poisoned = _poisoned_episode(video_id, cache)
        if poisoned:
            cache = {**cache, "episode": None}

        current = state.video_facts(video_id)
        if poisoned:
            current = {**current, "episode": None}
        # Resumable in BOTH modes: a row that already carries duration, episode
        # AND date has nothing left to fill — skip the (possibly expensive)
        # resolution and the index repair. Checking every field, not just the
        # duration, is what lets a video that already had a duration still gain
        # its missing episode/date (the old duration-only skip left them NULL).
        if all(current[field] is not None for field in ("duration", "episode", "date")):
            report.videos_skipped += 1
            continue

        facts, fetched = _resolve_facts(
            video_id,
            cache,
            cache_dir,
            transcripts_dir,
            fetch_missing=fetch_missing,
            fetch=fetcher,
            sleeper=sleeper,
        )
        if fetched:
            report.fetched += 1

        try:
            state.record_video_stats(
                video_id,
                duration=facts["duration"],
                episode=facts["episode"],
                date=facts["date"],
            )
        except Exception:  # noqa: BLE001 — never fatal to the backfill
            logger.warning("stats write failed for %s", video_id, exc_info=True)
            continue

        if repair_index and client is not None:
            # the payload keeps episode as the string the ingest pipeline writes
            pending = {
                "duration": facts["duration"],
                "episode": (str(facts["episode"]) if facts["episode"] is not None else None),
                "date": facts["date"],
            }
            missing = {f: v for f, v in pending.items() if current[f] is None}
            try:
                if _repair_payload(
                    client,
                    collection,
                    video_id,
                    missing,
                    overwrite={"episode"} if poisoned else set(),
                ):
                    report.index_updated += 1
            except Exception:  # noqa: BLE001 — repair must never fail the run
                logger.warning("index payload repair failed for %s", video_id, exc_info=True)

        if facts["duration"] is not None and current["duration"] is None:
            report.videos_backfilled += 1
        elif facts["duration"] is None:
            report.still_missing += 1

    stats = state.archive_stats()
    logger.info(report.summary(stats))
    return {
        "videos_considered": report.videos_considered,
        "videos_backfilled": report.videos_backfilled,
        "videos_skipped": report.videos_skipped,
        "still_missing": report.still_missing,
        "fetched": report.fetched,
        "index_updated": report.index_updated,
        "videos": stats["videos"],
        "seconds": stats["seconds"],
        "summary": report.summary(stats),
    }
