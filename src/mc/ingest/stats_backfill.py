"""One-off archive-facts backfill (Story 6.4; tab-driven date hydration: Story 8.1).

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

Story 8.1 adds ``run_tab_backfill`` (--from-tabs): the production index is
undated, so the temporal sort modes have nothing to order by. One yt-dlp
FULL-extraction pass per channel tab prints id/upload_date/title/duration
as TSV — ``upload_date`` is not available in --flat-playlist listings, and
the approximate-date variant decays to year granularity on old entries
(decision D13), so exact extraction is the only trustworthy source. Pacing
is yt-dlp's own randomized sleep (3–8 s, the reference-script parameters)
plus one process per tab instead of per video. Every write is idempotent:
the cache merge never regresses known fields, the row upsert is COALESCE
non-null-wins and the payload repair only fills empty fields.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models

from mc import config
from mc.config import COLLECTION_NAME
from mc.ingest.discover import _TAB_ABSENT_RE, _VIDEO_ID_RE, _channel_tabs, load_channels
from mc.ingest.legacy import iter_legacy_titles
from mc.ingest.metadata import (
    _default_fetch,
    _normalize,
    _parse_episode,
    duration_int,
    episode_int,
)
from mc.state import StateStore

logger = logging.getLogger(__name__)

#: Seconds to sleep between yt-dlp fetches when --fetch-missing is in effect.
_FETCH_BACKOFF_S = 5.0

#: Output template for the tab pass: TSV so titles can hold any separator
#: character other than the tab itself (which YouTube titles cannot contain).
_TAB_TEMPLATE = "%(id)s\t%(upload_date)s\t%(title)s\t%(duration)s"

#: Randomized pacing flags — the live_gerard reference script's proven
#: parameters: yt-dlp sleeps a random 3–8 s between extractions and 1 s
#: between API requests, retrying transient failures 3 times.
_TAB_PACING = (
    "--sleep-requests",
    "1",
    "--sleep-interval",
    "3",
    "--max-sleep-interval",
    "8",
    "--extractor-retries",
    "3",
    "--retry-sleep",
    "5",
)


@dataclass
class StatsBackfillReport:
    videos_considered: int = 0
    videos_backfilled: int = 0  # gained a non-null duration this run
    videos_skipped: int = 0  # already complete — resumable no-op
    still_missing: int = 0  # duration still None after the run
    fetched: int = 0  # videos whose missing facts arrived via a yt-dlp fetch
    index_updated: int = 0  # videos whose payload gained a missing fact
    episodes_repaired: int = 0  # rows whose episode was re-derived/cleared via --from-db

    def summary(self, stats: dict) -> str:
        seconds = int(stats.get("seconds") or 0)
        line = (
            f"stats-backfill: {self.videos_backfilled} backfilled, "
            f"{self.videos_skipped} already complete, "
            f"{self.still_missing} still missing duration; "
            f"{stats['videos']} videos, {seconds} s "
            f"({seconds // 60} min, {seconds // 3600} h, {seconds // 86400} d)"
        )
        if self.episodes_repaired:
            line += f"; {self.episodes_repaired} episodes re-derived"
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
    clear: set[str] | None = None,
) -> int:
    """Fill this video's payloads where a fact is missing (Story 6.4 index repair).

    One ``set_payload`` per field — payload-only, no re-embedding — each scoped
    by a filter that requires the ``video_id`` AND that the field is still
    empty. A value the index already carries is therefore never overwritten,
    which keeps the whole operation idempotent and safe to re-run. Fields named
    in ``overwrite`` drop that is-empty guard: a value known to be wrong
    (placeholder-title poison, or one the authoritative title corrects) must
    be REPLACED even though the field is not empty. Fields named in ``clear``
    are REMOVED from the payload instead (``delete_payload``) — for a fact the
    title proves should not exist at all. Returns how many operations were
    offered (0 → nothing to do)."""
    overwrite = overwrite or set()
    clear = clear or set()
    offered = 0
    video_must = models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))
    for field in sorted(clear):
        client.delete_payload(
            collection_name=collection,
            keys=[field],
            points=models.Filter(must=[video_must]),
        )
        offered += 1
    for field, value in facts.items():
        if value is None:
            continue
        must = [video_must]
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
    from_db: Path | None = None,
) -> dict:
    """Backfill the archive's per-video facts and repair the Qdrant payloads.

    Video universe = ``state.processed`` ∪ metadata-cache ids. Per video,
    episode/date come from the cache and duration follows the precedence above.
    Each fact the row still lacks is written to ``video_stats``, and — when a
    ``client`` is given and ``repair_index`` is set — patched into that video's
    payloads (payload-only; only where the field is still empty). ``limit``
    bounds the number of videos considered this run. Returns a report dict; the
    operator-facing summary is printed by the CLI.

    With ``from_db`` (a legacy ``mc_transcripts.db``), each video's chunked
    title is the AUTHORITATIVE episode source: rows, cache files and payloads
    are re-derived from it — a declared marker keeps (or gains) its episode,
    and a marker-less title clears the junk the COALESCE upsert could never
    erase. Fully offline; a run over ~3.5k videos is seconds of SQLite reads.
    """
    state = state or StateStore(config.INGEST_DB_PATH)
    cache_dir = Path(cache_dir) if cache_dir is not None else config.META_CACHE_DIR
    transcripts_dir = (
        Path(transcripts_dir) if transcripts_dir is not None else config.TRANSCRIPTS_DIR
    )
    fetcher = fetch if fetch is not None else (_default_fetch if fetch_missing else None)
    titles: dict[str, str] | None = None
    if from_db is not None:
        titles = dict(iter_legacy_titles(Path(from_db)))

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
        row_episode = current["episode"]
        if poisoned:
            current = {**current, "episode": None}

        # --from-db: the title is the authoritative episode source — a row
        # value that disagrees is junk to repair, not knowledge to keep.
        title_episode: int | None = None
        has_title = titles is not None and video_id in titles
        if has_title:
            title_episode = episode_int(_parse_episode(titles[video_id]))
            cache = {**cache, "episode": title_episode}
        episode_repair_needed = has_title and row_episode != title_episode
        if episode_repair_needed:
            report.episodes_repaired += 1

        # Resumable in BOTH modes: a row that already carries duration, episode
        # AND date has nothing left to fill — skip the (possibly expensive)
        # resolution and the index repair. Checking every field, not just the
        # duration, is what lets a video that already had a duration still gain
        # its missing episode/date (the old duration-only skip left them NULL).
        # Rows the title proves wrong are never "done". And when --from-db is
        # given the title is authoritative for the payload too, so a title-
        # bearing video is never skipped: its payload episode is re-synced to
        # the truth even when the row already agrees (the payload may lag).
        if (
            all(current[field] is not None for field in ("duration", "episode", "date"))
            and not episode_repair_needed
            and not has_title
        ):
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
            if episode_repair_needed and title_episode is None:
                # COALESCE can never erase — the authoritative clear is explicit
                state.set_video_episode(video_id, None)
        except Exception:  # noqa: BLE001 — never fatal to the backfill
            logger.warning("stats write failed for %s", video_id, exc_info=True)
            continue

        if has_title and (cache_dir / f"{video_id}.json").exists():
            # write-through: a plain re-run (no --from-db) reads the cache and
            # would re-junk the corrected row if the file kept the old value
            stored = _load_cache(cache_dir, video_id)
            wanted = str(title_episode) if title_episode is not None else None
            if stored.get("episode") != wanted:
                stored["episode"] = wanted
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / f"{video_id}.json").write_text(json.dumps(stored), encoding="utf-8")

        if repair_index and client is not None:
            # the payload keeps episode as the string the ingest pipeline writes
            pending = {
                "duration": facts["duration"],
                "episode": (str(facts["episode"]) if facts["episode"] is not None else None),
                "date": facts["date"],
            }
            missing = {f: v for f, v in pending.items() if current[f] is None}
            overwrite: set[str] = set()
            clear: set[str] = set()
            if poisoned:
                overwrite.add("episode")
            if has_title:
                # the title is authoritative for the payload episode: overwrite
                # any stale value (junk or otherwise) with the truth, or clear
                # it when the title declares none. The row may already agree
                # while the payload lags behind, so this runs whenever --from-db
                # is in effect — not only when the row itself needed repair.
                if facts["episode"] is not None:
                    overwrite.add("episode")
                    missing["episode"] = str(facts["episode"])
                else:
                    clear.add("episode")
                    missing.pop("episode", None)
            elif episode_repair_needed:
                # no title: only a poison hit forces a replace
                if facts["episode"] is not None:
                    overwrite.add("episode")
                else:
                    # the title declares no episode: the junk must GO, not stay
                    clear.add("episode")
                    overwrite.discard("episode")
                    missing.pop("episode", None)
            try:
                if _repair_payload(
                    client,
                    collection,
                    video_id,
                    missing,
                    overwrite=overwrite,
                    clear=clear,
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
        "episodes_repaired": report.episodes_repaired,
        "videos": stats["videos"],
        "seconds": stats["seconds"],
        "summary": report.summary(stats),
    }


@dataclass
class TabBackfillReport:
    tabs_listed: int = 0  # tabs whose pass ran to completion
    tabs_skipped: int = 0  # tabs the channel does not publish (normal)
    tabs_failed: int = 0  # bot gate, network, anything else — never fatal
    videos_seen: int = 0  # valid video ids parsed across all passes
    videos_dated: int = 0  # rows that gained a date this run
    cache_written: int = 0  # cache files that gained a field
    index_updated: int = 0  # payloads that gained a missing fact

    def summary(self) -> str:
        return (
            f"stats-backfill tabs: {self.videos_seen} videos seen across "
            f"{self.tabs_listed} tabs, {self.videos_dated} dated, "
            f"{self.cache_written} caches upgraded, "
            f"{self.index_updated} index payloads repaired; "
            f"{self.tabs_skipped} tabs skipped, {self.tabs_failed} tabs failed"
        )


def _default_tab_runner(argv: list[str]) -> subprocess.CompletedProcess:
    """Run one tab pass. No timeout: a full tab is hours of paced extraction,
    and killing it mid-pass would only cost the elapsed work (re-runs are
    idempotent, but there is no reason to make the operator wait twice)."""
    return subprocess.run(argv, capture_output=True, text=True)


def _tab_argv(tab_url: str) -> list[str]:
    from mc.config import COOKIES_PATH

    argv = [
        "yt-dlp",
        "--no-warnings",
        "--skip-download",
        "--ignore-errors",  # one gated video must not kill a multi-hour pass
        "--print",
        _TAB_TEMPLATE,
        *_TAB_PACING,
    ]
    if Path(COOKIES_PATH).exists():
        argv += ["--cookies", str(COOKIES_PATH)]
    argv.append(tab_url)
    return argv


def _duration_text(value: str) -> float | None:
    """yt-dlp prints durations as floats ("5400.0"); NA/garbage is absence."""
    if value == "NA":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_tab_line(line: str) -> tuple[str, dict] | None:
    """(video_id, raw-for-_normalize) from one TSV line, or None to skip.

    yt-dlp prints NA for fields an entry lacks; NA is absence, not data."""
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    video_id, upload_date, title = parts[0], parts[1], parts[2]
    duration = parts[3] if len(parts) > 3 else "NA"
    if not _VIDEO_ID_RE.fullmatch(video_id):
        logger.warning("skipping entry with invalid video id: %r", video_id)
        return None
    na = lambda value: None if value == "NA" else value  # noqa: E731
    return video_id, {
        "upload_date": na(upload_date),
        "title": title,
        "duration": _duration_text(duration),
    }


def run_tab_backfill(
    *,
    state: StateStore | None = None,
    cache_dir: Path | None = None,
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
    channels: tuple[str, ...] | list[str] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None,
    repair_index: bool = True,
) -> dict:
    """Hydrate exact upload dates from the channel tabs themselves (Story 8.1).

    For every configured channel (config.yaml → MC_CHANNELS) and every one of
    its /videos /streams /shorts tabs, one uncapped yt-dlp full-extraction pass
    prints ``id/upload_date/title/duration`` TSV. Each parsed line merges into
    the meta cache (known fields never regressed), dates the ``video_stats``
    row and — when a ``client`` is given and ``repair_index`` is set — fills
    that video's empty Qdrant payloads. Writes for videos the index does not
    carry are harmless: the row becomes "seen, stats unknown" and the payload
    repair no-ops. An absent tab is normal; any other tab failure is logged,
    counted and the run continues. Returns a report dict; the CLI prints the
    summary.
    """
    state = state or StateStore(config.INGEST_DB_PATH)
    cache_dir = Path(cache_dir) if cache_dir is not None else config.META_CACHE_DIR
    runner = runner or _default_tab_runner
    channel_urls = tuple(channels) if channels is not None else load_channels()

    report = TabBackfillReport()
    for channel in channel_urls:
        for tab_url in _channel_tabs(channel):
            result = runner(_tab_argv(tab_url))
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()
                if _TAB_ABSENT_RE.search(stderr):
                    logger.warning("channel %s has no %s tab — skipped", channel, tab_url)
                    report.tabs_skipped += 1
                else:
                    logger.warning("tab pass failed for %s: %s", tab_url, stderr[:400])
                    report.tabs_failed += 1
                sleeper(_FETCH_BACKOFF_S)
                continue

            for line in (result.stdout or "").splitlines():
                parsed = _parse_tab_line(line.strip())
                if parsed is None:
                    continue
                video_id, raw = parsed
                report.videos_seen += 1

                meta = _normalize(raw)
                cache = _load_cache(cache_dir, video_id)
                # The cache is authoritative for fields it already knows (the
                # tab pass fills gaps; it must never churn existing values) —
                # unless the entry is poisoned, in which case the tab data is
                # the healing truth and overwrites (poison is not knowledge).
                fresh = {k: v for k, v in meta.items() if v is not None}
                merged = (
                    {**cache, **fresh}
                    if _poisoned_episode(video_id, cache)
                    else {**fresh, **cache}
                )
                if merged != cache:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    (cache_dir / f"{video_id}.json").write_text(
                        json.dumps(merged), encoding="utf-8"
                    )
                    report.cache_written += 1

                facts = {
                    "duration": duration_int(merged.get("duration")),
                    "episode": episode_int(merged.get("episode")),
                    "date": merged.get("date") or None,
                }
                try:
                    current = state.video_facts(video_id)
                    state.record_video_stats(
                        video_id,
                        duration=facts["duration"],
                        episode=facts["episode"],
                        date=facts["date"],
                    )
                    if current["date"] is None and facts["date"] is not None:
                        report.videos_dated += 1
                except Exception:  # noqa: BLE001 — never fatal to the backfill
                    logger.warning("stats write failed for %s", video_id, exc_info=True)
                    continue

                if repair_index and client is not None:
                    # the payload keeps episode as the string the ingest
                    # pipeline writes; only fields the index lacks are offered
                    pending = {
                        "duration": facts["duration"],
                        "episode": (
                            str(facts["episode"]) if facts["episode"] is not None else None
                        ),
                        "date": facts["date"],
                    }
                    missing = {f: v for f, v in pending.items() if current[f] is None}
                    try:
                        if _repair_payload(client, collection, video_id, missing):
                            report.index_updated += 1
                    except Exception:  # noqa: BLE001 — repair must never fail the run
                        logger.warning(
                            "index payload repair failed for %s", video_id, exc_info=True
                        )
            report.tabs_listed += 1
            sleeper(_FETCH_BACKOFF_S)

    logger.info(report.summary())
    return {
        "tabs_listed": report.tabs_listed,
        "tabs_skipped": report.tabs_skipped,
        "tabs_failed": report.tabs_failed,
        "videos_seen": report.videos_seen,
        "videos_dated": report.videos_dated,
        "cache_written": report.cache_written,
        "index_updated": report.index_updated,
        "summary": report.summary(),
    }
