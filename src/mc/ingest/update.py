"""Daily ingestion orchestrator (Story 3.4 / 10.2).

Discover → download → chunk → embed → upsert → mark processed, one
atomic per-video transaction at a time. Chunk records are staged to
``data/queue/pending.jsonl`` so ``mc push``/``mc apply`` (Mode B) ship
exactly the same increment that Mode A indexes directly.

Story 10.2 (NFR-3) closes the silent-empty-run hole: an empty discovery, a
discovered backlog that is entirely ignore-listed, and any transient failure
(429/timeout/5xx) all fail the run (exit 1) without recording a successful
ingest — so the health gate keeps alerting instead of reporting "up to date"
while the archive stops growing. Every run logs a discovery summary at INFO
before any early return, so "nothing new" is distinguishable from "nothing
found" in the cron log.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient

from mc.config import (
    COLLECTION_NAME,
    COOKIES_PATH,
    DATA_DIR,
    INGEST_DB_PATH,
    META_CACHE_DIR,
    QDRANT_URL,
    TRANSCRIPTS_DIR,
)
from mc.ingest.chunker import chunk_segments
from mc.ingest.discover import list_channel_videos, load_channels
from mc.ingest.download import (
    IGNORE_LIST_FILENAME,
    add_to_ignore_list,
    download_transcript,
    is_ignored,
)
from mc.ingest.lock import RunLock
from mc.ingest.metadata import enrich, episode_int
from mc.qdrant import client_kwargs, ensure_collection, upsert_video
from mc.state import StateStore

logger = logging.getLogger(__name__)

PENDING_NAME = "pending.jsonl"

# Story 5.6: stale cookies look like any other download failure to yt-dlp,
# but ignore-listing the video would skip it forever. Auth-shaped errors are
# reported as a distinct reason and never ignore-list.
_AUTH_MARKERS = (
    "sign in",
    "sign-in",
    "log in",
    "login",
    "cookies",
    "confirm you",
    "not a bot",
    "membership",
    "unauthorized",
)

# Story 10.2: a transient failure is the network's fault, not the video's.
# Ignore-listing it (the pre-fix behaviour) hid the video from every later
# run — that is how production accumulated a permanently poisoned list.
_TRANSIENT_MARKERS = (
    "429",
    "too many requests",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "502",
    "503",
    "504",
)


def is_auth_failure(reason: str) -> bool:
    lowered = (reason or "").lower()
    return any(marker in lowered for marker in _AUTH_MARKERS)


def is_transient_failure(reason: str) -> bool:
    """True for retryable failures (rate limit, timeout, 5xx) — never ignored."""
    lowered = (reason or "").lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)



class StagingError(RuntimeError):
    """Local failure writing the Mode B increment after a successful upsert.

    Never ignore-listed: the video is already indexed, so the next run must
    retry the whole ingest (push/apply dedupe by content hash).
    """


def run_update(
    *,
    dry_run: bool = False,
    channels: tuple[str, ...] | None = None,
    state: StateStore | None = None,
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
    discover: Callable[[tuple[str, ...]], list[str] | set[str]] | None = None,
    downloader: Callable[[str], list[dict] | None] | None = None,
    metadata_fetch: Callable[[list[str]], dict] | None = None,
    lock_path: Path | None = None,
    cookies_path: str | None = None,
    transcripts_dir: Path | None = None,
    queue_dir: Path | None = None,
) -> int:
    """Run one daily ingestion pass. Returns a process exit code."""
    # Same db as push/apply/backfill and the API freshness check (review fix:
    # a private data/ingest.db left /health blind to update's last_success).
    state = state or StateStore(INGEST_DB_PATH)
    client = client or QdrantClient(url=QDRANT_URL, **client_kwargs(timeout=60))
    channels = channels or load_channels()
    discover = discover or list_channel_videos
    downloader = downloader or _default_downloader(cookies_path, transcripts_dir)
    transcripts_dir = Path(transcripts_dir or TRANSCRIPTS_DIR)
    queue_dir = Path(queue_dir or DATA_DIR / "queue")
    lock_path = lock_path or DATA_DIR / "update.lock"

    discovered = list(discover(channels))
    if not discovered:
        # Story 10.2: an empty listing is an outage or a broken config, never
        # "up to date". Failing loud (and not recording success) keeps the
        # health gate alerting instead of reporting a fresh archive.
        logger.error(
            "discovery returned no videos for %d channel(s) — refusing to report success",
            len(channels),
        )
        return 1

    # An ignore-listed video is a KNOWN uninjestable (permanently subtitle-less):
    # re-attempting it every run failed the whole pass and fired the cron alert
    # forever (observed in production with wtptWokZag8). The list is a
    # discovery-level exclusion — skip before enrich/download, count it as
    # neither new nor failed, and let the run exit 0.
    ignored = {
        vid for vid in discovered if is_ignored(vid, transcripts_dir / IGNORE_LIST_FILENAME)
    }
    if ignored:
        logger.info(
            "%d discovered videos are in the ignore list — skipped", len(ignored)
        )
    unprocessed = [vid for vid in discovered if not state.is_processed(vid)]
    new_ids = sorted(vid for vid in unprocessed if vid not in ignored)

    # Story 10.2: logged at INFO before any early return — the cron log must
    # let the operator tell "nothing new" from "nothing found".
    logger.info(
        "discovery summary: %d discovered, %d already processed, %d ignored, "
        "%d new videos",
        len(discovered),
        len(discovered) - len(unprocessed),
        len(ignored),
        len(new_ids),
    )

    if not unprocessed:
        logger.info("nothing to do — archive is up to date")
        if not dry_run:
            state.record_success("update")
        return 0

    if not new_ids:
        # Every discovered-but-unprocessed video is ignore-listed: the
        # pipeline cannot make progress. That is a failure, not a clean run.
        logger.error(
            "no progress: all %d unprocessed discovered videos are ignore-listed — "
            "prune transcripts/%s if the entries are wrong (e.g. past 429s)",
            len(unprocessed),
            IGNORE_LIST_FILENAME,
        )
        return 1

    if dry_run:
        logger.info("dry run: %d new videos would be ingested", len(new_ids))
        return 0

    with RunLock(lock_path):
        ensure_collection(client, collection)
        metadata = (
            metadata_fetch(new_ids)
            if metadata_fetch is not None
            else enrich(new_ids, cache_dir=META_CACHE_DIR)
        )

        failures: list[tuple[str, str]] = []
        for video_id in new_ids:
            try:
                _ingest_one(
                    client, collection, video_id, downloader, metadata, state, queue_dir
                )
            except Exception as exc:  # noqa: BLE001 — isolate every video
                # Whitespace-collapsed: exc text lands in ignored_videos.txt
                # and log lines, so embedded newlines must not forge entries.
                reason = " ".join((str(exc) or type(exc).__name__).split())
                if is_auth_failure(reason):
                    logger.error(
                        "auth failure for %s: cookies likely stale — "
                        "rotate them (docs/tasks/cookies.md)",
                        video_id,
                    )
                    failures.append((video_id, f"auth: {reason}"))
                    continue
                if isinstance(exc, StagingError):
                    logger.error(
                        "staging failure for %s (%s) — indexed but increment "
                        "not staged; will be retried next run",
                        video_id,
                        reason,
                    )
                    failures.append((video_id, f"staging: {reason}"))
                    continue
                if is_transient_failure(reason):
                    # Story 10.2: retry tomorrow, never ignore-list — a 429 is
                    # IP-level, and writing it to the list hides the video
                    # from every later run.
                    logger.error(
                        "transient failure for %s (%s) — will be retried next run",
                        video_id,
                        reason,
                    )
                    failures.append((video_id, f"transient: {reason}"))
                    continue
                logger.exception("video %s failed", video_id)
                failures.append((video_id, reason))
                add_to_ignore_list(
                    video_id, transcripts_dir / IGNORE_LIST_FILENAME, reason
                )


        logger.info(
            "run summary: %d new videos, %d indexed, %d failed",
            len(new_ids),
            len(new_ids) - len(failures),
            len(failures),
        )
        if failures:
            for video_id, reason in failures:
                logger.warning("failed video %s: %s", video_id, reason)
            return 1

        state.record_success("update")
        return 0


def _ingest_one(
    client: QdrantClient,
    collection: str,
    video_id: str,
    downloader: Callable[[str], list[dict] | None],
    metadata: dict[str, dict],
    state: StateStore,
    queue_dir: Path,
) -> None:
    """Download → chunk → upsert → stage → mark processed for ONE video."""
    segments = downloader(video_id)
    if segments is None:
        raise LookupError("no subtitles available in any supported language")

    info = metadata.get(video_id, {})
    chunks = [
        dataclasses.replace(
            chunk,
            date=info.get("date"),
            episode=info.get("episode"),
            channel=info.get("channel"),
            duration=info.get("duration"),
        )
        for chunk in chunk_segments(segments, video_id)
    ]
    upsert_video(client, collection, video_id, chunks)
    try:
        _stage_for_push(chunks, queue_dir)
    except Exception as exc:
        raise StagingError(str(exc) or type(exc).__name__) from exc
    # Mark only after the Mode B increment is durable (review fix): a crash
    # before this line leaves the video re-discoverable, and push/apply
    # dedupe by content hash so a re-staged duplicate is a no-op.
    state.mark_processed(video_id, "update")
    # Story 6.3: record archive stats only AFTER the durable commit (same
    # post-commit ordering). Best-effort — the ingest already succeeded.
    _record_stats(state, video_id, info)


def _record_stats(state: StateStore, video_id: str, info: dict) -> None:
    """Write this video's archive stats (Story 6.3). Post-commit and
    non-fatal: the ingest already succeeded, so a stats failure is logged and
    swallowed — the idempotent upsert lets the next run re-record."""
    try:
        state.record_video_stats(
            video_id,
            duration=info.get("duration"),
            episode=episode_int(info.get("episode")),
            date=info.get("date"),
        )
    except Exception:  # noqa: BLE001 — stats must never break ingest
        logger.warning("archive-stats write failed for %s", video_id, exc_info=True)


def _stage_for_push(chunks, queue_dir: Path) -> None:
    """Append the chunk records to data/queue/pending.jsonl (Mode B increment)."""
    queue_dir.mkdir(parents=True, exist_ok=True)
    with open(queue_dir / PENDING_NAME, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_payload(), sort_keys=True) + "\n")


def _default_downloader(cookies_path: str | None, transcripts_dir: Path | None):
    cookies = cookies_path or COOKIES_PATH
    dir_ = transcripts_dir or TRANSCRIPTS_DIR

    def download(video_id: str) -> list[dict] | None:
        return download_transcript(
            video_id, cookies_path=cookies, transcripts_dir=Path(dir_)
        )

    return download
