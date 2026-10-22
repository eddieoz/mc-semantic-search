"""Apply queued increments to the index (Story 3.7, Mode B server).

Consumes the dated JSONL files pushed into the queue directory by
`mc push`: parses the chunk records, groups them per video and reuses
the embed+upsert path (`upsert_video`, atomic per-video transaction,
Story 1.4). Applied-once bookkeeping is per CONTENT HASH of the chunk
record (same convention as `mc push`, key prefix `applied:`), marked
only after the video's upsert commits — so a crash or corrupt file
leaves everything retryable and a re-run is a no-op. Malformed files
are quarantined into <queue_dir>/failed/ instead of blocking the queue.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient

from mc import config
from mc.ingest.lock import RunLock
from mc.ingest.metadata import duration_int, episode_int
from mc.ingest.models import Chunk
from mc.qdrant import client_kwargs, upsert_video
from mc.state import StateStore

logger = logging.getLogger(__name__)

_APPLIED_PREFIX = "applied:"
_PENDING_NAME = "pending.jsonl"  # `mc update`'s staging file — never an increment
_FAILED_DIR = "failed"
_LOCK_NAME = "update.lock"  # shared with `mc update`/`mc push` so they never overlap
_MAX_TEXT_CHARS = 50_000  # a 30-segment chunk is ~2k chars; anything above is hostile
_MAX_FILE_BYTES = 100 * 1024 * 1024  # refuse to load larger increments into memory


def _content_hash(record: dict) -> str:
    """sha256 of the canonical JSON dump — the same key convention as push."""
    payload = json.dumps(record, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _queue_files(queue_dir: Path) -> list[Path]:
    """Only dated `*.jsonl` increment files; skip update's pending.jsonl."""
    return sorted(p for p in queue_dir.glob("*.jsonl") if p.name != _PENDING_NAME)


def _parse_file(path: Path) -> list[dict]:
    """Parse one increment file; raise ValueError on the first malformed line."""
    size = path.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise ValueError(
            f"file of {size} bytes exceeds the {_MAX_FILE_BYTES}-byte cap — not loading into memory"
        )
    records: list[dict] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {lineno}: invalid JSON ({exc})") from None
        if not isinstance(record, dict):
            raise ValueError(f"line {lineno}: not a JSON object")
        records.append(record)
    return records


def _group_chunks(records: list[dict]) -> dict[str, list[Chunk]]:
    """Validate records into Chunks grouped by video_id; ValueError if malformed."""
    grouped: dict[str, list[Chunk]] = {}
    for record in records:
        try:
            url = record["url"]
            text = record["text"]
            # Collapse whitespace so a queue record can't forge log lines or
            # ignore-list entries with an embedded newline in video_id.
            video_id = " ".join(str(record["video_id"]).split())
            chunk = Chunk(
                video_id=video_id,
                text=text,
                url=url,
                start_seconds=int(record["start_seconds"]),
                date=record.get("date"),
                episode=record.get("episode"),
                channel=record.get("channel"),
                # Story 6.1; old records lack it. Coerced because this is
                # untrusted queue input: a non-numeric value must degrade to
                # None, not poison SUM(duration) as SQLite TEXT.
                duration=duration_int(record.get("duration")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed record {record!r}: {exc}") from None
        # Producer emits YouTube watch URLs; anything else (or a missing/
        # oversized/empty text) is untrusted queue input — fail the file.
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError(
                f"record for {chunk.video_id!r}: url must be an http(s) URL, got {url!r}"
            )
        if not isinstance(text, str) or not text:
            raise ValueError(f"record for {chunk.video_id!r}: text must be a non-empty string")
        if len(text) > _MAX_TEXT_CHARS:
            raise ValueError(f"record for {chunk.video_id!r}: text exceeds {_MAX_TEXT_CHARS} chars")
        grouped.setdefault(chunk.video_id, []).append(chunk)
    return grouped


def _record_stats(state: StateStore, video_id: str, chunk: Chunk) -> None:
    """Best-effort archive-stats write from an applied video's first chunk
    (Story 6.3). Post-commit and non-fatal: apply already upserted the video,
    so a stats failure is logged and swallowed — the upsert is idempotent."""
    try:
        state.record_video_stats(
            video_id,
            duration=chunk.duration,
            episode=episode_int(chunk.episode),
            date=chunk.date,
        )
    except Exception:  # noqa: BLE001 — stats must never break apply
        logger.warning("archive-stats write failed for %s", video_id, exc_info=True)


def _quarantine(queue_dir: Path, path: Path, exc: ValueError) -> None:
    """Move a malformed increment into <queue_dir>/failed/ and log the error."""
    failed_dir = queue_dir / _FAILED_DIR
    failed_dir.mkdir(exist_ok=True)
    path.replace(failed_dir / path.name)
    logger.error("quarantined %s: %s", path.name, exc)


def run_apply(
    *,
    queue_dir: Path | None = None,
    state: StateStore | Path | None = None,
    client: QdrantClient | None = None,
    collection: str = config.COLLECTION_NAME,
    lock_path: Path | None = None,
) -> dict:
    """Apply queued increments. Returns a report dict (see cli `mc apply`)."""
    queue_dir = Path(queue_dir) if queue_dir is not None else config.QUEUE_DIR
    if state is None:
        state = StateStore(config.INGEST_DB_PATH)
    elif not isinstance(state, StateStore):
        state = StateStore(Path(state))
    if client is None:
        client = QdrantClient(url=config.QDRANT_URL, **client_kwargs(timeout=30))
    lock_path = Path(lock_path) if lock_path is not None else queue_dir.parent / _LOCK_NAME

    report: dict = {
        "applied_chunks": 0,
        "files_processed": 0,
        "failed_files": 0,
        "notes": [],
    }
    files = _queue_files(queue_dir)
    if not files:
        report["notes"].append("nothing to apply")
        return report

    with RunLock(lock_path):  # AlreadyRunning propagates to the caller (cron guard)
        for path in files:
            _apply_file(queue_dir, path, state, client, collection, report)

    if report["applied_chunks"] == 0 and report["failed_files"] == 0:
        report["notes"].append("nothing to apply")
    return report


def _apply_file(
    queue_dir: Path,
    path: Path,
    state: StateStore,
    client: QdrantClient,
    collection: str,
    report: dict,
) -> None:
    try:
        records = _parse_file(path)
        pairs = [
            (record, _content_hash(record))
            for record in records
            if state.get_key(f"{_APPLIED_PREFIX}{_content_hash(record)}") is None
        ]
        chunks_by_video = _group_chunks([record for record, _ in pairs])
    except ValueError as exc:
        _quarantine(queue_dir, path, exc)
        report["failed_files"] += 1
        return

    if not pairs:
        # every chunk in this file was already applied — nothing new to
        # upsert, but the videos must still enter Mode A's checkpoint (pre-fix
        # Mode B runs recorded content hashes without mark_processed)
        for video_id in {record["video_id"] for record in records}:
            state.mark_processed(video_id, run_type="apply")
        return

    applied_hashes: list[str] = []
    for video_id, chunks in sorted(chunks_by_video.items()):
        try:
            upsert_video(client, collection, video_id, chunks)
        except Exception:
            logger.exception("upsert failed for %s — %s left for retry", video_id, path)
            continue
        hashes = [h for record, h in pairs if record["video_id"] == video_id]
        for h in hashes:  # mark only AFTER the atomic per-video upsert committed
            state.record_key(f"{_APPLIED_PREFIX}{h}", _now_iso())
        applied_hashes.extend(hashes)
        _record_stats(state, video_id, chunks[0])  # Story 6.3: archive stats
        # Mode B's checkpoint: without this, a server-side `mc update` (Mode A)
        # sees the video as undiscovered and re-attempts it.
        state.mark_processed(video_id, run_type="apply")

    if applied_hashes:
        report["applied_chunks"] += len(applied_hashes)
        report["files_processed"] += 1
    else:
        report["notes"].append(f"{path.name}: no video applied — left for retry")
