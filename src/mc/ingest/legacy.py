"""Story 2.1 — reader for the legacy SQLite FTS5 transcript database.

The legacy ``mc_transcripts.db`` stores one row per chunk in an FTS5 table
``transcripts(title, url, description)`` where ``description`` holds the chunk
text and ``url`` embeds the video id and the start timestamp
(e.g. ``https://www.youtube.com/watch?v=aqaFdI8T1lQ&t=1234``).
Timestamps may be plain seconds (``t=1234``), seconds with a suffix
(``t=1234s``) or minutes plus seconds (``t=20m30s``).
"""

from __future__ import annotations

import csv
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from mc.ingest.models import Chunk

_WATCH_RE = re.compile(r"watch\?v=([A-Za-z0-9_-]{11})")
_SHORT_RE = re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})")
_T_PARAM_RE = re.compile(r"[?&]t=([^&#]*)")
_MIN_SEC_RE = re.compile(r"^(\d+)m(\d{1,2})s?$")
_SEC_RE = re.compile(r"^(\d+)s?$")

_QUARANTINE_HEADER = ["video_id_or_url", "reason"]

# A cell starting with one of these executes as a formula when the quarantine
# CSV is opened in Excel/LibreOffice — prefix it so it is treated as text.
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _defuse_formula(cell: str) -> str:
    """Prefix formula-looking CSV cells with a single quote (spreadsheet safety)."""
    if isinstance(cell, str) and cell.startswith(_FORMULA_PREFIXES):
        return f"'{cell}"
    return cell


def parse_video_id(url: str) -> str | None:
    """Return the 11-char YouTube video id from a url, or ``None``."""
    match = _WATCH_RE.search(url) or _SHORT_RE.search(url)
    return match.group(1) if match else None


def parse_start_seconds(url: str) -> int | None:
    """Parse the ``t=`` parameter into seconds.

    Handles ``t=1234``, ``t=1234s`` and ``t=20m30s``; a missing/empty ``t``
    means the video start (0). Returns ``None`` when the value is present but
    unparseable.
    """
    match = _T_PARAM_RE.search(url)
    if match is None:
        return 0
    value = match.group(1).strip()
    if not value:
        return 0
    if min_sec := _MIN_SEC_RE.match(value):
        return int(min_sec.group(1)) * 60 + int(min_sec.group(2))
    if sec := _SEC_RE.match(value):
        return int(sec.group(1))
    return None


def _append_quarantine(url: str, video_id: str | None, reason: str, out_path: Path | None) -> None:
    """Append one unparseable row to the quarantine CSV, if requested."""
    if out_path is None:
        return
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(_QUARANTINE_HEADER)
        writer.writerow(
            [
                _defuse_formula(video_id if video_id is not None else url),
                _defuse_formula(reason),
            ]
        )


def iter_legacy_chunks(db_path: Path, quarantine_path: Path | None = None) -> Iterator[Chunk]:
    """Yield one :class:`Chunk` per parseable legacy row, read-only.

    Rows whose url has no extractable video id, or an unparseable ``t=``
    value, are appended to ``quarantine_path`` (when given) and skipped.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT url, description FROM transcripts")
        for url, description in rows:
            video_id = parse_video_id(url)
            if video_id is None:
                _append_quarantine(url, None, "no_video_id", quarantine_path)
                continue
            start_seconds = parse_start_seconds(url)
            if start_seconds is None:
                _append_quarantine(url, video_id, "unparseable_timestamp", quarantine_path)
                continue
            yield Chunk(
                video_id=video_id,
                text=description,
                url=url,
                start_seconds=start_seconds,
            )
    finally:
        con.close()


def iter_legacy_titles(db_path: Path) -> Iterator[tuple[str, str]]:
    """Yield ``(video_id, title)`` once per distinct video, read-only.

    The first row's title wins (chunks of one video share it). Feeds the
    stats-backfill ``--from-db`` episode re-derivation: the chunked title is
    the authoritative episode source, so a video whose title is absent or
    empty is simply not overridden.
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        seen: set[str] = set()
        for url, title in con.execute("SELECT url, title FROM transcripts"):
            video_id = parse_video_id(url)
            if video_id is None or video_id in seen or not title:
                continue
            seen.add(video_id)
            yield video_id, title
    finally:
        con.close()


def count_legacy(db_path: Path) -> dict:
    """Dry-run counts: total rows, parsed chunks, distinct videos, quarantined."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        total = con.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        distinct_videos: set[str] = set()
        parsed = 0
        for (url,) in con.execute("SELECT url FROM transcripts"):
            video_id = parse_video_id(url)
            if video_id is None or parse_start_seconds(url) is None:
                continue
            parsed += 1
            distinct_videos.add(video_id)
    finally:
        con.close()
    return {
        "total_rows": total,
        "parsed_chunks": parsed,
        "distinct_videos": len(distinct_videos),
        "quarantined": total - parsed,
    }
