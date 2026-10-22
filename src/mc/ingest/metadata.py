"""Story 2.2 — yt-dlp metadata enrichment with a resumable on-disk cache.

One JSON file per video under the metadata cache dir (config.META_CACHE_DIR by
default), so a re-run never re-invokes yt-dlp for an id it already enriched.
Videos that fail enrichment after the retries end up in ``missing.txt`` and are
simply omitted from the returned mapping; the rest of the batch continues.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from mc import config

logger = logging.getLogger(__name__)

Fetcher = Callable[[str], dict]

_YOUTUBE_URL = "https://www.youtube.com/watch?v={video_id}"
_EPISODE_RE = r"#(\d+)"
#: ``[ Morning Crypto ][ ep 441 ]``, ``EP. 372`` — the archive's real marker
#: (case-insensitive; a doubled space between the word and the number).
_EP_WORD_RE = re.compile(r"\bep\.?\s*(\d+)", re.IGNORECASE)
_MAX_ATTEMPTS = 3
_YTDLP_TIMEOUT_S = 120


def duration_int(value: object) -> int | None:
    """Whole seconds as an int for storage/aggregates (Story 6.1).

    yt-dlp returns an int or float of seconds (it sometimes yields ``"N/A"``),
    and a Mode B queue record is untrusted input, so coercion happens in ONE
    place: anything absent/non-numeric → ``None`` (never a bogus 0). Keeping a
    junk value out of ``video_stats.duration`` matters — SQLite would store a
    string as TEXT and ``SUM(duration)`` would then count it as 0, silently
    under-reporting the archive's total duration.
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError: int(float("inf")); ValueError: int(float("nan")).
        return None


def _duration_seconds(raw: dict) -> int | None:
    """yt-dlp's ``duration`` in whole seconds (Story 6.1)."""
    return duration_int(raw.get("duration"))


def _default_fetch(video_id: str) -> dict:
    """Fetch raw metadata for one video via the pinned yt-dlp dependency.

    ``--ignore-no-formats-error`` keeps the metadata dump alive when YouTube's
    bot gating leaves a video with no playable formats (storyboards only) —
    the extractor still knows title/date/duration, and the default format
    selector would otherwise abort the whole dump. The conventional cookies
    file (``MC_COOKIES_PATH``, the download pipeline's file) is passed when it
    exists. A failed fetch raises with yt-dlp's stderr tail so the operator
    can tell a bot gate from a dead video, not just "exit status 1".
    """
    url = _YOUTUBE_URL.format(video_id=video_id)
    cmd = [
        "yt-dlp",
        "--dump-single-json",
        "--no-warnings",
        "--ignore-no-formats-error",
    ]
    cookies = Path(config.COOKIES_PATH)
    if cookies.exists():
        if cookies.stat().st_mode & 0o077:  # group/other-readable credential file
            logger.warning(
                "cookies file %s is group/other-readable — tightening to 0600",
                cookies,
            )
            cookies.chmod(0o600)
        cmd += ["--cookies", str(cookies)]
    cmd.append(url)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=_YTDLP_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"yt-dlp metadata fetch timed out after {_YTDLP_TIMEOUT_S}s") from exc
    except subprocess.CalledProcessError as exc:
        stderr_tail = ((exc.stderr or "").strip())[-400:]
        raise RuntimeError(
            f"yt-dlp metadata fetch failed for {video_id}: {stderr_tail or exc}"
        ) from exc
    return json.loads(proc.stdout)


def _parse_episode(title: str) -> str | None:
    """Episode number from the title: ``#N`` or the ``ep N`` marker.

    There is deliberately NO bare-digit fallback: a title's leftover digit
    runs are dates, prices and milestones, never episodes ("[ IRL ][ ep 567 ]
    DeSci Nordics 2024" must parse as 567, not 2024; "Nunca use 123456" has
    no episode at all). A title that declares no episode parses as None.
    """
    match = re.search(_EPISODE_RE, title) or _EP_WORD_RE.search(title)
    return match.group(1) if match else None


def episode_int(episode: str | int | None) -> int | None:
    """Episode as a plain int for numeric storage/ordering (Story 6.2).

    ``"971"`` → ``971``; garbage, blanks or ``None`` → ``None``. Writers feed
    this to ``record_video_stats`` so ``MAX(episode)`` is numeric, never
    lexical (``1000`` beats ``999`` even though ``"1000" < "999"``).
    """
    if episode is None:
        return None
    try:
        return int(str(episode).strip())
    except (TypeError, ValueError):
        return None


def _normalize(raw: dict) -> dict:
    upload_date = raw.get("upload_date") or ""
    try:
        date = datetime.strptime(upload_date, "%Y%m%d").date().isoformat()
    except ValueError:
        date = None
    return {
        "date": date,
        "episode": _parse_episode(raw.get("title") or ""),
        "channel": raw.get("uploader") or raw.get("channel"),
        "duration": _duration_seconds(raw),
    }


def enrich(
    video_ids: list[str],
    *,
    cache_dir: Path | None = None,
    fetch: Fetcher | None = None,
    backoff: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, dict]:
    """Enrich video ids with {date, episode, channel}, cached on disk.

    Duplicated ids are fetched once; ids already cached are loaded from disk
    without invoking the fetcher. Fetch failures are retried with exponential
    backoff (``backoff`` delays); after the final failure the id is appended to
    ``missing.txt`` and omitted from the result.
    """
    cache = Path(cache_dir) if cache_dir is not None else config.META_CACHE_DIR
    fetcher = fetch if fetch is not None else _default_fetch
    cache.mkdir(parents=True, exist_ok=True)

    result: dict[str, dict] = {}
    for video_id in dict.fromkeys(video_ids):
        cache_file = cache / f"{video_id}.json"
        if cache_file.exists():
            result[video_id] = json.loads(cache_file.read_text(encoding="utf-8"))
            continue

        raw: dict | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                raw = fetcher(video_id)
                break
            except Exception:
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(backoff[min(attempt, len(backoff) - 1)])
        if raw is None:
            with (cache / "missing.txt").open("a", encoding="utf-8") as fh:
                fh.write(f"{video_id}\n")
            continue

        meta = _normalize(raw)
        cache_file.write_text(json.dumps(meta), encoding="utf-8")
        result[video_id] = meta
    return result
