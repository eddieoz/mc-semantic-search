"""Transcript download (Story 3.2 / 10.3): yt-dlp subtitle fetch with retry.

Ports the proven v3 pipeline logic into the mc package:

- language priority ['pt', 'pt-BR', 'en', 'en-US'], one language per attempt;
- json3 subtitle format; parsed transcript saved as ``<video_id>.srt`` (v3
  naming kept for consistency) and the temp json3 removed;
- 429 / "Too Many Requests" errors get exponential-backoff retries; once the
  budget is spent the download raises ``RateLimitedError`` (Story 10.3) —
  a 429 is IP-level, so trying the next language just burns more requests,
  and ignore-listing the video would hide it from every later run. The
  caller classifies it as transient and retries the video next run;
- a clean "no subtitles" result (yt-dlp wrote nothing) is the only path that
  appends to ``ignored_videos.txt``;
- a cached ``.srt`` short-circuits the download entirely.

Cookies come in as a parameter (config resolves ``MC_COOKIES_PATH`` via
``mc.config.COOKIES_PATH``) — no env access inside this module.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: YouTube video ids are 11 chars of [A-Za-z0-9_-]; the wider bound tolerates
#: future id shapes while still rejecting path separators and traversal.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: Language priority: Portuguese first, then English (v3 semantics).
LANGUAGE_PRIORITY: tuple[str, ...] = ("pt", "pt-BR", "en", "en-US")

IGNORE_LIST_FILENAME = "ignored_videos.txt"

#: Max backoff retries per language on rate-limit errors (v3 default).
MAX_RETRIES = 3


class DownloadError(RuntimeError):
    """Actionable failure before or during transcript download."""


class RateLimitedError(DownloadError):
    """YouTube kept answering 429 after the backoff budget (Story 10.3).

    Transient by nature: the caller must fail the run and retry the video on
    the next pass — never write it to ``ignored_videos.txt``."""


def _youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def parse_json3_subtitles(path: str | Path) -> list[dict]:
    """Parse a yt-dlp json3 subtitle file into transcript segments.

    Same output shape as v3: events with ``segs`` become
    ``{'text': ..., 'start': tStartMs/1000, 'duration': dDurationMs/1000}``;
    events whose joined text strips to empty are skipped.
    """
    transcript: list[dict] = []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for event in data.get("events", []):
        if "segs" not in event or event.get("tStartMs") is None:
            continue
        text = "".join(
            seg["utf8"] for seg in event["segs"] if "utf8" in seg
        ).strip()
        if not text:
            continue
        transcript.append(
            {
                "text": text,
                "start": event["tStartMs"] / 1000.0,
                "duration": event.get("dDurationMs", 0) / 1000.0,
            }
        )
    return transcript


def is_ignored(video_id: str, ignore_file: str | Path) -> bool:
    """True if ``video_id`` appears in the ignore list.

    Entries may carry a ``# reason`` suffix; matching is on the first
    whitespace token (v3 semantics).
    """
    path = Path(ignore_file)
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry and entry.split()[0] == video_id:
            return True
    return False


def add_to_ignore_list(
    video_id: str, ignore_file: str | Path, reason: str = ""
) -> None:
    """Append ``video_id`` to the ignore list once, with optional reason.

    Format mirrors v3: ``<id> # <reason>`` on a single line, no duplicates.
    Both fields are whitespace-collapsed so embedded newlines cannot forge
    extra ignore-list entries.
    """
    video_id = " ".join(video_id.split())
    reason = " ".join(reason.split())
    if is_ignored(video_id, ignore_file):
        return
    path = Path(ignore_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f" # {reason}" if reason else ""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{video_id}{suffix}\n")
    logger.info("added video %s to ignore list: %s", video_id, reason)


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc)
    return "429" in msg or "Too Many Requests" in msg


def _default_ytdlp_runner(url: str, ydl_opts: dict[str, Any]) -> None:
    """Run the yt-dlp Python API (no subprocess) with v3's option keys."""
    import yt_dlp

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def _build_ydl_opts(
    cookies_path: str, transcripts_dir: Path, lang: str
) -> dict[str, Any]:
    """yt-dlp options, same keys/semantics as the v3 implementation."""
    return {
        "cookiefile": cookies_path,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],  # only request THIS language
        "subtitlesformat": "json3",
        "skip_download": True,
        # We never need a media format. Without this, a run that yields zero
        # playable formats (n-challenge unsolved — no JS runtime on the prod
        # server) dies with "Requested format is not available" BEFORE the
        # subtitles are written, even though extraction itself succeeded.
        "ignore_no_formats_error": True,
        "outtmpl": str(transcripts_dir / "%(id)s.%(ext)s"),
        # Optimal sleep parameters per yt-dlp recommendations (v3 values).
        "sleep_requests": 1,
        "sleep_subtitles": 5,
        "sleep_interval": 3,
        "max_sleep_interval": 8,
        # Retry configuration (v3 values).
        "extractor_retries": 3,
        "fragment_retries": 3,
        "retry_sleep": 5,
        # False so download failures raise into _attempt_language's
        # classification path — True swallowed every error and made real
        # failures indistinguishable from "no subtitles" (permanent
        # ignore-list starvation).
        "ignoreerrors": False,
    }


def download_transcript(
    video_id: str,
    *,
    cookies_path: str,
    transcripts_dir: Path,
    max_retries: int = MAX_RETRIES,
    sleeper: Callable[[float], None] = time.sleep,
    runner: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[dict] | None:
    """Download (or load cached) transcript segments for one video.

    Returns parsed segments ``[{'text', 'start', 'duration'}, ...]`` from
    ``<transcripts_dir>/<video_id>.srt`` — downloading via yt-dlp when the
    cache is missing — or ``None`` when the video has no usable subtitles
    (in which case it is appended to ``ignored_videos.txt`` exactly once).
    Download failures other than missing subtitles propagate as exceptions
    so the caller (``run_update``) can classify them (e.g. stale cookies).
    """
    if not _VIDEO_ID_RE.fullmatch(video_id):
        # Defense in depth: this function builds <transcripts_dir>/<id>.srt,
        # so an unvalidated id (e.g. "../x") would write outside the dir.
        raise DownloadError(f"invalid video id: {video_id!r}")
    cookies = Path(cookies_path)
    if not cookies.exists():
        raise DownloadError(
            f"cookies file not found: {cookies_path!r}. Export cookies from "
            "your browser for youtube.com (e.g. with a cookie exporter "
            "extension) and pass the file path — see docs/tasks/cookies.md"
        )
    if cookies.stat().st_mode & 0o077:  # group/other-readable credential file
        logger.warning(
            "cookies file %s is group/other-readable — tightening to 0600",
            cookies,
        )
        cookies.chmod(0o600)

    run = runner if runner is not None else _default_ytdlp_runner
    transcripts_dir = Path(transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    ignore_file = transcripts_dir / IGNORE_LIST_FILENAME
    transcript_path = transcripts_dir / f"{video_id}.srt"

    if is_ignored(video_id, ignore_file):
        logger.info("skipping video %s (in ignore list)", video_id)
        return None

    if transcript_path.exists():
        logger.info("transcript for %s exists, loading from file", video_id)
        return json.loads(transcript_path.read_text(encoding="utf-8"))

    url = _youtube_url(video_id)
    for lang in LANGUAGE_PRIORITY:
        logger.info("attempting subtitles for %s in %s", video_id, lang)
        ydl_opts = _build_ydl_opts(cookies_path, transcripts_dir, lang)
        subtitle_file = transcripts_dir / f"{video_id}.{lang}.json3"
        segments = _attempt_language(
            run=run,
            url=url,
            ydl_opts=ydl_opts,
            subtitle_file=subtitle_file,
            transcript_path=transcript_path,
            max_retries=max_retries,
            sleeper=sleeper,
        )
        if segments is not None:
            logger.info(
                "successfully downloaded transcript for %s in %s", video_id, lang
            )
            return segments

    msg = (
        f"No subtitles found for video {video_id} in any supported "
        f"languages: {list(LANGUAGE_PRIORITY)}"
    )
    logger.warning(msg)
    add_to_ignore_list(video_id, ignore_file, "No subtitles available")
    return None


def _attempt_language(
    *,
    run: Callable[[str, dict[str, Any]], None],
    url: str,
    ydl_opts: dict[str, Any],
    subtitle_file: Path,
    transcript_path: Path,
    max_retries: int,
    sleeper: Callable[[float], None],
) -> list[dict] | None:
    """Try one language; backoff on 429; other failures raise to the caller.

    Rate-limit errors retry with backoff and then raise ``RateLimitedError``
    (Story 10.3): a 429 is IP-level, so the next language would be answered
    the same way, and ``run_update`` must retry the video tomorrow instead of
    ignore-listing it forever. Any other failure propagates so ``run_update``
    can classify it (auth-shaped errors are never ignore-listed); a merely
    subtitle-less video raises nothing — yt-dlp exits cleanly without writing
    the file.
    """
    try:
        run(url, ydl_opts)
    except Exception as exc:
        if not _is_rate_limit_error(exc):
            logger.warning("subtitle attempt failed for %s: %s", url, exc)
            raise
        logger.warning("rate limited, retrying with backoff")
        last_error: BaseException = exc
        for retry in range(max_retries):
            delay = (2**retry) * 10 + random.uniform(2, 15)
            logger.info("retry %d/%d — waiting %.1fs", retry + 1, max_retries, delay)
            sleeper(delay)
            try:
                run(url, ydl_opts)
            except Exception as retry_exc:
                if not _is_rate_limit_error(retry_exc):
                    raise
                last_error = retry_exc
                continue
            return _consume_subtitle(subtitle_file, transcript_path)
        raise RateLimitedError(
            f"still rate limited after {max_retries} retries for {url}: {last_error}"
        ) from last_error
    return _consume_subtitle(subtitle_file, transcript_path)


def _consume_subtitle(
    subtitle_file: Path, transcript_path: Path
) -> list[dict] | None:
    """Parse the json3 file if it appeared; cache as .srt and clean up."""
    if not subtitle_file.exists():
        return None
    transcript = parse_json3_subtitles(subtitle_file)
    # Saved under the .srt name like v3, even though the content is JSON.
    transcript_path.write_text(
        json.dumps(transcript), encoding="utf-8"
    )
    subtitle_file.unlink()
    return transcript
