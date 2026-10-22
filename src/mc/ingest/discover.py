"""Channel video discovery (Story 3.1): yt-dlp --flat-playlist listing.

Channel list is configuration (env MC_CHANNELS / config.yaml) — never
hardcoded. Each channel is listed across its /videos, /streams and /shorts
tabs and the results are unioned: Morning Crypto episodes are live streams
that only appear under /streams. Output parsing is strict: malformed yt-dlp
output is a clear DiscoveryError, not a silent empty set.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

#: Only real YouTube video ids (exactly 11 chars) become path components /
#: URL fragments. A degraded listing (no cookies) can surface playlist-type
#: entries whose id is the 24-char channel id — those must never reach the
#: pipeline.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

#: Channel tabs listed for every configured channel, in a stable order.
_TAB_PATHS = ("/videos", "/streams", "/shorts")

#: yt-dlp's error for a tab a channel does not publish (e.g. no /streams).
_TAB_ABSENT_RE = re.compile(r"does not have a \w+ tab", re.IGNORECASE)

_YTDLP_TIMEOUT_S = 120


class DiscoveryError(RuntimeError):
    pass


def _channel_tabs(url: str) -> tuple[str, ...]:
    """Listing URLs for one channel: its videos, streams and shorts tabs.

    A configured URL that already points at a tab is listed as-is, never
    double-appended."""
    stripped = url.rstrip("/")
    if stripped.endswith(_TAB_PATHS):
        return (stripped,)
    return tuple(f"{stripped}{tab}" for tab in _TAB_PATHS)


def load_channels() -> tuple[str, ...]:
    """Channels from config.yaml, falling back to the MC_CHANNELS env."""
    import os

    import yaml

    from mc.config import PROJECT_ROOT

    cfg = PROJECT_ROOT / "config.yaml"
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text()) or {}
        channels = data.get("channels")
        if channels:
            return tuple(str(c) for c in channels)
    raw = os.environ.get("MC_CHANNELS", "")
    return tuple(c.strip() for c in raw.split(",") if c.strip())


def _run_ytdlp(listing_url: str) -> subprocess.CompletedProcess:
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        "--no-warnings",
        "--playlist-end",
        "50",
    ]
    # Without cookies YouTube serves degraded layouts whose "entries" are
    # playlist references rather than videos — same file the rest of the
    # pipeline uses, resolved at call time so config stays monkeypatchable.
    from mc.config import COOKIES_PATH

    if Path(COOKIES_PATH).exists():
        cmd += ["--cookies", str(COOKIES_PATH)]
    cmd.append(listing_url)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=_YTDLP_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise DiscoveryError(
            f"yt-dlp timed out after {_YTDLP_TIMEOUT_S}s for {listing_url}"
        ) from exc


def list_channel_videos(channels: tuple[str, ...] | list[str]) -> set[str]:
    """Current video IDs across all configured channels and all their tabs."""
    videos: set[str] = set()
    for channel in channels:
        for tab_url in _channel_tabs(channel):
            result = _run_ytdlp(tab_url)
            if result.returncode != 0:
                stderr = result.stderr.strip()
                # Not every channel has every tab — an absent tab is normal,
                # everything else (bot gate, network) fails loudly.
                if _TAB_ABSENT_RE.search(stderr):
                    logger.warning("channel %s has no %s tab — skipped", channel, tab_url)
                    continue
                raise DiscoveryError(f"yt-dlp failed for {tab_url}: {stderr}")
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise DiscoveryError(
                    f"yt-dlp --flat-playlist output for {tab_url} is not JSON: "
                    f"{result.stdout[:200]!r}"
                ) from exc
            entries = data.get("entries")
            if entries is None:
                raise DiscoveryError(f"flat-playlist output for {tab_url} has no 'entries' key")
            for entry in entries:
                if not isinstance(entry, dict) or not entry.get("id"):
                    raise DiscoveryError(f"malformed playlist entry for {tab_url}: {entry!r}")
                video_id = str(entry["id"])
                if not _VIDEO_ID_RE.fullmatch(video_id):
                    # Traversal-shaped or non-video ids (e.g. the channel id
                    # of a playlist-type entry) must never reach the
                    # f"{video_id}.srt" path building — skip and log.
                    logger.warning(
                        "skipping entry with invalid video id for %s: %r",
                        tab_url,
                        video_id,
                    )
                    continue
                videos.add(video_id)
            logger.info("channel tab %s: %d entries", tab_url, len(entries))
    return videos
