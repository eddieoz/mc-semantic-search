"""Channel video discovery (Story 3.1 / 10.1).

Channel list is configuration (env MC_CHANNELS / config.yaml) — never
hardcoded.

Story 10.1 (NFR-3): a tab listing is a *bounded window* (``--playlist-end``
entries per tab), so any video older than the window is invisible to every
run — that is how ~150 published videos never reached the archive. When a
YouTube Data API key is configured, discovery walks the channel's whole
uploads playlist instead (page after page, no window); the tabs remain the
fallback for an API outage, and a tab that returns the entire cap is a loud
``DiscoveryError`` (a possibly truncated listing must never be ingested
silently). Each channel is listed across its /videos, /streams and /shorts
tabs and the results are unioned: Morning Crypto episodes are live streams
that only appear under /streams. Output parsing is strict: malformed yt-dlp
output is a clear DiscoveryError, not a silent empty set.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Callable
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


def _run_ytdlp(listing_url: str, playlist_end: int | None = None) -> subprocess.CompletedProcess:
    """List one channel tab with yt-dlp, capped at ``playlist_end`` entries.

    The cap is read from ``config.DISCOVERY_TAB_LIMIT`` at call time (not
    imported as a value) so operators and tests can tune it without a
    reimport."""
    from mc.config import COOKIES_PATH, DISCOVERY_TAB_LIMIT

    cap = DISCOVERY_TAB_LIMIT if playlist_end is None else playlist_end
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        "--no-warnings",
        "--playlist-end",
        str(cap),
    ]
    # Without cookies YouTube serves degraded layouts whose "entries" are
    # playlist references rather than videos — same file the rest of the
    # pipeline uses, resolved at call time so config stays monkeypatchable.
    if Path(COOKIES_PATH).exists():
        cmd += ["--cookies", str(COOKIES_PATH)]
    cmd.append(listing_url)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=_YTDLP_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise DiscoveryError(
            f"yt-dlp timed out after {_YTDLP_TIMEOUT_S}s for {listing_url}"
        ) from exc


def _list_via_api(
    channels: tuple[str, ...] | list[str],
    api_key: str,
    transport: Callable[[str, dict], dict] | None,
) -> set[str]:
    """Walk every channel's uploads playlist (Story 10.1) — no window.

    Imported lazily: ``youtube_api`` imports ``_VIDEO_ID_RE`` from here, so a
    module-level import would be circular."""
    from mc.ingest.youtube_api import fetch_channel_uploads, iter_playlist_videos

    uploads = fetch_channel_uploads(api_key, transport=transport, channels=channels)
    videos: set[str] = set()
    for channel_url, playlist_id in uploads.items():
        count = 0
        for video_id in iter_playlist_videos(api_key, playlist_id, transport=transport):
            videos.add(video_id)
            count += 1
        logger.info(
            "channel %s: %d uploads (uploads playlist %s)",
            channel_url,
            count,
            playlist_id,
        )
    return videos


def _list_via_tabs(channels: tuple[str, ...] | list[str]) -> set[str]:
    """Fallback discovery: the /videos, /streams, /shorts tabs, capped."""
    from mc.config import DISCOVERY_TAB_LIMIT

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
            if len(entries) >= DISCOVERY_TAB_LIMIT:
                # A tab that fills the cap may have more behind it. Ingesting
                # a silently truncated listing is the exact bug Story 10.1
                # fixes, so it must be loud — and it names the knob.
                raise DiscoveryError(
                    f"tab {tab_url} returned {len(entries)} entries, the whole "
                    f"window cap (MC_DISCOVERY_TAB_LIMIT={DISCOVERY_TAB_LIMIT}) — "
                    "the listing may be truncated; set YOUTUBE_API_KEY for full "
                    "enumeration or raise the cap"
                )
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


def list_channel_videos(
    channels: tuple[str, ...] | list[str],
    *,
    api_key: str | None = None,
    transport: Callable[[str, dict], dict] | None = None,
    use_api: bool = True,
) -> set[str]:
    """Current video IDs across all configured channels and all their tabs.

    API-first (Story 10.1): with a Data API key, the channel's whole uploads
    playlist is enumerated. Any API failure degrades to the tab path with a
    warning — an API outage must not lose the run. An API result that is
    empty is also treated as a failure of the API path (a renamed handle must
    not look like an empty archive)."""
    if api_key is None:
        import mc.config as config

        api_key = config.YOUTUBE_API_KEY
    if use_api and api_key:
        try:
            videos = _list_via_api(channels, api_key, transport)
        except Exception as exc:  # noqa: BLE001 — API outage degrades to tabs
            logger.warning(
                "YouTube Data API discovery failed (%s) — falling back to yt-dlp tabs",
                exc,
            )
        else:
            if videos:
                return videos
            logger.warning(
                "YouTube Data API discovery returned no videos — falling back to yt-dlp tabs"
            )
    return _list_via_tabs(channels)
