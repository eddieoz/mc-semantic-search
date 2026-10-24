"""Story 9.1 + 9.2 — YouTube Data API v3 fact parsing and transport client.

Story 9.1 — pure parsers that turn the API's raw fact formats into the
internal ones, so the rest of the pipeline (chunk payloads, ``video_stats``,
Mode B queue records) never sees a YouTube-specific string:

- ``parse_duration_iso``: ``contentDetails.duration`` (ISO-8601 ``PT#H#M#S``)
  → total seconds, matching the int-or-``None`` convention of
  ``metadata.duration_int`` (Story 6.1).
- ``published_at_to_upload_date``: ``snippet.publishedAt`` → the zero-padded
  ``YYYYMMDD`` string that ``metadata._normalize`` (``%Y%m%d`` strptime)
  expects (decision **D17** — the UTC date component is truncated as written,
  never timezone-shifted: a shift would move late-night uploads to the
  "wrong" day relative to what YouTube displays in Brazil; truncation is the
  honest, format-consistent choice).

Story 9.2 — the thin transport client (channels → playlistItems →
videos.list) behind ``mc stats-backfill --from-api``: the whole archive's
metadata costs ~150 quota units instead of ~3,600 authenticated scrapes:

- ``fetch_channel_uploads``: resolves each configured channel URL to its
  uploads playlist id (1 unit/channel).
- ``iter_playlist_videos``: walks a playlist page by page, yielding
  11-char video ids (1 unit/page).
- ``fetch_video_facts``: batch-fetches per-video facts (1 unit/batch).

Quota accounting — **design choice**: a tiny mutable ``QuotaCounter`` passed
in by the caller (Story 9.3 threads one counter through the whole run so the
report can show the true cost). When omitted, each function creates a
throwaway counter so the same accounting code path always runs; the total is
simply not observed. A counter object was chosen over return-value totals
because Story 9.3 composes all three functions in one pass and would
otherwise have to thread partial totals through its own accumulators.

Transport — **injectable** (D16): every client function takes a callable
``(url, params) -> dict`` returning the parsed JSON body; the default is
``_httpx_get`` (plain httpx GETs — no google-api-python-client, D16). Tests
inject a fake; no test ever touches the network.

Invariants served:

- **NFR-3** — pure, deterministic parsing plus idempotent fact values keep
  re-ingest/backfill idempotent: the same API facts always produce the same
  stored values.
- **NFR-5** — the API path uses an API key, never cookies; the key travels
  in the ``key`` query param inside ``params`` and is **never logged**.
- **NFR-7** — no new dependency: httpx (already pinned) + stdlib only.
- **D18** — quota exhaustion (HTTP 403 ``quotaExceeded``) raises
  ``QuotaExceededError`` and other HTTP errors raise ``YouTubeApiError``:
  both fail loud. Missing data (empty channel resolution, 404 playlist,
  deleted/private videos) is *not* an error — it is logged and skipped,
  mirroring Story 8.1's absent-tab tolerance.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

import httpx

from mc.ingest.discover import _VIDEO_ID_RE

logger = logging.getLogger(__name__)

# ISO-8601 duration as the API emits it: PT#H#M#S with optional components.
# P0D (deleted/live placeholder) fails to match because "D" is not accepted —
# a day-based duration is not a real video length, only a placeholder.
_ISO_DURATION_RE = re.compile(
    r"^PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?$"
)

# publishedAt comes as an ISO-8601 timestamp; the API returns the Zulu form
# but +00:00 spelling is accepted for symmetry with other ISO producers.
_PUBLISHED_AT_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z")


def parse_duration_iso(value: str | None) -> int | None:
    """ISO-8601 ``PT#H#M#S`` → total seconds; anything unparseable → ``None``.

    ``"P0D"`` (the placeholder YouTube returns for deleted/upcoming videos),
    empty and malformed input yield ``None`` — callers store absence, never a
    bogus 0, so archive duration aggregates stay truthful (NFR-3).
    """
    if not value:
        return None
    match = _ISO_DURATION_RE.match(value)
    if match is None:
        return None
    parts = match.groupdict()
    # Bare "PT" (no H/M/S component at all) is not a duration — reject it;
    # only PT0S and friends are a legitimate zero (NFR-3: no bogus values).
    if not any(parts.values()):
        return None
    return (
        int(parts["h"] or 0) * 3600
        + int(parts["m"] or 0) * 60
        + int(parts["s"] or 0)
    )


def published_at_to_upload_date(value: str | None) -> str | None:
    """``publishedAt`` → zero-padded ``YYYYMMDD`` (UTC truncate, D17) or ``None``.

    No timezone shift is applied (D17): the date component is taken exactly as
    the API wrote it, matching what YouTube displays. ``None``/empty/unparseable
    input → ``None``.
    """
    if not value:
        return None
    text = value.strip()
    for fmt in _PUBLISHED_AT_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        # D17: strftime the date component as written (no tz shift); the
        # %z spelling parses only to validate, its offset is never applied.
        # Zero-padding comes from strftime, so 2021-03-05 → "20210305".
        return parsed.strftime("%Y%m%d")
    return None


# ---------------------------------------------------------------------------
# Story 9.2 — transport client (channels → playlistItems → videos.list)
# ---------------------------------------------------------------------------

CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
PLAYLIST_ITEMS_URL = "https://www.googleapis.com/youtube/v3/playlistItems"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

#: API page/batch size — the endpoints' maximum; fewer units per archive walk.
_PAGE_SIZE = 50

#: httpx GET timeout (s) — generous but bounded: a hung API call must not
#: stall the cron run forever.
_HTTP_TIMEOUT_S = 30.0

#: Tab suffixes a configured channel URL may carry (same list as
#: ``discover._TAB_PATHS``); stripped before handle/id extraction.
_TAB_SUFFIXES = ("/videos", "/streams", "/shorts")

#: A real YouTube channel id is "UC" + exactly 22 id chars. Strict on purpose:
#: a `/c/` vanity name that merely starts with "UC" must not be misrouted to
#: the ``id=`` query (it would silently resolve to nothing).
_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")

#: Transport callable: ``(url, params) -> parsed JSON dict`` (D16). The API
#: key rides inside ``params`` (``key=``) and must never be logged (NFR-5).
Transport = Callable[[str, dict], dict]


class YouTubeApiError(RuntimeError):
    """A YouTube Data API HTTP failure other than quota exhaustion (D18).

    ``status`` carries the HTTP status code (0 when the transport could not
    determine one) so callers can special-case statuses — e.g. a 404 playlist
    — without parsing the message text.
    """

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class QuotaExceededError(YouTubeApiError):
    """The project's daily YouTube API quota is exhausted (D18 — fail loud).

    Subclasses ``YouTubeApiError`` so a bare ``except YouTubeApiError``
    catches both, but the 404-tolerant paths must re-raise it: quota silence
    would hide a misconfigured deployment.
    """


def _error_from_payload(status: int, body: dict) -> YouTubeApiError:
    """Map an API HTTP status + JSON body to this module's exception types.

    Lives outside the transport so the status→exception mapping is testable
    without network: the default transport calls it, unit tests exercise it
    directly with canned error envelopes.
    """
    reason = ""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        errors = error.get("errors")
        if isinstance(errors, list) and errors and isinstance(errors[0], dict):
            reason = str(errors[0].get("reason") or "")
        if not reason:
            # Some envelopes carry only error.status ("PERMISSION_DENIED").
            reason = str(error.get("status") or "")
    if status == 403 and reason == "quotaExceeded":
        return QuotaExceededError(
            f"YouTube API quota exhausted: HTTP 403, reason={reason}", status=403
        )
    return YouTubeApiError(
        f"YouTube API request failed: HTTP {status} ({reason or 'no reason given'})",
        status=status,
    )


def _httpx_get(url: str, params: dict) -> dict:
    """Default transport: one httpx GET; parsed JSON in, mapped errors out.

    Logs nothing at all — ``params`` carries the API key (NFR-5).
    """
    response = httpx.get(url, params=params, timeout=_HTTP_TIMEOUT_S)
    if response.is_error:
        try:
            body: dict = response.json()
        except ValueError:
            body = {}
        raise _error_from_payload(response.status_code, body)
    return response.json()


@dataclass
class QuotaCounter:
    """Running total of API quota units consumed (1 unit per HTTP call).

    Client functions accept an optional shared counter (Story 9.3 threads one
    through the run so the report shows the true cost). When omitted, a
    throwaway counter is created per call — the accounting code path is
    identical, the total is simply not observed.
    """

    units: int = 0

    def charge(self, units: int = 1) -> None:
        self.units += units


def _channel_query(url: str) -> tuple[str, str] | None:
    """Map a configured channel URL to a ``channels.list`` query parameter.

    Returns ``("id", "UC…")`` for the ``/channel/UC…`` form and
    ``("forHandle", "@…")`` for ``/@handle`` plus the legacy ``/c/`` and
    ``/user/`` forms (their vanity name *is* the handle). A trailing tab
    suffix (``/videos`` etc., as ``discover`` appends them) is stripped
    first. ``None`` when the URL is not a recognizable channel form — the
    caller logs it and skips (configuration noise gets 8.1's tolerance, and
    no quota is spent on it).
    """
    path = urlparse(url).path
    for tab in _TAB_SUFFIXES:
        if path.endswith(tab):
            path = path[: -len(tab)]
            break
    segments = [segment for segment in path.split("/") if segment]
    if not segments:
        return None
    last = segments[-1]
    if last.startswith("@"):
        return ("forHandle", last)
    if _CHANNEL_ID_RE.match(last):
        return ("id", last)
    if len(segments) >= 2 and segments[-2] in ("c", "user"):
        return ("forHandle", f"@{last}")
    return None


def fetch_channel_uploads(
    api_key: str,
    transport: Transport | None = None,
    channels: tuple[str, ...] | list[str] = (),
    quota: QuotaCounter | None = None,
) -> dict[str, str]:
    """Resolve each configured channel URL to its uploads playlist id.

    ``channels.list?part=contentDetails`` per channel (1 quota unit each) →
    ``items[0].contentDetails.relatedPlaylists.uploads`` (the API already
    returns the ``UU…`` id — used as-is). Channel URLs come from
    ``discover.load_channels()`` (config.yaml / MC_CHANNELS).

    A channel that resolves to nothing (renamed/deleted, empty ``items``) is
    logged and absent from the mapping — the run continues (Story 8.1's
    absent-tab tolerance). Transport errors propagate (D18). Returns a dict
    keyed by the original channel URL → uploads playlist id.
    """
    transport = transport or _httpx_get
    quota = quota or QuotaCounter()
    uploads: dict[str, str] = {}
    for channel_url in channels:
        query = _channel_query(channel_url)
        if query is None:
            logger.warning("unrecognized channel URL, skipping: %s", channel_url)
            continue
        # 1 quota unit per channels.list call.
        quota.charge(1)
        data = transport(
            CHANNELS_URL,
            {"part": "contentDetails", query[0]: query[1], "key": api_key},
        )
        items = data.get("items") or []
        if not items:
            logger.warning(
                "channel resolved to nothing, continuing without it: %s", channel_url
            )
            continue
        playlist = (
            items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        )
        if not playlist:
            logger.warning(
                "channel has no uploads playlist, continuing without it: %s",
                channel_url,
            )
            continue
        uploads[channel_url] = playlist
    return uploads


def iter_playlist_videos(
    api_key: str,
    playlist_id: str,
    transport: Transport | None = None,
    quota: QuotaCounter | None = None,
) -> Iterator[str]:
    """Yield the video ids of a playlist, page by page (1 quota unit/page).

    ``playlistItems.list?part=contentDetails&maxResults=50`` walking
    ``nextPageToken``. Each id is validated with ``discover._VIDEO_ID_RE``
    (exactly 11 chars); invalid ids are logged and skipped, never yielded.

    HTTP 404 on the playlist means "no public uploads" — returns without
    yielding (the caller reports it, it is not fatal). Quota exhaustion and
    other HTTP errors propagate (D18).
    """
    transport = transport or _httpx_get
    quota = quota or QuotaCounter()
    page_token: str | None = None
    while True:
        params: dict = {
            "part": "contentDetails",
            "maxResults": _PAGE_SIZE,
            "playlistId": playlist_id,
            "key": api_key,
        }
        if page_token is not None:
            params["pageToken"] = page_token
        # 1 quota unit per playlistItems page.
        quota.charge(1)
        try:
            data = transport(PLAYLIST_ITEMS_URL, params)
        except YouTubeApiError as exc:
            if exc.status == 404:
                # Playlist gone/private: "no public uploads". Not fatal.
                # QuotaExceededError (403) cannot land here — re-raised below.
                logger.warning("playlist %s not found — no public uploads", playlist_id)
                return
            raise
        for item in data.get("items") or []:
            video_id = (item.get("contentDetails") or {}).get("videoId")
            if not video_id or not _VIDEO_ID_RE.fullmatch(video_id):
                logger.warning(
                    "skipping invalid video id in playlist %s: %r", playlist_id, video_id
                )
                continue
            yield video_id
        page_token = data.get("nextPageToken")
        if not page_token:
            return


def fetch_video_facts(
    api_key: str,
    video_ids: Sequence[str],
    transport: Transport | None = None,
    quota: QuotaCounter | None = None,
) -> tuple[dict[str, dict], list[str]]:
    """Batch-fetch per-video facts: ``{upload_date, title, duration}`` per id.

    ``videos.list?part=snippet,contentDetails`` in batches of 50 ids (1 quota
    unit per call). ``upload_date`` and ``duration`` go through Story 9.1's
    parsers; unparseable facts are stored as ``None`` (NFR-3: never a bogus
    value). Returns ``(facts, missing_ids)``: ids absent from any response
    are deleted/private (coverage contract #4) — counted as ``still_missing``
    in request order, never retried in-run, never fatal. Empty input returns
    ``({}, [])`` without a call.
    """
    if not video_ids:
        return {}, []
    transport = transport or _httpx_get
    quota = quota or QuotaCounter()
    facts: dict[str, dict] = {}
    requested: list[str] = []
    for offset in range(0, len(video_ids), _PAGE_SIZE):
        batch = list(video_ids[offset : offset + _PAGE_SIZE])
        requested.extend(batch)
        # 1 quota unit per videos.list batch (≤50 ids per call).
        quota.charge(1)
        data = transport(
            VIDEOS_URL,
            {
                "part": "snippet,contentDetails",
                "maxResults": _PAGE_SIZE,
                "id": ",".join(batch),
                "key": api_key,
            },
        )
        for item in data.get("items") or []:
            snippet = item.get("snippet") or {}
            details = item.get("contentDetails") or {}
            facts[item.get("id")] = {
                "upload_date": published_at_to_upload_date(snippet.get("publishedAt")),
                "title": snippet.get("title"),
                "duration": parse_duration_iso(details.get("duration")),
            }
    missing = [video_id for video_id in requested if video_id not in facts]
    return facts, missing
