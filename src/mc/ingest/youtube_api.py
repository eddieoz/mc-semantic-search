"""Story 9.1 — ISO-8601 fact parsing for the YouTube Data API v3 (NFR-3/5/7).

Pure parsers that turn the API's raw fact formats into the internal ones, so
the rest of the pipeline (chunk payloads, ``video_stats``, Mode B queue
records) never sees a YouTube-specific string:

- ``parse_duration_iso``: ``contentDetails.duration`` (ISO-8601 ``PT#H#M#S``)
  → total seconds, matching the int-or-``None`` convention of
  ``metadata.duration_int`` (Story 6.1).
- ``published_at_to_upload_date``: ``snippet.publishedAt`` → the zero-padded
  ``YYYYMMDD`` string that ``metadata._normalize`` (``%Y%m%d`` strptime)
  expects (decision **D17** — the UTC date component is truncated as written,
  never timezone-shifted: a shift would move late-night uploads to the
  "wrong" day relative to what YouTube displays in Brazil; truncation is the
  honest, format-consistent choice).

This module will also host the YouTube Data API v3 client (Story 9.2); the
API facts backfill (``mc stats-backfill --from-api``) uses it. Invariants it
serves:

- **NFR-3** — pure, deterministic functions keep re-ingest/backfill
  idempotent: the same API facts always produce the same stored values.
- **NFR-5** — the API path uses an API key from an env var, never cookies;
  nothing here logs the key or any credential.
- **NFR-7** — no new dependency: stdlib ``re``/``datetime`` only (httpx is
  already pinned for the 9.2 client).
"""

from __future__ import annotations

import re
from datetime import datetime

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
