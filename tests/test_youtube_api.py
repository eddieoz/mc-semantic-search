"""Story 9.1 — ISO-8601 fact parsing.

Pure-function tests for the YouTube Data API v3 fact parsers:
``contentDetails.duration`` (ISO-8601 ``PT#H#M#S``) → total seconds, and
``snippet.publishedAt`` (ISO-8601 timestamp) → zero-padded ``YYYYMMDD``.
"""

from __future__ import annotations

from mc.ingest.youtube_api import parse_duration_iso, published_at_to_upload_date

# ---------------------------------------------------------------------------
# parse_duration_iso — contentDetails.duration
# ---------------------------------------------------------------------------


def test_parse_duration_full_combination() -> None:
    # 1h + 2m + 3s = 3600 + 120 + 3. (The story text says 3753, but that is
    # PT1H2M33S; PT1H2M3S is 3723 — verified against ISO-8601 arithmetic.)
    assert parse_duration_iso("PT1H2M3S") == 3723


def test_parse_duration_minutes_only() -> None:
    assert parse_duration_iso("PT5M") == 300


def test_parse_duration_seconds_only() -> None:
    assert parse_duration_iso("PT30S") == 30


def test_parse_duration_hours_only() -> None:
    assert parse_duration_iso("PT2H") == 7200


def test_parse_duration_hours_and_seconds() -> None:
    assert parse_duration_iso("PT1H30S") == 3630


def test_parse_duration_zero_is_valid() -> None:
    # PT0S is a legitimate zero (rare, but a real API value) — not None.
    assert parse_duration_iso("PT0S") == 0


def test_parse_duration_deleted_or_live_placeholder() -> None:
    # YouTube returns P0D for deleted and upcoming/live streams — no duration.
    assert parse_duration_iso("P0D") is None


def test_parse_duration_none() -> None:
    assert parse_duration_iso(None) is None


def test_parse_duration_empty() -> None:
    assert parse_duration_iso("") is None


def test_parse_duration_garbage_strings() -> None:
    for garbage in ("abc", "PT", "1H"):
        assert parse_duration_iso(garbage) is None, garbage


# ---------------------------------------------------------------------------
# published_at_to_upload_date — snippet.publishedAt
# ---------------------------------------------------------------------------


def test_published_at_zulu_truncated_to_date() -> None:
    # D17: truncate the UTC date component as written — no timezone shift.
    assert published_at_to_upload_date("2021-03-05T18:30:00Z") == "20210305"


def test_published_at_explicit_utc_offset() -> None:
    # The API only returns Z; +00:00 is the same instant in a different spelling.
    assert published_at_to_upload_date("2021-03-05T18:30:00+00:00") == "20210305"


def test_published_at_zero_padded() -> None:
    # Single-digit month/day must stay zero-padded in the output.
    assert published_at_to_upload_date("2021-03-05T00:00:00Z") == "20210305"
    assert published_at_to_upload_date("1999-01-02T23:59:59Z") == "19990102"


def test_published_at_none() -> None:
    assert published_at_to_upload_date(None) is None


def test_published_at_empty() -> None:
    assert published_at_to_upload_date("") is None


def test_published_at_garbage() -> None:
    for garbage in ("garbage", "2021-13-05", "05/03/2021", "not-a-date"):
        assert published_at_to_upload_date(garbage) is None, garbage
