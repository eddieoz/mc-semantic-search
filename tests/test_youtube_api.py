"""Story 9.1 + 9.2 — ISO-8601 fact parsing and the API transport client.

Story 9.1: pure-function tests for the YouTube Data API v3 fact parsers
(``contentDetails.duration`` (ISO-8601 ``PT#H#M#S``) → total seconds, and
``snippet.publishedAt`` (ISO-8601 timestamp) → zero-padded ``YYYYMMDD``).

Story 9.2: transport-client tests against a fake transport (a callable keyed
by endpoint URL) — no test ever touches the network (D16).
"""

from __future__ import annotations

import pytest

from mc.ingest.youtube_api import (
    CHANNELS_URL,
    PLAYLIST_ITEMS_URL,
    VIDEOS_URL,
    QuotaCounter,
    QuotaExceededError,
    YouTubeApiError,
    _error_from_payload,
    fetch_channel_uploads,
    fetch_video_facts,
    iter_playlist_videos,
    parse_duration_iso,
    published_at_to_upload_date,
)

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


# ---------------------------------------------------------------------------
# Story 9.2 — fixtures: fake transport + canned API responses
# ---------------------------------------------------------------------------

API_KEY = "fake-api-key-for-tests"

#: A syntactically valid channel id: "UC" + 22 chars (YouTube's shape).
CHANNEL_ID = "UC0123456789abcdefghijkl"
#: The uploads playlist the API derives from it: "UU" + the same 22 chars.
UPLOADS_ID = "UU0123456789abcdefghijkl"

HANDLE_URL = "https://www.youtube.com/@MorningCrypto"
CHANNEL_ID_URL = f"https://www.youtube.com/channel/{CHANNEL_ID}"
C_URL = "https://www.youtube.com/c/morningcrypto"
USER_URL = "https://www.youtube.com/user/morningcrypto"
HANDLE_TAB_URL = "https://www.youtube.com/@MorningCrypto/videos"


class FakeTransport:
    """Injectable transport keyed by endpoint URL (no network, ever).

    Each route maps a URL to a list of outcomes (JSON dict or Exception)
    consumed in order; a single-outcome list repeats, so unbounded
    pagination loops against the last page still terminate in tests.
    Records every call as ``(url, params)`` for assertions.
    """

    def __init__(self, routes: dict[str, list]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, params: dict) -> dict:
        self.calls.append((url, dict(params)))
        if url not in self.routes:
            raise AssertionError(f"unexpected transport call to {url}: {params}")
        seq = self.routes[url]
        outcome = seq[0] if len(seq) == 1 else seq.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def calls_to(self, url: str) -> list[dict]:
        """Params dicts recorded for one endpoint, in call order."""
        return [params for called_url, params in self.calls if called_url == url]


def channels_response(uploads_id: str = UPLOADS_ID) -> dict:
    """channels.list body for a channel that resolves to an uploads playlist."""
    return {
        "items": [
            {"contentDetails": {"relatedPlaylists": {"uploads": uploads_id}}}
        ]
    }


def playlist_page(video_ids: list[str], next_token: str | None = None) -> dict:
    """playlistItems.list body for one page of video ids."""
    page: dict = {
        "items": [{"contentDetails": {"videoId": vid}} for vid in video_ids]
    }
    if next_token is not None:
        page["nextPageToken"] = next_token
    return page


def videos_response(
    facts: dict[str, tuple[str, str, str]],
) -> dict:
    """videos.list body: id → (publishedAt, title, duration)."""
    return {
        "items": [
            {
                "id": vid,
                "snippet": {"publishedAt": published_at, "title": title},
                "contentDetails": {"duration": duration},
            }
            for vid, (published_at, title, duration) in facts.items()
        ]
    }


# ---------------------------------------------------------------------------
# Story 9.2 — fetch_channel_uploads
# ---------------------------------------------------------------------------


def test_fetch_channel_uploads_handle_form() -> None:
    fake = FakeTransport({CHANNELS_URL: [channels_response()]})
    result = fetch_channel_uploads(API_KEY, fake, (HANDLE_URL,))
    assert result == {HANDLE_URL: UPLOADS_ID}
    params = fake.calls_to(CHANNELS_URL)[0]
    assert params["forHandle"] == "@MorningCrypto"
    assert "id" not in params
    assert params["part"] == "contentDetails"
    assert params["key"] == API_KEY


def test_fetch_channel_uploads_channel_id_form() -> None:
    fake = FakeTransport({CHANNELS_URL: [channels_response()]})
    result = fetch_channel_uploads(API_KEY, fake, (CHANNEL_ID_URL,))
    assert result == {CHANNEL_ID_URL: UPLOADS_ID}
    params = fake.calls_to(CHANNELS_URL)[0]
    assert params["id"] == CHANNEL_ID
    assert "forHandle" not in params


def test_fetch_channel_uploads_c_form() -> None:
    # Legacy /c/ vanity URLs resolve via forHandle (the vanity name is the handle).
    fake = FakeTransport({CHANNELS_URL: [channels_response()]})
    result = fetch_channel_uploads(API_KEY, fake, (C_URL,))
    assert result == {C_URL: UPLOADS_ID}
    assert fake.calls_to(CHANNELS_URL)[0]["forHandle"] == "@morningcrypto"


def test_fetch_channel_uploads_user_form() -> None:
    # Legacy /user/ URLs resolve the same way as /c/ (forHandle).
    fake = FakeTransport({CHANNELS_URL: [channels_response()]})
    result = fetch_channel_uploads(API_KEY, fake, (USER_URL,))
    assert result == {USER_URL: UPLOADS_ID}
    assert fake.calls_to(CHANNELS_URL)[0]["forHandle"] == "@morningcrypto"


def test_fetch_channel_uploads_strips_tab_suffix() -> None:
    # A configured URL may already point at a tab — the tab suffix is stripped
    # before the handle is extracted, exactly one channels.list call is made.
    fake = FakeTransport({CHANNELS_URL: [channels_response()]})
    result = fetch_channel_uploads(API_KEY, fake, (HANDLE_TAB_URL,))
    assert result == {HANDLE_TAB_URL: UPLOADS_ID}
    assert fake.calls_to(CHANNELS_URL)[0]["forHandle"] == "@MorningCrypto"
    assert len(fake.calls) == 1


def test_fetch_channel_uploads_empty_items_skipped_run_continues() -> None:
    # A channel that resolves to nothing is logged, absent from the mapping,
    # and the run continues (Story 8.1's absent-tab tolerance).
    fake = FakeTransport(
        {CHANNELS_URL: [{"items": []}, channels_response()]}
    )
    result = fetch_channel_uploads(API_KEY, fake, (HANDLE_URL, CHANNEL_ID_URL))
    assert result == {CHANNEL_ID_URL: UPLOADS_ID}


def test_fetch_channel_uploads_unrecognized_url_skipped_without_call() -> None:
    # A URL that is not a channel form is logged and skipped — no API call,
    # no quota spent for it, the other channels still resolve.
    fake = FakeTransport({CHANNELS_URL: [channels_response()]})
    result = fetch_channel_uploads(
        API_KEY, fake, ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", HANDLE_URL)
    )
    assert result == {HANDLE_URL: UPLOADS_ID}
    assert len(fake.calls) == 1


def test_fetch_channel_uploads_charges_one_unit_per_channel() -> None:
    fake = FakeTransport(
        {CHANNELS_URL: [channels_response()] * 3}
    )
    quota = QuotaCounter()
    fetch_channel_uploads(
        API_KEY, fake, (HANDLE_URL, CHANNEL_ID_URL, C_URL), quota=quota
    )
    assert quota.units == 3


def test_fetch_channel_uploads_api_error_propagates() -> None:
    # Other HTTP failures are fatal (D18 — fail loud), not swallowed.
    fake = FakeTransport(
        {CHANNELS_URL: [YouTubeApiError("boom", status=500)]}
    )
    with pytest.raises(YouTubeApiError):
        fetch_channel_uploads(API_KEY, fake, (HANDLE_URL,))


# ---------------------------------------------------------------------------
# Story 9.2 — iter_playlist_videos
# ---------------------------------------------------------------------------


def test_iter_playlist_pagination_yields_ids_in_page_order() -> None:
    fake = FakeTransport(
        {
            PLAYLIST_ITEMS_URL: [
                playlist_page(["a" * 11, "b" * 11], next_token="tok1"),
                playlist_page(["c" * 11]),
            ]
        }
    )
    ids = list(iter_playlist_videos(API_KEY, UPLOADS_ID, fake))
    assert ids == ["a" * 11, "b" * 11, "c" * 11]
    assert len(fake.calls) == 2
    first_params = fake.calls_to(PLAYLIST_ITEMS_URL)[0]
    assert first_params["playlistId"] == UPLOADS_ID
    assert first_params["maxResults"] == 50
    assert "pageToken" not in first_params
    assert fake.calls_to(PLAYLIST_ITEMS_URL)[1]["pageToken"] == "tok1"


def test_iter_playlist_invalid_id_logged_and_skipped() -> None:
    # A 12-char id is not a video id — never yielded (discover._VIDEO_ID_RE).
    fake = FakeTransport(
        {PLAYLIST_ITEMS_URL: [playlist_page(["a" * 11, "b" * 12, "c" * 11])]}
    )
    ids = list(iter_playlist_videos(API_KEY, UPLOADS_ID, fake))
    assert ids == ["a" * 11, "c" * 11]


def test_iter_playlist_404_means_no_public_uploads() -> None:
    # HTTP 404 on the playlist → no yields, not fatal; the caller reports it.
    fake = FakeTransport(
        {PLAYLIST_ITEMS_URL: [YouTubeApiError("not found", status=404)]}
    )
    assert list(iter_playlist_videos(API_KEY, UPLOADS_ID, fake)) == []
    # The first (and only) page was still charged before the 404 came back.
    assert len(fake.calls) == 1


def test_iter_playlist_other_http_error_propagates() -> None:
    fake = FakeTransport(
        {PLAYLIST_ITEMS_URL: [YouTubeApiError("server exploded", status=500)]}
    )
    with pytest.raises(YouTubeApiError, match="server exploded"):
        list(iter_playlist_videos(API_KEY, UPLOADS_ID, fake))


def test_iter_playlist_quota_exceeded_propagates() -> None:
    # QuotaExceededError is a YouTubeApiError subclass — the 404 branch must
    # not swallow it (D18: quota exhaustion fails loud everywhere).
    fake = FakeTransport({PLAYLIST_ITEMS_URL: [QuotaExceededError("quota", status=403)]})
    with pytest.raises(QuotaExceededError):
        list(iter_playlist_videos(API_KEY, UPLOADS_ID, fake))


def test_iter_playlist_charges_one_unit_per_page() -> None:
    fake = FakeTransport(
        {
            PLAYLIST_ITEMS_URL: [
                playlist_page(["a" * 11], next_token="tok1"),
                playlist_page(["b" * 11], next_token="tok2"),
                playlist_page(["c" * 11]),
            ]
        }
    )
    quota = QuotaCounter()
    list(iter_playlist_videos(API_KEY, UPLOADS_ID, fake, quota=quota))
    assert quota.units == 3


# ---------------------------------------------------------------------------
# Story 9.2 — fetch_video_facts
# ---------------------------------------------------------------------------


def test_fetch_video_facts_batches_50_ids_per_call() -> None:
    ids = [f"v{i:010d}" for i in range(73)]
    fake = FakeTransport({VIDEOS_URL: [videos_response({}), videos_response({})]})
    facts, missing = fetch_video_facts(API_KEY, ids, fake)
    assert facts == {}
    # 73 ids → exactly 2 calls; the second batch carries the remaining 23.
    assert len(fake.calls) == 2
    first_ids = fake.calls_to(VIDEOS_URL)[0]["id"].split(",")
    second_ids = fake.calls_to(VIDEOS_URL)[1]["id"].split(",")
    assert len(first_ids) == 50
    assert len(second_ids) == 23
    assert first_ids + second_ids == ids
    # All 73 were absent from the responses → still_missing, in request order.
    assert missing == ids
    assert fake.calls_to(VIDEOS_URL)[0]["part"] == "snippet,contentDetails"


def test_fetch_video_facts_missing_ids_landed_in_missing_list() -> None:
    # Deleted/private videos are absent from the response — they land in the
    # missing list (coverage contract #4), never retried in-run, never fatal.
    ids = ["a" * 11, "b" * 11, "c" * 11]
    fake = FakeTransport(
        {
            VIDEOS_URL: [
                videos_response(
                    {
                        "a" * 11: ("2021-03-05T18:30:00Z", "Ep 1", "PT30M"),
                        "c" * 11: ("2022-07-01T10:00:00Z", "Ep 3", "PT1H"),
                    }
                )
            ]
        }
    )
    facts, missing = fetch_video_facts(API_KEY, ids, fake)
    assert missing == ["b" * 11]
    assert set(facts) == {"a" * 11, "c" * 11}


def test_fetch_video_facts_parses_through_story_91_parsers() -> None:
    fake = FakeTransport(
        {
            VIDEOS_URL: [
                videos_response(
                    {"a" * 11: ("2021-03-05T18:30:00Z", "Bitcoin caiu", "PT1H2M3S")}
                )
            ]
        }
    )
    facts, missing = fetch_video_facts(API_KEY, ["a" * 11], fake)
    assert missing == []
    assert facts == {
        "a" * 11: {"upload_date": "20210305", "title": "Bitcoin caiu", "duration": 3723}
    }


def test_fetch_video_facts_preserves_input_batch_order() -> None:
    ids = ["z" * 11, "a" * 11, "m" * 11]
    fake = FakeTransport({VIDEOS_URL: [videos_response({})]})
    _, missing = fetch_video_facts(API_KEY, ids, fake)
    assert missing == ids


def test_fetch_video_facts_empty_input_skips_the_call() -> None:
    fake = FakeTransport({})
    assert fetch_video_facts(API_KEY, [], fake) == ({}, [])
    assert fake.calls == []


def test_fetch_video_facts_charges_one_unit_per_batch() -> None:
    ids = [f"v{i:010d}" for i in range(60)]
    fake = FakeTransport({VIDEOS_URL: [videos_response({}), videos_response({})]})
    quota = QuotaCounter()
    fetch_video_facts(API_KEY, ids, fake, quota=quota)
    assert quota.units == 2


# ---------------------------------------------------------------------------
# Story 9.2 — QuotaCounter
# ---------------------------------------------------------------------------


def test_quota_counter_accumulates_across_clients() -> None:
    # Story 9.3 passes one counter through the whole run; the totals here
    # (2 channels + 2 pages + 1 batch) must land in the same object.
    fake = FakeTransport(
        {
            CHANNELS_URL: [channels_response(), channels_response()],
            PLAYLIST_ITEMS_URL: [
                playlist_page(["a" * 11], next_token="tok"),
                playlist_page(["b" * 11]),
            ],
            VIDEOS_URL: [videos_response({})],
        }
    )
    quota = QuotaCounter()
    fetch_channel_uploads(API_KEY, fake, (HANDLE_URL, CHANNEL_ID_URL), quota=quota)
    list(iter_playlist_videos(API_KEY, UPLOADS_ID, fake, quota=quota))
    fetch_video_facts(API_KEY, ["a" * 11], fake, quota=quota)
    assert quota.units == 5


def test_quota_counter_charge_defaults_to_one_unit() -> None:
    counter = QuotaCounter()
    counter.charge()
    counter.charge(3)
    assert counter.units == 4


# ---------------------------------------------------------------------------
# Story 9.2 — error mapping (the helper the default transport uses)
# ---------------------------------------------------------------------------


def test_error_payload_quota_exceeded_maps_to_quota_exceeded_error() -> None:
    body = {"error": {"code": 403, "errors": [{"reason": "quotaExceeded"}]}}
    error = _error_from_payload(403, body)
    assert isinstance(error, QuotaExceededError)
    assert error.status == 403


def test_error_payload_quota_reason_on_wrong_status_is_not_quota_error() -> None:
    # Only HTTP 403 + quotaExceeded means quota; anything else is a plain error.
    body = {"error": {"errors": [{"reason": "quotaExceeded"}]}}
    error = _error_from_payload(400, body)
    assert not isinstance(error, QuotaExceededError)
    assert isinstance(error, YouTubeApiError)
    assert error.status == 400


def test_error_payload_other_reason_carries_reason_in_message() -> None:
    body = {"error": {"errors": [{"reason": "playlistNotFound"}]}}
    error = _error_from_payload(404, body)
    assert isinstance(error, YouTubeApiError)
    assert not isinstance(error, QuotaExceededError)
    assert "playlistNotFound" in str(error)
    assert error.status == 404


def test_error_payload_unparseable_body_still_raises_with_status() -> None:
    # A non-JSON/HTML error body must not crash the mapping — a generic
    # YouTubeApiError with the status survives.
    error = _error_from_payload(502, {})
    assert isinstance(error, YouTubeApiError)
    assert error.status == 502
    assert "502" in str(error)


def test_error_payload_falls_back_to_error_status_field() -> None:
    # Some envelopes only carry error.status ("PERMISSION_DENIED") and no
    # errors[] list — the reason string falls back to it.
    body = {"error": {"code": 403, "status": "PERMISSION_DENIED"}}
    error = _error_from_payload(403, body)
    assert isinstance(error, YouTubeApiError)
    assert "PERMISSION_DENIED" in str(error)
