"""Story 6.4 — `mc stats-backfill`: bootstrap archive stats for the videos
already indexed, without any yt-dlp fetch by default.

Duration precedence: metadata cache > local transcripts cache
(`max(start + duration)` over the JSON segment list) > (with --fetch-missing)
one yt-dlp fetch that upgrades the cache file. Resumable: a video whose stats
row already has duration, episode AND date is skipped, and a fetched cache
entry is never re-fetched.
"""

from __future__ import annotations

import json
from pathlib import Path

from mc.ingest.stats_backfill import run_stats_backfill
from mc.state import StateStore


def _write_cache(cache_dir: Path, video_id: str, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{video_id}.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_transcript(transcripts_dir: Path, video_id: str, segments: list[dict]) -> None:
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    (transcripts_dir / f"{video_id}.srt").write_text(json.dumps(segments), encoding="utf-8")


def _ctx(tmp_path: Path):
    state = StateStore(tmp_path / "ingest.db")
    return state, tmp_path / "meta", tmp_path / "transcripts"


def test_universe_is_processed_union_cache_ids(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.mark_processed("p1", "update")
    state.mark_processed("p2", "update")
    _write_cache(cache_dir, "p2", {"episode": "1", "date": "2026-01-01"})
    _write_cache(cache_dir, "c1", {"episode": "2", "date": "2026-01-02"})

    report = run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)

    assert report["videos_considered"] == 3  # p1 ∪ p2 ∪ c1 (deduped)


def test_duration_derived_from_transcript_segments(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})
    _write_transcript(
        transcripts_dir,
        "v1",
        [
            {"text": "a", "start": 0.0, "duration": 10.0},
            {"text": "b", "start": 10.0, "duration": 5.0},
        ],
    )

    report = run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)
    stats = state.archive_stats()
    assert stats["seconds"] == 15  # int(max(0+10, 10+5))
    assert stats["last_episode"] == 971
    assert report["videos_backfilled"] == 1


def test_cache_duration_precedes_transcript(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "1", "date": "2026-01-01", "duration": 999})
    _write_transcript(transcripts_dir, "v1", [{"start": 0.0, "duration": 10.0}])

    run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)
    assert state.archive_stats()["seconds"] == 999


def test_transcript_precedes_fetch(tmp_path):
    """A local transcript beats a fetch even when --fetch-missing is set."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "1", "date": "2026-01-01"})
    _write_transcript(transcripts_dir, "v1", [{"start": 0.0, "duration": 42.0}])

    def fake_fetch(_vid):
        raise AssertionError("must not fetch when a transcript exists")

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=fake_fetch,
        sleeper=lambda _: None,
    )
    assert state.archive_stats()["seconds"] == 42
    assert report["fetched"] == 0


def test_limit_bounds_a_run(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    for i in range(3):
        _write_cache(cache_dir, f"c{i}", {"episode": str(i), "date": "2026-01-01"})

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        limit=2,
    )
    assert report["videos_considered"] == 2
    assert state.archive_stats()["videos"] == 2


def test_second_run_is_resumable(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})
    _write_transcript(transcripts_dir, "v1", [{"start": 0.0, "duration": 15.0}])

    first = run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)
    assert first["videos_backfilled"] == 1
    assert state.archive_stats()["seconds"] == 15

    second = run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)
    assert second["videos_backfilled"] == 0
    assert second["videos_skipped"] == 1
    assert state.archive_stats()["seconds"] == 15  # never doubled


def test_fetch_missing_upgrades_cache_with_duration(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})  # no duration

    calls: list[str] = []

    def fake_fetch(video_id: str) -> dict:
        calls.append(video_id)
        return {"upload_date": "20261020", "title": "EP #971", "duration": 5400}

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=fake_fetch,
        sleeper=lambda _: None,
    )
    assert report["fetched"] == 1
    assert state.archive_stats()["seconds"] == 5400
    # the cache file is upgraded with duration, so the fetch happens once ever
    cached = json.loads((cache_dir / "v1.json").read_text())
    assert cached["duration"] == 5400

    calls.clear()
    run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=fake_fetch,
        sleeper=lambda _: None,
    )
    assert calls == []  # cache now has duration → no second fetch


def test_missing_transcript_without_flag_is_null_no_exception(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})  # no duration

    report = run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)
    stats = state.archive_stats()
    assert stats["videos"] == 1
    assert stats["seconds"] is None  # no duration source, no flag → NULL
    assert stats["last_episode"] == 971  # episode/date still recorded
    assert report["still_missing"] == 1


def test_garbage_transcript_is_skipped_not_fatal(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "1", "date": "2026-01-01"})
    (transcripts_dir).mkdir(parents=True, exist_ok=True)
    (transcripts_dir / "v1.srt").write_text("not json at all", encoding="utf-8")

    report = run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)
    assert state.archive_stats()["seconds"] is None
    assert report["still_missing"] == 1


def test_fetch_missing_skips_already_complete_videos(tmp_path):
    """--fetch-missing targets videos *still lacking* a duration: a video whose
    stats row already carries one is skipped in both modes, so a re-run
    converges and never touches YouTube again."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.record_video_stats("v1", duration=5400, episode=971, date="2026-10-20")
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})  # no duration

    def boom(_vid):
        raise AssertionError("must not fetch a video that already has a duration")

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=boom,
        sleeper=lambda _: None,
    )
    assert report["fetched"] == 0
    assert report["videos_skipped"] == 1
    assert report["videos_backfilled"] == 0
    assert state.archive_stats()["seconds"] == 5400


def test_garbage_cache_duration_falls_through_to_transcript(tmp_path):
    """A junk duration in a (hand-edited) cache entry must not crash — it
    falls through to the transcript derivation instead."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(
        cache_dir,
        "v1",
        {"episode": "1", "date": "2026-01-01", "duration": "not-a-number"},
    )
    _write_transcript(transcripts_dir, "v1", [{"start": 0.0, "duration": 15.0}])

    report = run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)
    assert state.archive_stats()["seconds"] == 15
    assert report["videos_backfilled"] == 1


# --- Index repair: fill missing episode/date in the Qdrant payload -----------
#
# The search result card renders episode/date from the *payload* (engine.py),
# so a video whose payload predates the enrichment shows "ep — · sem data"
# forever. The backfill repairs those payloads in place — payload only, no
# re-embedding — filling a field ONLY where it is empty (never overwriting a
# value the index already knows).


class _SpyClient:
    """Minimal QdrantClient stand-in: records every set_payload call."""

    def __init__(self):
        self.calls: list[dict] = []

    def set_payload(self, *, collection_name, payload, points):
        self.calls.append({"collection": collection_name, "payload": payload, "points": points})

    def payloads(self) -> list[dict]:
        return [call["payload"] for call in self.calls]


def _targets_empty_field(filt, field: str) -> bool:
    """The repair filter must scope the write to points where `field` is empty."""
    from qdrant_client import models

    return any(
        isinstance(cond, models.IsEmptyCondition) and cond.is_empty.key == field
        for cond in filt.must
    )


def test_backfill_repairs_missing_index_payload_fields(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20", "duration": 5400})
    spy = _SpyClient()

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        client=spy,
        collection="mc-transcripts-v1",
    )

    # episode rides as the string the ingest pipeline writes; duration as int
    written = {k: v for call in spy.calls for k, v in call["payload"].items()}
    assert written == {"episode": "971", "date": "2026-10-20", "duration": 5400}
    for call in spy.calls:
        assert call["collection"] == "mc-transcripts-v1"
        (field,) = call["payload"].keys()
        assert _targets_empty_field(call["points"], field)
    assert report["index_updated"] == 1


def test_backfill_index_repair_skips_unknown_fields(tmp_path):
    """Only fields we actually resolved are written — never a null/guess."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"duration": 120})  # no episode, no date
    spy = _SpyClient()

    run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        client=spy,
        collection="c",
    )
    assert spy.payloads() == [{"duration": 120}]


def test_backfill_index_repair_can_be_disabled(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "1", "date": "2026-01-01", "duration": 10})
    spy = _SpyClient()

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        client=spy,
        collection="c",
        repair_index=False,
    )
    assert spy.calls == []
    assert report["index_updated"] == 0
    # the stats table is still written with repair off
    assert state.archive_stats()["videos"] == 1


def test_backfill_index_repair_failure_is_not_fatal(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "1", "date": "2026-01-01", "duration": 10})

    class _Boom:
        def set_payload(self, **kwargs):
            raise RuntimeError("qdrant unreachable")

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        client=_Boom(),
        collection="c",
    )
    assert report["index_updated"] == 0
    assert state.archive_stats()["videos"] == 1  # stats survived the repair failure


def test_backfill_fills_episode_and_date_when_duration_already_known(tmp_path):
    """The old duration-only skip left episode/date NULL forever for any video
    that already had a duration. Those fields must still be filled."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.record_video_stats("v1", duration=5400)  # duration known; ep/date missing
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})

    run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)

    stats = state.archive_stats()
    assert stats["last_episode"] == 971
    assert stats["last_episode_date"] == "2026-10-20"


def test_fully_populated_row_is_skipped_and_index_untouched(tmp_path):
    """Resumability: a row with duration+episode+date is complete — skip it and
    do not touch the index again."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.record_video_stats("v1", duration=1, episode=1, date="2026-01-01")
    _write_cache(cache_dir, "v1", {"episode": "1", "date": "2026-01-01", "duration": 1})
    spy = _SpyClient()

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        client=spy,
        collection="c",
    )
    assert report["videos_skipped"] == 1
    assert spy.calls == []


def test_index_repair_fills_only_the_missing_field(tmp_path):
    """A row missing only `date` writes only `date` — the fields the archive
    already knows are never overwritten."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.record_video_stats("v1", duration=100, episode=971)  # date missing
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20", "duration": 100})
    spy = _SpyClient()

    run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        client=spy,
        collection="c",
    )
    assert spy.payloads() == [{"date": "2026-10-20"}]


def test_fetch_missing_backfills_episode_and_date_in_the_same_run(tmp_path):
    """A fetch brings episode/date as well as duration. Because the fetch
    rewrites the cache file, the freshly written values must be re-read — the
    stale in-memory dict would otherwise drop them until the next run."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.mark_processed("v1", "update")  # nothing cached yet

    def fake_fetch(_vid):
        return {"upload_date": "20261020", "title": "EP #971", "duration": 5400}

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=fake_fetch,
        sleeper=lambda _: None,
    )

    stats = state.archive_stats()
    assert stats["last_episode"] == 971
    assert stats["last_episode_date"] == "2026-10-20"
    assert stats["seconds"] == 5400
    assert report["fetched"] == 1


def test_fetch_missing_repairs_the_index_too(tmp_path):
    """An old point that never had episode/date gets both after a
    --fetch-missing run — that is the whole point for pre-enrichment videos."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.mark_processed("v1", "update")
    spy = _SpyClient()

    run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=lambda _vid: {
            "upload_date": "20261020",
            "title": "EP #971",
            "duration": 5400,
        },
        sleeper=lambda _: None,
        client=spy,
        collection="mc-transcripts-v1",
    )

    written = {k: v for call in spy.calls for k, v in call["payload"].items()}
    assert written == {"episode": "971", "date": "2026-10-20", "duration": 5400}


def test_fetch_missing_fires_when_only_episode_and_date_are_missing(tmp_path):
    """The old early-return resolved the duration from the local transcript and
    then never fetched — a pre-enrichment video with a transcript on disk kept
    'ep — · sem data' forever. The fetch must still fire when episode/date are
    the missing facts."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.mark_processed("v1", "update")
    _write_transcript(transcripts_dir, "v1", [{"start": 0.0, "duration": 600.0}])
    spy = _SpyClient()  # no metadata cache entry at all

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=lambda _vid: {
            "upload_date": "20261020",
            "title": "EP #971",
            "duration": 5400,
        },
        sleeper=lambda _: None,
        client=spy,
        collection="c",
    )

    stats = state.archive_stats()
    assert stats["last_episode"] == 971
    assert stats["last_episode_date"] == "2026-10-20"
    # the fetch upgrades the cache, and cache beats the transcript derivation
    assert stats["seconds"] == 5400
    assert report["fetched"] == 1
    written = {k: v for call in spy.calls for k, v in call["payload"].items()}
    assert written == {"episode": "971", "date": "2026-10-20", "duration": 5400}


def test_fetch_missing_fires_for_a_cache_entry_lacking_episode_and_date(tmp_path):
    """A cache entry that only carries a duration must not shield itself from
    the fetch either — the missing facts here are episode and date."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"duration": 120})

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=lambda _vid: {
            "upload_date": "20261020",
            "title": "EP #971",
            "duration": 5400,
        },
        sleeper=lambda _: None,
    )

    stats = state.archive_stats()
    assert stats["last_episode"] == 971
    assert stats["last_episode_date"] == "2026-10-20"
    assert report["fetched"] == 1
    # the cache entry is upgraded in place, so the next run never re-fetches
    cached = json.loads((cache_dir / "v1.json").read_text())
    assert cached["episode"] == "971"
    assert cached["date"] == "2026-10-20"


# --- Placeholder-title poison (degraded yt-dlp dumps) -----------------------
#
# A bot-gated yt-dlp (old version) exits 0 with a degraded dump whose title is
# literally "youtube video #<video_id>", so the episode parser read digits out
# of the VIDEO ID ("xg301415xZU" → 301415, "--5z_jFh-_g" → 5) and the landing
# page crowned "último episódio #301415". Marker: the cache entry carries no
# duration (degraded dumps never do; real fetches always write the key) AND
# the episode's digits occur inside the video id.


def test_transcript_duration_does_not_complete_a_poisoned_row(tmp_path):
    """With transcripts on disk the duration resolves offline — a poisoned row
    must NOT be completed by it (that would make the junk id-derived episode
    permanent), so the fetch still fires to heal the episode."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(
        cache_dir,
        "--5z_jFh-_g",
        {"date": "2021-09-21", "episode": "5", "channel": "EddieOz"},
    )
    _write_transcript(transcripts_dir, "--5z_jFh-_g", [{"start": 0.0, "duration": 9114.0}])

    def fake_fetch(_vid):
        return {"upload_date": "20210921", "title": "EP #88", "duration": 9114}

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=fake_fetch,
        sleeper=lambda _: None,
    )
    assert report["fetched"] == 1
    assert state.archive_stats()["last_episode"] == 88


def test_poisoned_row_completed_by_transcript_pass_is_healed_by_fetch_pass(tmp_path):
    """Pass 1 (no fetch) fills the duration from the transcript. Pass 2 with
    --fetch-missing must NOT treat that poisoned row as done."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(
        cache_dir,
        "--5z_jFh-_g",
        {"date": "2021-09-21", "episode": "5", "channel": "EddieOz"},
    )
    _write_transcript(transcripts_dir, "--5z_jFh-_g", [{"start": 0.0, "duration": 9114.0}])

    run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)
    assert state.video_facts("--5z_jFh-_g")["duration"] == 9114

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=lambda _v: {"upload_date": "20210921", "title": "EP #88", "duration": 9114},
        sleeper=lambda _: None,
    )
    assert report["videos_skipped"] == 0
    assert report["fetched"] == 1
    assert state.video_facts("--5z_jFh-_g")["episode"] == 88


def test_poisoned_payload_episode_is_overwritten_not_preserved(tmp_path):
    """The broken run already wrote the junk episode into the Qdrant payloads —
    the healed value must OVERWRITE it, so for a poisoned video the episode
    repair drops the is-empty guard (duration/date still fill empties only)."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(
        cache_dir,
        "--5z_jFh-_g",
        {"date": "2021-09-21", "episode": "5", "channel": "EddieOz"},
    )
    spy = _SpyClient()

    run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=lambda _v: {"upload_date": "20210921", "title": "EP #88", "duration": 9114},
        sleeper=lambda _: None,
        client=spy,
        collection="c",
    )
    episode_calls = [c for c in spy.calls if "episode" in c["payload"]]
    assert episode_calls, "the healed episode must reach the index"
    assert episode_calls[0]["payload"]["episode"] == "88"
    assert not _targets_empty_field(episode_calls[0]["points"], "episode")
    for call in spy.calls:
        if "episode" not in call["payload"]:
            assert _targets_empty_field(call["points"], next(iter(call["payload"])))


def test_real_fetch_cache_is_never_treated_as_poison(tmp_path):
    """The marker requires a durationless cache entry: a real fetch (which
    always writes the duration key) is trusted even when the episode digits
    occur inside the video id."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.record_video_stats("v5", duration=100, episode=5, date="2026-01-01")
    _write_cache(cache_dir, "v5", {"episode": "5", "date": "2026-01-01", "duration": 100})

    def boom(_vid):
        raise AssertionError("a real-fetch cache entry must not be refetched")

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=boom,
        sleeper=lambda _: None,
    )
    assert report["videos_skipped"] == 1
    assert report["fetched"] == 0


def test_poison_without_fetch_leaves_cache_and_row_untouched(tmp_path):
    """Without --fetch-missing the poison is recognized (not 'known') but
    nothing is erased: the row keeps its values and the cache file is not
    rewritten — the next --fetch-missing run does the healing."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.record_video_stats("--5z_jFh-_g", duration=9114, episode=5, date="2021-09-21")
    _write_cache(
        cache_dir,
        "--5z_jFh-_g",
        {"date": "2021-09-21", "episode": "5", "channel": "EddieOz"},
    )

    report = run_stats_backfill(state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir)
    assert report["fetched"] == 0
    assert state.video_facts("--5z_jFh-_g")["episode"] == 5
    cached = json.loads((cache_dir / "--5z_jFh-_g.json").read_text())
    assert cached["episode"] == "5"
