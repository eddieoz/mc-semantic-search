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

import pytest

from mc.ingest.discover import load_channels
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
        self.deletes: list[dict] = []

    def set_payload(self, *, collection_name, payload, points):
        self.calls.append({"collection": collection_name, "payload": payload, "points": points})

    def delete_payload(self, *, collection_name, keys, points):
        self.deletes.append({"collection": collection_name, "keys": keys, "points": points})

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


# --- --from-db: the chunked titles are the authoritative episode source -----
#
# The legacy mc_transcripts.db stores each video's real title. Re-deriving the
# episode from it (new parser: declared markers only) repairs every junk form
# at once — id digits parsed from a placeholder title (301415), and old-parser
# bare-digit runs from real titles ("Nunca use 123456" → 123456) that the
# resumability skip would otherwise preserve forever.


def _write_legacy_db(path: Path, rows: list[tuple[str, str, str]]) -> None:
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("CREATE TABLE transcripts (title TEXT, url TEXT, description TEXT)")
    con.executemany("INSERT INTO transcripts VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()


def test_from_db_title_overrides_junk_episode(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(
        cache_dir,
        "abc747junk1",
        {"date": "2025-03-16", "episode": "301415", "channel": "EddieOz"},
    )
    state.record_video_stats("abc747junk1", duration=100, episode=301415, date="2025-03-16")
    db = tmp_path / "legacy.db"
    _write_legacy_db(
        db,
        [
            (
                "[ IRL ][ ep 747 ] Shopping na Estonia",
                "https://www.youtube.com/watch?v=abc747junk1&t=0",
                "texto",
            )
        ],
    )
    spy = _SpyClient()

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        from_db=db,
        client=spy,
        collection="c",
    )

    assert state.video_facts("abc747junk1")["episode"] == 747
    episode_calls = [c for c in spy.calls if "episode" in c["payload"]]
    assert episode_calls, "the corrected episode must reach the index"
    assert not _targets_empty_field(episode_calls[0]["points"], "episode")
    # the cache file is corrected too, so a plain re-run cannot re-junk the row
    cached = json.loads((cache_dir / "abc747junk1.json").read_text())
    assert cached["episode"] == "747"
    assert report["episodes_repaired"] == 1


def test_from_db_clears_episode_when_title_declares_none(tmp_path):
    """'Nunca use 123456' is not an episode: the row AND the payload must lose
    the junk entirely — COALESCE alone can never erase a value."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.record_video_stats("abc123456x1", duration=100, episode=123456, date="2026-01-01")
    _write_cache(
        cache_dir,
        "abc123456x1",
        {"duration": 100, "episode": "123456", "date": "2026-01-01"},
    )
    db = tmp_path / "legacy.db"
    _write_legacy_db(
        db,
        [
            (
                "Nunca use 123456  Proteja seu Access Code",
                "https://www.youtube.com/watch?v=abc123456x1&t=0",
                "t",
            )
        ],
    )
    spy = _SpyClient()

    run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        from_db=db,
        client=spy,
        collection="c",
    )

    assert state.video_facts("abc123456x1")["episode"] is None
    assert spy.deletes and spy.deletes[0]["keys"] == ["episode"]
    cached = json.loads((cache_dir / "abc123456x1.json").read_text())
    assert cached["episode"] is None


def test_from_db_resyncs_payload_episode_even_when_row_agrees(tmp_path):
    """The payload can lag the row: a row whose episode already matches the
    title-derived value is still re-synced at the index, so a stale payload
    (e.g. left over from before this repair existed) is overwritten with the
    truth rather than skipped as "done"."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.record_video_stats("abc747ok123", duration=100, episode=747, date="2025-03-16")
    _write_cache(
        cache_dir,
        "abc747ok123",
        {"duration": 100, "episode": "747", "date": "2025-03-16"},
    )
    db = tmp_path / "legacy.db"
    _write_legacy_db(
        db,
        [
            (
                "[ IRL ][ ep 747 ] Shopping na Estonia",
                "https://www.youtube.com/watch?v=abc747ok123&t=0",
                "t",
            )
        ],
    )
    spy = _SpyClient()

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        from_db=db,
        client=spy,
        collection="c",
    )

    assert report["videos_skipped"] == 0  # title-bearing rows are never skipped
    assert report["episodes_repaired"] == 0  # the row itself needed no repair
    episode_calls = [c for c in spy.calls if "episode" in c["payload"]]
    assert episode_calls and episode_calls[0]["payload"]["episode"] == "747"
    assert not _targets_empty_field(episode_calls[0]["points"], "episode")
    assert spy.deletes == []


def test_from_db_video_missing_from_titles_keeps_normal_flow(tmp_path):
    """Only videos present in the legacy db get the title override; the rest
    resolve exactly as before."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.mark_processed("abc999ok123", "update")
    db = tmp_path / "legacy.db"
    _write_legacy_db(
        db,
        [
            (
                "[ IRL ][ ep 747 ] Shopping na Estonia",
                "https://www.youtube.com/watch?v=abc747junk1&t=0",
                "t",
            )
        ],
    )

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        from_db=db,
        fetch_missing=True,
        fetch=lambda _v: {"upload_date": "20261020", "title": "EP #999", "duration": 5400},
        sleeper=lambda _: None,
    )

    stats = state.archive_stats()
    assert stats["last_episode"] == 999
    assert report["episodes_repaired"] == 0


# ---------------------------------------------------------------------------
# Story 8.1 — `--from-tabs`: one yt-dlp full-extraction pass per channel tab
# prints id/upload_date/title/duration as TSV; each parsed line merges into
# the meta cache (never regressing known fields), dates the stats row and
# repairs the Qdrant payload where it is still empty. Exact dates only:
# `upload_date` is not available in flat listings and the approximate variant
# degrades to year-granularity on old entries (decision D13).
# ---------------------------------------------------------------------------

from subprocess import CompletedProcess  # noqa: E402

from mc.ingest.stats_backfill import run_tab_backfill  # noqa: E402

_TAB_TEMPLATE = "%(id)s\t%(upload_date)s\t%(title)s\t%(duration)s"


class _Recorder:
    """Fake yt-dlp runner: maps tab URL → (returncode, stdout, stderr)."""

    def __init__(self, responses: dict[str, tuple[int, str, str]]):
        self.responses = responses
        self.argvs: list[list[str]] = []

    def __call__(self, argv):
        self.argvs.append(argv)
        url = argv[-1]
        code, out, err = self.responses.get(url, (0, "", ""))
        return CompletedProcess(argv, code, stdout=out, stderr=err)


def _tsv(video_id: str, date: str, title: str, duration: str) -> str:
    return f"{video_id}\t{date}\t{title}\t{duration}"


def test_tab_pass_runs_full_extraction_per_tab(tmp_path):
    """One uncapped yt-dlp pass per tab — full extraction (no --flat-playlist,
    that is what makes upload_date available) with the reference script's
    proven randomized pacing."""
    rec = _Recorder({})
    state, cache_dir, _ = _ctx(tmp_path)

    run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=rec,
        sleeper=lambda _: None,
    )

    urls = [argv[-1] for argv in rec.argvs]
    assert urls == [
        "https://youtube.com/@chan/videos",
        "https://youtube.com/@chan/streams",
        "https://youtube.com/@chan/shorts",
    ]
    for argv in rec.argvs:
        assert "--flat-playlist" not in argv
        assert "--playlist-end" not in argv  # the daily-discovery cap must not apply
        print_idx = argv.index("--print")
        assert argv[print_idx + 1] == _TAB_TEMPLATE
        assert "--sleep-requests" in argv and "--sleep-interval" in argv
        assert argv[argv.index("--sleep-interval") + 1] == "3"
        assert argv[argv.index("--max-sleep-interval") + 1] == "8"


def test_tab_pass_passes_cookies_only_when_the_file_exists(tmp_path, monkeypatch):
    rec = _Recorder({})
    state, cache_dir, _ = _ctx(tmp_path)

    monkeypatch.setattr("mc.config.COOKIES_PATH", str(tmp_path / "cookies.txt"))
    run_tab_backfill(
        state=state, cache_dir=cache_dir, channels=("https://youtube.com/@c",),
        runner=rec, sleeper=lambda _: None,
    )
    assert all("--cookies" not in argv for argv in rec.argvs)

    (tmp_path / "cookies.txt").write_text("# netscape\n", encoding="utf-8")
    run_tab_backfill(
        state=state, cache_dir=cache_dir, channels=("https://youtube.com/@c",),
        runner=rec, sleeper=lambda _: None,
    )
    assert rec.argvs[-1][rec.argvs[-1].index("--cookies") + 1] == str(tmp_path / "cookies.txt")


def test_tab_pass_parses_tsv_and_dates_the_rows(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)
    rec = _Recorder({
        "https://youtube.com/@chan/videos": (
            0,
            _tsv("abc747video", "20240315", "[ IRL ][ ep 747 ] x", "5400.0") + "\n",
            "",
        ),
    })

    report = run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=rec,
        sleeper=lambda _: None,
    )

    facts = state.video_facts("abc747video")
    assert facts["date"] == "2024-03-15"
    assert facts["episode"] == 747
    assert facts["duration"] == 5400
    assert report["videos_seen"] == 1
    assert report["videos_dated"] == 1
    assert report["tabs_listed"] == 3


def test_tab_pass_writes_the_cache_without_regressing_known_fields(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)
    _write_cache(cache_dir, "abc747video", {"date": "2020-01-01", "episode": "700"})
    rec = _Recorder({
        "https://youtube.com/@chan/videos": (
            0,
            _tsv("abc747video", "20240315", "[ IRL ][ ep 747 ] x", "5400.0") + "\n",
            "",
        ),
    })

    run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=rec,
        sleeper=lambda _: None,
    )

    import json

    stored = json.loads((cache_dir / "abc747video.json").read_text(encoding="utf-8"))
    assert stored["date"] == "2020-01-01"  # known cache fields are never regressed
    assert stored["duration"] == 5400  # missing ones are gained
    assert state.video_facts("abc747video")["date"] == "2020-01-01"


def test_tab_pass_repairs_empty_index_payloads(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)
    rec = _Recorder({
        "https://youtube.com/@chan/videos": (
            0,
            _tsv("abc747video", "20240315", "[ IRL ][ ep 747 ] x", "5400.0") + "\n",
            "",
        ),
    })
    spy = _SpyClient()

    report = run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=rec,
        sleeper=lambda _: None,
        client=spy,
        collection="c",
    )

    written = {k: v for call in spy.calls for k, v in call["payload"].items()}
    assert written == {"date": "2024-03-15", "episode": "747", "duration": 5400}
    for call in spy.calls:
        (field,) = call["payload"].keys()
        assert _targets_empty_field(call["points"], field)
    assert report["index_updated"] == 1


def test_tab_pass_skips_invalid_video_ids(tmp_path):
    """A degraded no-cookie listing can surface non-video ids — never store them."""
    state, cache_dir, _ = _ctx(tmp_path)
    rec = _Recorder({
        "https://youtube.com/@chan/videos": (
            0,
            _tsv("UCx24charchannelid24", "20240315", "t", "1") + "\n",
            "",
        ),
    })

    report = run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=rec,
        sleeper=lambda _: None,
    )

    assert state.video_facts("UCx24charchannelid24") == {
        "duration": None,
        "episode": None,
        "date": None,
    }
    assert report["videos_seen"] == 0


def test_tab_pass_handles_na_fields(tmp_path):
    """yt-dlp prints NA for fields an entry lacks — NA is absence, not data."""
    state, cache_dir, _ = _ctx(tmp_path)
    rec = _Recorder({
        "https://youtube.com/@chan/shorts": (
            0,
            _tsv("abcshortvid", "NA", "Some Short", "NA") + "\n",
            "",
        ),
    })
    spy = _SpyClient()

    report = run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=rec,
        sleeper=lambda _: None,
        client=spy,
    )

    assert state.video_facts("abcshortvid")["date"] is None
    assert report["videos_dated"] == 0
    assert spy.calls == []  # nothing resolved → nothing offered to the index


def test_tab_pass_tolerates_an_absent_tab(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)
    rec = _Recorder({
        "https://youtube.com/@chan/streams": (
            1,
            "",
            "ERROR: This channel does not have a streams tab",
        ),
    })

    report = run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=rec,
        sleeper=lambda _: None,
    )

    assert report["tabs_skipped"] == 1
    assert report["tabs_failed"] == 0
    assert report["tabs_listed"] == 2  # videos + shorts still listed


def test_tab_pass_continues_after_a_failed_tab(tmp_path):
    """A bot gate or network error on one tab must not kill the run."""
    state, cache_dir, _ = _ctx(tmp_path)
    rec = _Recorder({
        "https://youtube.com/@chan/videos": (1, "", "Sign in to confirm you're not a bot"),
        "https://youtube.com/@chan/streams": (
            0,
            _tsv("abc747video", "20240315", "[ IRL ][ ep 747 ] x", "5400.0") + "\n",
            "",
        ),
    })

    report = run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=rec,
        sleeper=lambda _: None,
    )

    assert report["tabs_failed"] == 1
    assert state.video_facts("abc747video")["date"] == "2024-03-15"
    assert "1 tabs failed" in report["summary"]


def test_tab_pass_sleeps_between_tabs(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)
    sleeps: list[float] = []
    run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=_Recorder({}),
        sleeper=sleeps.append,
    )
    assert len(sleeps) >= 2  # between the 3 tabs of one channel


def test_tab_pass_overwrites_a_poisoned_cache_entry(tmp_path):
    """A placeholder-title poison (episode digits inside the video id, no
    duration) is not knowledge — the tab pass heals it instead of preserving
    it, exactly like the classic fetch path does."""
    state, cache_dir, _ = _ctx(tmp_path)
    _write_cache(cache_dir, "ab301415cde", {"episode": "301415"})  # no duration → poison
    rec = _Recorder({
        "https://youtube.com/@chan/videos": (
            0,
            _tsv("ab301415cde", "20240315", "[ IRL ][ ep 747 ] x", "5400.0") + "\n",
            "",
        ),
    })

    run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=rec,
        sleeper=lambda _: None,
    )

    import json

    stored = json.loads((cache_dir / "ab301415cde.json").read_text(encoding="utf-8"))
    assert stored["episode"] == "747"
    assert stored["duration"] == 5400
    assert state.video_facts("ab301415cde")["episode"] == 747


def test_tab_pass_salvages_partial_output_from_a_failed_tab(tmp_path):
    """A tab that dies mid-pass (bot gate) may still have extracted dozens of
    videos before failing — returncode != 0 must not discard their TSV lines,
    or a flagged session converts a half-finished tab into zero progress."""
    state, cache_dir, _ = _ctx(tmp_path)
    rec = _Recorder({
        "https://youtube.com/@chan/videos": (
            1,
            _tsv("abc747video", "20240315", "[ IRL ][ ep 747 ] x", "5400.0")
            + "\n"
            + _tsv("abc747nex9x", "20240320", "[ IRL ][ ep 748 ] y", "5300.0")
            + "\n",
            "ERROR: [youtube] 9_6aaZWmhpI: Sign in to confirm you're not a bot",
        ),
        # the other tabs are absent — a _Recorder default of (0, "", "") would
        # fake a completed pass and inflate tabs_listed
        "https://youtube.com/@chan/streams": (
            1,
            "",
            "ERROR: [youtube] @chan: This channel does not have a streams tab",
        ),
        "https://youtube.com/@chan/shorts": (
            1,
            "",
            "ERROR: [youtube] @chan: This channel does not have a shorts tab",
        ),
    })

    report = run_tab_backfill(
        state=state,
        cache_dir=cache_dir,
        channels=("https://youtube.com/@chan",),
        runner=rec,
        sleeper=lambda _: None,
    )

    assert report["tabs_failed"] == 1
    assert report["tabs_skipped"] == 2  # streams/shorts absent, as declared above
    assert report["tabs_listed"] == 0  # the tab did not run to completion
    assert report["videos_seen"] == 2  # salvaged from the failed tab
    assert state.video_facts("abc747video")["date"] == "2024-03-15"
    assert state.video_facts("abc747nex9x")["date"] == "2024-03-20"


# ---------------------------------------------------------------------------
# Story 9.3 — `run_api_backfill`: the YouTube Data API fact pass (D14/D15).
# Discovery (uploads playlists) ∪ indexed universe → batched facts → the SAME
# write sequence as the tab pass: cache merge non-regressing, COALESCE row
# upsert, fills-empty payload repair. Complete rows are skipped BEFORE any API
# call, so a re-run after a full backlog run costs 0 quota units (NFR-3).
# ---------------------------------------------------------------------------

from mc.ingest.stats_backfill import run_api_backfill  # noqa: E402
from mc.ingest.youtube_api import QuotaExceededError, YouTubeApiError  # noqa: E402


class _FakeApiClient:
    """Story 9.2 client stand-in: canned facts, no network, call counters."""

    def __init__(
        self,
        facts: dict[str, dict],
        *,
        playlists: dict[str, str] | None = None,
        discovered: list[str] | None = None,
        missing: set[str] | None = None,
    ):
        self.facts = facts
        self.playlists = playlists or {c: f"UU{i}" for i, c in enumerate(["chan"])}
        self.discovered = list(discovered or [])
        self.missing = set(missing or set())
        self.facts_calls = 0
        self.batches: list[list[str]] = []
        self.quota_units = 0  # same shape as the real _ApiClient adapter

    def channel_uploads(self, channels):
        self.channel_calls = channels
        return dict(self.playlists)

    def playlist_videos(self, playlist_id):
        return list(self.discovered)

    def video_facts(self, ids):
        self.facts_calls += 1
        self.batches.append(list(ids))
        served = {i: dict(self.facts[i]) for i in ids if i in self.facts}
        missing = [i for i in ids if i in self.missing or i not in self.facts]
        return served, missing


def test_api_universe_is_discovery_union_index(tmp_path):
    """Coverage contract #3: a video known only to the index still gets facts,
    a video known only to discovery still gets a row (D15)."""
    state, cache_dir, _ = _ctx(tmp_path)
    state.mark_processed("idxonly", "update")
    _write_cache(cache_dir, "cacheonly", {})
    api = _FakeApiClient(
        {
            "idxonly": {"upload_date": "20260101", "title": "t", "duration": 60},
            "cacheonly": {"upload_date": "20260102", "title": "t", "duration": 60},
            "disconly001": {"upload_date": "20260103", "title": "t", "duration": 60},
        },
        discovered=["disconly001"],
    )

    report = run_api_backfill(
        state=state, cache_dir=cache_dir, api_client=api, repair_index=False
    )

    assert report["videos_seen"] == 3
    assert report["discovered"] == 1
    assert report["channels_resolved"] == 1
    assert state.video_facts("idxonly")["date"] == "2026-01-01"
    assert state.video_facts("cacheonly")["date"] == "2026-01-02"
    assert state.video_facts("disconly001")["date"] == "2026-01-03"


def test_api_cache_merge_never_regresses_known_fields(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"duration": 999, "episode": "5", "date": "2020-01-01"})
    api = _FakeApiClient({"v1": {"upload_date": "20260101", "title": "t", "duration": 100}})

    run_api_backfill(state=state, cache_dir=cache_dir, api_client=api, repair_index=False)

    facts = state.video_facts("v1")
    assert facts["duration"] == 999
    assert facts["episode"] == 5
    assert facts["date"] == "2020-01-01"


def test_api_poisoned_cache_entry_is_healed_by_api_truth(tmp_path):
    """A placeholder-title poison (digits from the video id, no duration) is
    not knowledge — the API facts REPLACE it, like the tab pass does."""
    state, cache_dir, _ = _ctx(tmp_path)
    _write_cache(cache_dir, "xg301415xZU", {"episode": "301415"})
    api = _FakeApiClient(
        {"xg301415xZU": {"upload_date": "20260101", "title": "[ IRL ][ ep 747 ] x", "duration": 60}}
    )

    report = run_api_backfill(state=state, cache_dir=cache_dir, api_client=api, repair_index=False)

    assert state.video_facts("xg301415xZU")["episode"] == 747
    assert report["cache_written"] == 1
    stored = json.loads((cache_dir / "xg301415xZU.json").read_text())
    assert stored["episode"] == "747"


def test_api_episode_derived_from_api_title(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)
    api = _FakeApiClient(
        {
            "vid00000441": {
                "upload_date": "20260101",
                "title": "[ Morning Crypto ][ ep 441 ] x",
                "duration": 60,
            }
        },
        discovered=["vid00000441"],
    )

    run_api_backfill(state=state, cache_dir=cache_dir, api_client=api, repair_index=False)

    assert state.video_facts("vid00000441")["episode"] == 441


def test_api_payload_repair_fills_only_missing_fields(tmp_path):
    """The payload repair is the tab pass's: only fields the index lacks are
    offered, each scoped by the is-empty guard."""
    state, cache_dir, _ = _ctx(tmp_path)
    state.mark_processed("v1", "update")
    state.record_video_stats("v1", duration=999)  # the row already knows duration
    spy = _SpyClient()
    api = _FakeApiClient(
        {"v1": {"upload_date": "20260101", "title": "[ IRL ][ ep 747 ] x", "duration": 100}}
    )

    report = run_api_backfill(
        state=state, cache_dir=cache_dir, client=spy, api_client=api
    )

    durations = [p["duration"] for p in spy.payloads() if "duration" in p]
    assert durations == []  # the index's duration is knowledge, never re-offered
    dates = [p["date"] for p in spy.payloads() if "date" in p]
    assert dates == ["2026-01-01"]
    episodes = [p["episode"] for p in spy.payloads() if "episode" in p]
    assert episodes == ["747"]
    assert report["index_updated"] == 1


def test_api_complete_rows_skip_every_api_call(tmp_path):
    """A re-run over complete rows is seconds of local reads and 0 quota
    units — the fake client raises if it is touched at all."""
    state, cache_dir, _ = _ctx(tmp_path)
    state.mark_processed("v1", "update")
    state.record_video_stats("v1", duration=60, episode=747, date="2026-01-01")

    class _Untouched(_FakeApiClient):
        def channel_uploads(self, channels):
            raise AssertionError("discovery must not run for a complete universe")

        def playlist_videos(self, playlist_id):
            raise AssertionError("discovery must not run for a complete universe")

        def video_facts(self, ids):
            raise AssertionError("facts must not be fetched for a complete universe")

    report = run_api_backfill(
        state=state, cache_dir=cache_dir, api_client=_Untouched({}), repair_index=False
    )

    assert report["quota_units"] == 0
    assert report["videos_seen"] == 0


def test_api_missing_ids_are_counted_never_fatal(tmp_path):
    """Deleted/private ids are counted as still missing (contract #4); every
    other undated video is still dated."""
    state, cache_dir, _ = _ctx(tmp_path)
    state.mark_processed("goner", "update")
    state.mark_processed("v1", "update")
    api = _FakeApiClient(
        {"v1": {"upload_date": "20260101", "title": "t", "duration": 60}},
        missing={"goner"},
    )

    report = run_api_backfill(state=state, cache_dir=cache_dir, api_client=api, repair_index=False)

    assert report["still_missing"] == 1
    assert report["videos_dated"] == 1
    assert state.video_facts("v1")["date"] == "2026-01-01"


def test_api_quota_units_landed_in_the_report(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)
    api = _FakeApiClient({"v1": {"upload_date": "20260101", "title": "t", "duration": 60}})
    api.quota_units = 7

    report = run_api_backfill(state=state, cache_dir=cache_dir, api_client=api, repair_index=False)

    assert report["quota_units"] == 7


def test_api_quota_exceeded_propagates(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)

    class _QuotaDead(_FakeApiClient):
        def channel_uploads(self, channels):
            raise QuotaExceededError("quotaExceeded", status=403)

    with pytest.raises(QuotaExceededError):
        run_api_backfill(state=state, cache_dir=cache_dir, api_client=_QuotaDead({}))


def test_api_other_api_error_propagates(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)

    class _Dead(_FakeApiClient):
        def channel_uploads(self, channels):
            raise YouTubeApiError("backendError", status=500)

    with pytest.raises(YouTubeApiError):
        run_api_backfill(state=state, cache_dir=cache_dir, api_client=_Dead({}))


def test_api_missing_key_fails_loud_before_any_work(tmp_path, monkeypatch):
    state, cache_dir, _ = _ctx(tmp_path)
    state.mark_processed("v1", "update")
    monkeypatch.setattr("mc.config.YOUTUBE_API_KEY", None)

    with pytest.raises(YouTubeApiError, match="YOUTUBE_API_KEY"):
        run_api_backfill(state=state, cache_dir=cache_dir, api_client=None)


def test_api_paces_between_fact_batches(tmp_path):
    state, cache_dir, _ = _ctx(tmp_path)
    state.mark_processed("v1", "update")
    api = _FakeApiClient({"v1": {"upload_date": "20260101", "title": "t", "duration": 60}})
    slept: list[float] = []

    run_api_backfill(
        state=state, cache_dir=cache_dir, api_client=api, repair_index=False,
        sleeper=slept.append,
    )

    assert len(slept) == 1  # one batch of facts → one politeness pause
    assert slept[0] > 0


def test_api_channels_default_to_load_channels(tmp_path, monkeypatch):
    state, cache_dir, _ = _ctx(tmp_path)
    seen: list[tuple] = []

    class _Recording(_FakeApiClient):
        def channel_uploads(self, channels):
            seen.append(tuple(channels))
            return {}

    run_api_backfill(
        state=state, cache_dir=cache_dir, api_client=_Recording({}), repair_index=False
    )
    # config.yaml's real channel list reaches the client untouched
    assert seen and seen[0] == load_channels()
