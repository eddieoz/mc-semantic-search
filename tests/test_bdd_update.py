"""BDD step definitions for tests/bdd/update.feature (Story 3.4).

Runs against the local qdrant container. Discovery, download, metadata
and the embedder are fakes/spies; the assertions prove per-video
transactions, processed markers, failure isolation and the run summary.
"""

from __future__ import annotations

import json
import logging
import uuid

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from qdrant_client import QdrantClient

from mc.ingest import update
from mc.state import StateStore

scenarios("bdd/update.feature")

pytestmark = pytest.mark.integration


def _client() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=30)
    client.get_collections()
    return client


def _segments(n_chunks: int, seed: str) -> list[dict]:
    """Enough segments for n_chunks (30 per chunk), each text unique."""
    return [
        {"text": f"{seed} segment {i}", "start": float(i), "duration": 2.0}
        for i in range(n_chunks * 30)
    ]


@pytest.fixture
def ctx(tmp_path):
    try:
        client = _client()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    collection = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    ctx = {
        "client": client,
        "collection": collection,
        "state": StateStore(tmp_path / "ingest.db"),
        "tmp_path": tmp_path,
        "downloaded": [],
        "transcripts_dir": tmp_path / "transcripts",
        "queue_dir": tmp_path / "queue",
        "ignore_file": tmp_path / "transcripts" / "ignored_videos.txt",
        "exit_code": None,
    }
    yield ctx
    client.delete_collection(collection)


def _make_updater(ctx, transcript_plan: dict[str, list[dict] | None]):
    """Fake downloader recording calls; None return simulates failure."""
    def downloader(video_id: str):
        ctx["downloaded"].append(video_id)
        return transcript_plan.get(video_id)

    return downloader


def _run(ctx, metadata_fetch=None):
    def fake_embed_passage(texts):
        return [[0.5] * 384 for _ in texts]

    import mc.qdrant as q

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(q, "embed_passage", fake_embed_passage)
    try:
        ctx["exit_code"] = update.run_update(
            channels=("https://www.youtube.com/@test",),
            state=ctx["state"],
            client=ctx["client"],
            collection=ctx["collection"],
            discover=lambda channels: ctx["discovered"],
            downloader=ctx["downloader"],
            metadata_fetch=metadata_fetch or (lambda ids: ctx.get("metadata", {})),
            lock_path=ctx["tmp_path"] / "update.lock",
            transcripts_dir=ctx["transcripts_dir"],
            queue_dir=ctx["queue_dir"],
        )
    finally:
        monkeypatch.undo()


@given(
    parsers.parse('channels return video IDs ["{video_id}"] and none processed')
)
def channels_one_new(ctx, video_id):
    ctx["discovered"] = [video_id]
    ctx["downloader"] = _make_updater(ctx, {video_id: _segments(3, video_id)})


@given(parsers.parse('metadata for "{video_id}" reports duration {duration:d}'))
def metadata_reports_duration(ctx, video_id, duration):
    ctx["metadata"] = {video_id: {"duration": duration}}


@given(parsers.parse(
    'metadata for "{video_id}" reports duration {duration:d}, '
    'episode "{episode}" and date "{date}"'
))
def metadata_reports_full(ctx, video_id, duration, episode, date):
    ctx["metadata"] = {
        video_id: {"duration": duration, "episode": episode, "date": date}
    }


@when("`mc update` runs successfully")
def run_success(ctx):
    _run(ctx)


@then(
    parsers.parse(
        '"{video_id}" is searchable in Qdrant with {expected:d} chunk points'
    )
)
def searchable_with_points(ctx, video_id, expected):
    points, _ = ctx["client"].scroll(
        collection_name=ctx["collection"], limit=1000
    )
    chunks = [p for p in points if p.id != 0]  # exclude meta sentinel
    assert len(chunks) == expected
    assert all(p.payload["video_id"] == video_id for p in chunks)


@then(
    parsers.parse(
        'the state store marks "{video_id}" processed after the upsert commits'
    )
)
def state_marks_processed(ctx, video_id):
    assert ctx["state"].is_processed(video_id)


@then(
    parsers.parse(
        "the queue staging file contains the {expected:d} chunk records"
    )
)
def queue_staging_records(ctx, expected):
    pending = ctx["queue_dir"] / "pending.jsonl"
    assert pending.exists()
    lines = [
        json.loads(line)
        for line in pending.read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == expected
    assert set(lines[0]) >= {
        "video_id",
        "text",
        "url",
        "start_seconds",
        "date",
        "episode",
        "channel",
    }


@then(
    parsers.parse('every indexed point for "{video_id}" has payload "duration" = {expected:d}')
)
def indexed_points_carry_duration(ctx, video_id, expected):
    from qdrant_client.http import models as qm

    points, _ = ctx["client"].scroll(
        collection_name=ctx["collection"],
        scroll_filter=qm.Filter(
            must=[
                qm.FieldCondition(
                    key="video_id", match=qm.MatchValue(value=video_id)
                )
            ]
        ),
        limit=1000,
        with_payload=True,
    )
    chunks = [p for p in points if p.id != 0]  # exclude meta sentinel
    assert chunks, f"no indexed points for {video_id}"
    assert all(p.payload.get("duration") == expected for p in chunks)


@then(
    parsers.parse('the staged increment records carry "duration" = {expected:d}')
)
def staged_records_carry_duration(ctx, expected):
    pending = ctx["queue_dir"] / "pending.jsonl"
    lines = [
        json.loads(line)
        for line in pending.read_text().splitlines()
        if line.strip()
    ]
    assert lines, "no staged increment records"
    assert all(rec.get("duration") == expected for rec in lines)


@then(parsers.parse(
    'the state store records "{video_id}" stats with seconds {seconds:d} and episode {episode:d}'
))
def state_records_update_stats(ctx, video_id, seconds, episode):
    stats = ctx["state"].archive_stats()
    assert stats["videos"] == 1
    assert stats["seconds"] == seconds
    assert stats["last_episode"] == episode


@given(
    parsers.parse(
        'channels return ["{first}", "{second}"] and both are marked processed'
    )
)
def channels_all_processed(ctx, first, second):
    ctx["discovered"] = [first, second]
    for vid in (first, second):
        ctx["state"].mark_processed(vid, "update")
    ctx["downloader"] = _make_updater(ctx, {})


@when("`mc update` runs")
def run_plain(ctx, caplog):
    with caplog.at_level(logging.INFO, logger="mc.ingest.update"):
        _run(ctx)


@then(
    parsers.parse(
        'no transcript download happens for "{first}" or "{second}"'
    )
)
def no_downloads(ctx, first, second):
    assert ctx["downloaded"] == []


@then(parsers.parse('the run log reports "{expected}"'))
def run_log_reports(ctx, caplog, expected):
    assert expected in caplog.text


@given(
    parsers.parse(
        'video "{bad}" fails after retries and "{good}" succeeds'
    )
)
def one_bad_one_good(ctx, bad, good):
    ctx["discovered"] = [bad, good]
    ctx["downloader"] = _make_updater(
        ctx, {bad: None, good: _segments(3, good)}
    )


@when("`mc update` finishes")
def run_with_failure(ctx, caplog):
    with caplog.at_level(logging.INFO, logger="mc.ingest.update"):
        _run(ctx)


@then(
    parsers.parse('"{good}" is indexed, "{bad}" is in the ignore list')
)
def good_indexed_bad_ignored(ctx, good, bad):
    points, _ = ctx["client"].scroll(
        collection_name=ctx["collection"], limit=1000
    )
    chunks = [p for p in points if p.id != 0]
    assert chunks and all(p.payload["video_id"] == good for p in chunks)
    assert ctx["ignore_file"].exists()
    assert bad in ctx["ignore_file"].read_text()


@then("the run exits non-zero with a summary of the failure")
def exits_non_zero(ctx, caplog):
    assert ctx["exit_code"] == 1
    assert "bad1" in caplog.text and "failed" in caplog.text.lower()


def test_auth_failure_is_not_ignore_listed(tmp_path, caplog):
    """Story 5.6: stale-cookie (auth) failures get a distinct reason and the
    video is never added to the ignore list — the episode is retried after
    cookie rotation instead of skipped forever."""
    import logging as _logging
    import uuid as _uuid

    from qdrant_client import QdrantClient as QC

    client = QC(url="http://localhost:6333", timeout=30)
    collection = f"mc-bdd-{_uuid.uuid4().hex[:8]}"
    state = StateStore(tmp_path / "ingest.db")
    queue_dir = tmp_path / "queue"

    def stale_cookies(video_id: str):
        raise RuntimeError("Sign in to confirm you're not a bot")

    import mc.qdrant as q

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(q, "embed_passage", lambda texts: [[0.5] * 384 for _ in texts])
    try:
        with caplog.at_level(_logging.ERROR, logger="mc.ingest.update"):
            code = update.run_update(
                channels=("https://www.youtube.com/@test",),
                state=state,
                client=client,
                collection=collection,
                discover=lambda channels: ["stale1"],
                downloader=stale_cookies,
                metadata_fetch=lambda ids: {},
                lock_path=tmp_path / "update.lock",
                transcripts_dir=tmp_path / "transcripts",
                queue_dir=queue_dir,
            )
    finally:
        monkeypatch.undo()

    assert code == 1
    assert not (tmp_path / "transcripts" / "ignored_videos.txt").exists()
    assert "cookies likely stale" in caplog.text
    assert not state.is_processed("stale1")
    client.delete_collection(collection)


def test_no_subtitles_failure_is_ignore_listed(tmp_path):
    """Non-auth download failures keep the v3 ignore-list semantics."""
    import uuid as _uuid

    from qdrant_client import QdrantClient as QC

    client = QC(url="http://localhost:6333", timeout=30)
    collection = f"mc-bdd-{_uuid.uuid4().hex[:8]}"
    state = StateStore(tmp_path / "ingest.db")

    import mc.qdrant as q

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(q, "embed_passage", lambda texts: [[0.5] * 384 for _ in texts])
    try:
        code = update.run_update(
            channels=("https://www.youtube.com/@test",),
            state=state,
            client=client,
            collection=collection,
            discover=lambda channels: ["quiet1"],
            downloader=lambda vid: None,
            metadata_fetch=lambda ids: {},
            lock_path=tmp_path / "update.lock",
            transcripts_dir=tmp_path / "transcripts",
            queue_dir=tmp_path / "queue",
        )
    finally:
        monkeypatch.undo()

    assert code == 1
    assert "quiet1" in (tmp_path / "transcripts" / "ignored_videos.txt").read_text()
    client.delete_collection(collection)


def test_run_update_defaults_to_config_ingest_db_path(tmp_path, monkeypatch):
    """Review fix: the default state DB is config.INGEST_DB_PATH (state/ingest.db),
    not a separate data/ingest.db the API freshness check never reads."""
    from mc import config

    seen = {}

    class SpyState:
        def __init__(self, path):
            seen["path"] = path

    monkeypatch.setattr(update, "StateStore", SpyState)
    update.run_update(
        dry_run=True,
        state=None,
        discover=lambda channels: [],
        lock_path=tmp_path / "update.lock",
    )
    assert seen["path"] == config.INGEST_DB_PATH


def test_staging_runs_before_processed_marker(tmp_path, monkeypatch):
    """Review fix: staging (Mode B increment) must happen before the video is
    marked processed — a crash between the two must leave the video
    re-discoverable, and a staging failure must not ignore-list an
    already-indexed video. Qdrant boundary is faked: this is pure ordering."""
    state = StateStore(tmp_path / "ingest.db")
    queue_dir = tmp_path / "queue"
    processed_at_stage: list[bool] = []

    def failing_stage(chunks, queue_dir):
        processed_at_stage.append(state.is_processed("stage1"))
        raise OSError("disk full while staging")

    import mc.ingest.update as update_mod

    monkeypatch.setattr(update_mod, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(update_mod, "upsert_video", lambda *a, **k: None)
    monkeypatch.setattr(update_mod, "_stage_for_push", failing_stage)

    code = update_mod.run_update(
        channels=("https://www.youtube.com/@test",),
        state=state,
        client=object(),
        collection="fake",
        discover=lambda channels: ["stage1"],
        downloader=lambda vid: _segments(1, vid),
        metadata_fetch=lambda ids: {},
        lock_path=tmp_path / "update.lock",
        transcripts_dir=tmp_path / "transcripts",
        queue_dir=queue_dir,
    )

    assert code == 1
    # staging ran BEFORE the processed marker → video stays re-discoverable
    assert processed_at_stage == [False]
    assert not state.is_processed("stage1")
    # an indexed video must never be ignore-listed for a local staging error
    assert not (tmp_path / "transcripts" / "ignored_videos.txt").exists()


def test_failed_video_records_no_stats(tmp_path, monkeypatch):
    """Story 6.3: a video that fails download records no archive stats — the
    stats write happens only after a successful per-video commit."""
    import uuid as _uuid

    from qdrant_client import QdrantClient as QC

    from mc.ingest import update as update_mod
    from mc.state import StateStore

    client = QC(url="http://localhost:6333", timeout=30)
    collection = f"mc-bdd-{_uuid.uuid4().hex[:8]}"
    state = StateStore(tmp_path / "ingest.db")
    import mc.qdrant as q

    monkeypatch.setattr(q, "embed_passage", lambda texts: [[0.5] * 384 for _ in texts])
    try:
        code = update_mod.run_update(
            channels=("https://www.youtube.com/@test",),
            state=state,
            client=client,
            collection=collection,
            discover=lambda channels: ["badstat"],
            downloader=lambda vid: None,  # download fails → no commit
            metadata_fetch=lambda ids: {
                "badstat": {"duration": 5400, "episode": "971", "date": "2026-10-20"}
            },
            lock_path=tmp_path / "update.lock",
            transcripts_dir=tmp_path / "transcripts",
            queue_dir=tmp_path / "queue",
        )
    finally:
        monkeypatch.undo()

    assert code == 1
    assert state.archive_stats()["videos"] == 0
    client.delete_collection(collection)


def test_ignored_video_is_skipped_silently_on_later_runs(tmp_path):
    """A video already in ignored_videos.txt is a KNOWN uninjestable (e.g.
    permanently subtitle-less): re-attempting it every run failed the whole
    pass and fired the cron alert forever (observed with wtptWokZag8). The
    ignore list is a discovery-level exclusion — the video must be skipped
    before download, counted as neither new nor failed, and the run must
    exit 0."""
    import uuid as _uuid

    from qdrant_client import QdrantClient as QC

    client = QC(url="http://localhost:6333", timeout=30)
    collection = f"mc-bdd-{_uuid.uuid4().hex[:8]}"
    state = StateStore(tmp_path / "ingest.db")
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir(parents=True)
    (transcripts_dir / "ignored_videos.txt").write_text(
        "quiet1 # No subtitles available\n", encoding="utf-8"
    )
    downloaded: list[str] = []

    def downloader(video_id: str):
        downloaded.append(video_id)
        if video_id == "quiet1":
            raise AssertionError("ignored video must not reach the downloader")
        return [
            {"text": "linha um", "start": 0.0, "duration": 1.0},
            {"text": "linha dois", "start": 1.0, "duration": 1.0},
            {"text": "linha tres", "start": 2.0, "duration": 1.0},
        ]

    import mc.qdrant as q

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(q, "embed_passage", lambda texts: [[0.5] * 384 for _ in texts])
    try:
        code = update.run_update(
            channels=("https://www.youtube.com/@test",),
            state=state,
            client=client,
            collection=collection,
            discover=lambda channels: ["quiet1", "fresh1"],
            downloader=downloader,
            metadata_fetch=lambda ids: {},
            lock_path=tmp_path / "update.lock",
            transcripts_dir=transcripts_dir,
            queue_dir=tmp_path / "queue",
        )
    finally:
        monkeypatch.undo()
        client.delete_collection(collection)

    assert code == 0
    assert downloaded == ["fresh1"]  # the ignored video never reached download
    assert not state.is_processed("quiet1")  # skipped, not ingested
    assert state.is_processed("fresh1")
