"""BDD step definitions for tests/bdd/apply.feature (Story 3.7).

Runs against the local qdrant container (docker compose up -d qdrant).
Same embedder-faking pattern as the backfill BDD suite: `mc.qdrant.embed_passage`
is patched (upsert_video resolves it in that module namespace) with a fake
returning cheap 384-dim vectors, and a SPY records every embedded text so we
can assert the second `mc apply` run embeds nothing (applied-once tracking).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from qdrant_client import QdrantClient

from mc.state import StateStore

scenarios("bdd/apply.feature")

pytestmark = pytest.mark.integration


def _client() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=30)
    client.get_collections()
    return client


def _record(video_id: str, i: int) -> dict:
    return {
        "video_id": video_id,
        "text": f"chunk {i} do vídeo {video_id}",
        "url": f"https://youtube.com/watch?v={video_id}&t={i * 30}",
        "start_seconds": i * 30,
        "date": "2026-10-09",
        "episode": "ep",
        "channel": "Morning Crypto",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )


@pytest.fixture
def ctx(tmp_path):
    try:
        client = _client()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    ctx = {
        "client": client,
        "collection": f"mc-bdd-{uuid.uuid4().hex[:8]}",
        "state_db": tmp_path / "ingest.db",
        "queue_dir": queue_dir,
        "lock_path": tmp_path / "update.lock",
        "embedded_texts": [],
        "reports": [],
    }
    yield ctx
    client.delete_collection(ctx["collection"])
    client.close()


def _run_apply(ctx) -> dict:
    """One `mc apply` run with the dense embedder faked (spied)."""
    import mc.qdrant as q
    from mc.ingest.apply import run_apply

    def fake_embed_passage(texts):
        ctx["embedded_texts"].extend(texts)
        return [[0.5] * 384 for _ in texts]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(q, "embed_passage", fake_embed_passage)
    try:
        report = run_apply(
            queue_dir=ctx["queue_dir"],
            state=ctx["state_db"],
            client=ctx["client"],
            collection=ctx["collection"],
            lock_path=ctx["lock_path"],
        )
    finally:
        monkeypatch.undo()
    ctx["reports"].append(report)
    return report


def _point_count(ctx) -> int:
    client = ctx["client"]
    count = 0
    offset = None
    while True:
        scrolled, offset = client.scroll(
            collection_name=ctx["collection"], limit=1000, offset=offset
        )
        count += sum(1 for p in scrolled if p.id != 0)  # exclude meta sentinel
        if offset is None:
            return count


# --- Scenario: Increment applied exactly once --------------------------------


@given(parsers.parse("a queued increment with {n:d} chunks"))
def queued_increment(n, ctx):
    records = [_record("bddvid00001", i) for i in range(n)]
    _write_jsonl(ctx["queue_dir"] / "2026-10-09.jsonl", records)
    ctx["records"] = records


@when('"mc apply" runs twice')
def runs_twice(ctx):
    _run_apply(ctx)
    _run_apply(ctx)


@then("the second run is a no-op (chunks tracked by content hash)")
def second_run_is_noop(ctx):
    first, second = ctx["reports"]
    assert first["applied_chunks"] == len(ctx["records"])
    assert second["applied_chunks"] == 0
    # embedding spy: exactly the 3 chunks, ZERO new texts on run 2
    assert len(ctx["embedded_texts"]) == len(ctx["records"])
    assert "nothing to apply" in second["notes"]


@then(parsers.parse("all {n:d} chunks are searchable"))
def chunks_searchable(n, ctx):
    assert _point_count(ctx) == n


# --- Scenario: Corrupt increment is quarantined -------------------------------


@given("a malformed JSONL file in the queue directory")
def malformed_file(ctx):
    broken = ctx["queue_dir"] / "broken.jsonl"
    broken.write_text('{"video_id": "x"\n{not valid json\n', encoding="utf-8")
    ctx["broken"] = broken


@given(parsers.parse("another queued file with {n:d} valid chunks"))
def another_queued_file(n, ctx):
    records = [_record("bddvid00002", i) for i in range(n)]
    _write_jsonl(ctx["queue_dir"] / "2026-10-10.jsonl", records)
    ctx["other_records"] = records


@when('"mc apply" runs')
def apply_runs(ctx):
    _run_apply(ctx)


@then("the file is moved to queue/failed/ with the error logged")
def corrupt_file_quarantined(ctx, caplog):
    failed = ctx["queue_dir"] / "failed" / "broken.jsonl"
    assert failed.exists()
    assert not ctx["broken"].exists()
    assert ctx["reports"][-1]["failed_files"] == 1
    assert any("broken.jsonl" in r.getMessage() for r in caplog.records)


@then("other queued files are still applied")
def other_files_applied(ctx):
    report = ctx["reports"][-1]
    assert report["applied_chunks"] == len(ctx["other_records"])
    assert report["failed_files"] == 1
    assert _point_count(ctx) == len(ctx["other_records"])
    # the valid file was applied, not quarantined
    assert (ctx["queue_dir"] / "2026-10-10.jsonl").exists()


# --- Scenario: Applied increments update archive stats (Story 6.3) -----------


@given(
    parsers.parse(
        "an increment file carries a video with duration {duration:d}, "
        'episode "{episode}", date "{date}"'
    )
)
def increment_with_stats(ctx, duration, episode, date):
    records = [_record("statvid1", i) for i in range(3)]
    for r in records:
        r["duration"] = duration
        r["episode"] = episode
        r["date"] = date
    _write_jsonl(ctx["queue_dir"] / "2026-10-20.jsonl", records)


@when('"mc apply" processes it')
def apply_processes_once(ctx):
    _run_apply(ctx)


@then("the state store records that video's stats")
def apply_records_video_stats(ctx):
    stats = StateStore(ctx["state_db"]).archive_stats()
    assert stats["videos"] == 1
    assert stats["last_episode_date"] == "2026-10-20"


@then(parsers.parse('"archive_stats()" reports seconds {seconds:d} and last episode {episode:d}'))
def apply_stats_report(ctx, seconds, episode):
    stats = StateStore(ctx["state_db"]).archive_stats()
    assert stats["seconds"] == seconds
    assert stats["last_episode"] == episode


# --- Unit regressions: Mode B queue validation (no qdrant needed) ------------


def test_group_chunks_accepts_old_record_without_duration():
    """Story 6.1 regression: pre-duration increment records (Mode A servers
    before this story) still apply cleanly — duration defaults to None."""
    from mc.ingest.apply import _group_chunks

    record = _record("vid61", 0)  # _record never carried a duration key
    grouped = _group_chunks([record])
    assert grouped["vid61"][0].duration is None


def test_group_chunks_threads_duration_from_new_record():
    from mc.ingest.apply import _group_chunks

    record = _record("vid61", 0)
    record["duration"] = 5400
    grouped = _group_chunks([record])
    assert grouped["vid61"][0].duration == 5400


def test_group_chunks_nulls_garbage_duration():
    """A non-numeric duration from the untrusted queue degrades to None —
    never a string, which SQLite would store as TEXT and SUM() would then
    count as 0, silently under-reporting the archive's total duration."""
    from mc.ingest.apply import _group_chunks

    record = _record("vid61", 0)
    record["duration"] = "not-a-number"
    grouped = _group_chunks([record])
    assert grouped["vid61"][0].duration is None


def test_group_chunks_truncates_float_duration():
    from mc.ingest.apply import _group_chunks

    record = _record("vid61", 0)
    record["duration"] = 5400.9
    grouped = _group_chunks([record])
    assert grouped["vid61"][0].duration == 5400


def test_group_chunks_rejects_non_http_url():
    """Queue records are untrusted: url must be http(s) (producer emits YouTube
    watch URLs) or the whole file is quarantined."""
    from mc.ingest.apply import _group_chunks

    record = _record("vidX", 0)
    record["url"] = "file:///etc/passwd"
    with pytest.raises(ValueError, match="http"):
        _group_chunks([record])

    javascript = _record("vidX", 0)
    javascript["url"] = "javascript:alert(1)"
    with pytest.raises(ValueError, match="http"):
        _group_chunks([javascript])


def test_group_chunks_rejects_empty_or_oversized_text():
    from mc.ingest.apply import _group_chunks

    empty = _record("vidX", 0)
    empty["text"] = ""
    with pytest.raises(ValueError, match="text"):
        _group_chunks([empty])

    big = _record("vidX", 0)
    big["text"] = "x" * 50_001
    with pytest.raises(ValueError, match="50,?000"):
        _group_chunks([big])


def test_oversized_increment_file_quarantined_without_loading(tmp_path):
    """A >100 MB increment must be quarantined via stat, never read_text()."""
    from mc.ingest import apply as apply_mod

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    big = queue_dir / "big.jsonl"
    with open(big, "wb") as fh:  # sparse — no data written
        fh.truncate(apply_mod._MAX_FILE_BYTES + 1)

    report = {"applied_chunks": 0, "files_processed": 0, "failed_files": 0, "notes": []}
    apply_mod._apply_file(
        queue_dir, big, state=object(), client=object(), collection="x", report=report
    )

    assert report["failed_files"] == 1
    assert (queue_dir / "failed" / "big.jsonl").exists()
    assert not big.exists()


def test_apply_records_partial_stats_without_duration(ctx):
    """Story 6.3: an increment without a duration still records stats (duration
    NULL is acceptable — Story 6.4 fills it later); the episode still lands."""
    records = [_record("nodur", i) for i in range(2)]
    for r in records:
        r["episode"] = "971"  # episode known, duration absent (pre-6.1 style)
    _write_jsonl(ctx["queue_dir"] / "2026-10-22.jsonl", records)
    _run_apply(ctx)

    stats = StateStore(ctx["state_db"]).archive_stats()
    assert stats["videos"] == 1
    assert stats["seconds"] is None  # partial: NULL duration is fine
    assert stats["last_episode"] == 971


def test_stats_write_failure_does_not_fail_apply(ctx, monkeypatch):
    """Story 6.3: a stats-write failure is logged and non-fatal — the video was
    already durably upserted, so the apply run still succeeds."""

    def boom(*_a, **_k):
        raise RuntimeError("stats db full")

    monkeypatch.setattr(StateStore, "record_video_stats", boom)
    records = [_record("boomstat", i) for i in range(2)]
    _write_jsonl(ctx["queue_dir"] / "2026-10-21.jsonl", records)

    report = _run_apply(ctx)
    assert report["applied_chunks"] == 2  # upsert committed despite the failure
    assert _point_count(ctx) == 2


# --- Mode B checkpoint: apply must mark videos processed ---------------------


def test_apply_marks_video_processed(ctx):
    """After apply the video is mark_processed — so a server-side `mc update`
    (Mode A) will not re-attempt an already-indexed video: resumability
    across both ingest modes."""
    from mc.state import StateStore

    _write_jsonl(
        ctx["queue_dir"] / "2026-10-23.jsonl",
        [_record("bddvid00002", i) for i in range(2)],
    )
    _run_apply(ctx)
    assert StateStore(ctx["state_db"]).is_processed("bddvid00002")


def test_apply_backfills_processed_for_already_applied_chunks(ctx):
    """A file whose chunks were all applied earlier (hash keys present but the
    video never mark_processed — the pre-fix Mode B state) still backfills
    `processed` on re-apply, instead of early-returning and leaving the video
    invisible to Mode A resumability forever."""
    from mc.ingest.apply import _content_hash
    from mc.state import StateStore

    records = [_record("bddvid00003", i) for i in range(2)]
    _write_jsonl(ctx["queue_dir"] / "2026-10-23b.jsonl", records)
    state = StateStore(ctx["state_db"])
    for record in records:  # production's pre-fix state: hash applied, no processed
        state.record_key(f"applied:{_content_hash(record)}", "2026-10-23T00:00:00")

    report = _run_apply(ctx)

    assert state.is_processed("bddvid00003")
    assert report["applied_chunks"] == 0
