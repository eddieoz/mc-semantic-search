"""Story 1.5 — ingestion state store (state/ingest.db)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mc.state import StateStore


def _store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "ingest.db")


def test_mark_processed_idempotent(tmp_path):
    s = _store(tmp_path)
    s.mark_processed("vid1", run_type="backfill")
    s.mark_processed("vid1", run_type="backfill")
    assert s.is_processed("vid1")
    assert s.count_processed() == 1


def test_is_processed_false_for_unknown(tmp_path):
    s = _store(tmp_path)
    assert not s.is_processed("nope")


def test_last_success_only_via_record_success(tmp_path):
    s = _store(tmp_path)
    assert s.last_success() is None
    s.mark_processed("vid1", "update")  # must NOT set last_success
    assert s.last_success() is None
    s.record_success("update")
    ts = s.last_success()
    assert ts is not None
    s.record_success("update")
    assert s.last_success() >= ts


def test_wal_mode_and_busy_timeout(tmp_path):
    s = _store(tmp_path)
    s.mark_processed("vid1", "update")
    assert s.journal_mode() == "wal"
    assert s.busy_timeout_ms() > 0


def test_corrupt_db_recreated_with_warning(tmp_path):
    db = tmp_path / "ingest.db"
    db.write_bytes(b"not a sqlite database at all" * 10)
    s = StateStore(db)
    s.mark_processed("vid1", "backfill")  # must work despite corruption
    assert s.is_processed("vid1")


def test_missing_db_created(tmp_path):
    s = _store(tmp_path / "subdir" / "ingest.db")
    assert s.is_processed("x") is False


def test_processed_persisted(tmp_path):
    s1 = _store(tmp_path)
    s1.mark_processed("vid1", "update")
    s1.close()
    s2 = StateStore(tmp_path / "ingest.db")
    assert s2.is_processed("vid1")


def test_locked_db_raises_and_is_never_deleted(tmp_path, monkeypatch):
    """Review fix: a transient lock (OperationalError) must propagate, never
    trigger the corrupt-db unlink that wipes all bookkeeping."""

    import mc.state as state_mod

    monkeypatch.setattr(state_mod, "_BUSY_TIMEOUT_MS", 50)
    db = tmp_path / "ingest.db"
    holder = StateStore(db)
    holder.mark_processed("vid1", "update")
    holder.close()

    # Committed schema hole (DROP is DDL = autocommit) + a separate
    # connection holding the write lock → the next _open must write
    # (CREATE TABLE meta) and block.
    raw = sqlite3.connect(db)
    raw.execute("DROP TABLE meta")
    raw.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="lock"):
            StateStore(db)
    finally:
        raw.close()
    # file intact: the original data must survive the failed open
    assert StateStore(db).is_processed("vid1")


# --- Story 6.2 — video_stats + read-time aggregates -------------------------


def test_archive_stats_empty(tmp_path):
    assert _store(tmp_path).archive_stats() == {
        "videos": 0,
        "seconds": None,
        "last_episode": None,
        "last_episode_date": None,
    }


def test_archive_stats_mixed_nulls(tmp_path):
    s = _store(tmp_path)
    s.record_video_stats("v1", duration=3600, episode=970, date="2026-10-18")
    s.record_video_stats("v2", duration=5400, episode=971, date="2026-10-20")
    s.record_video_stats("v3")  # seen, stats unknown — all None

    stats = s.archive_stats()
    assert stats["videos"] == 3
    assert stats["seconds"] == 9000  # SUM ignores the NULL duration
    assert stats["last_episode"] == 971
    assert stats["last_episode_date"] == "2026-10-20"


def test_record_video_stats_non_null_wins_and_upserts(tmp_path):
    s = _store(tmp_path)
    s.record_video_stats("v1", duration=3600, episode=970, date="2026-10-18")
    s.record_video_stats("v1", duration=5400)  # better duration → upgraded
    assert s.archive_stats()["seconds"] == 5400
    # re-record with all-None → the known values are kept, row not duplicated
    s.record_video_stats("v1")
    stats = s.archive_stats()
    assert stats["videos"] == 1
    assert stats["seconds"] == 5400
    assert stats["last_episode"] == 970
    assert stats["last_episode_date"] == "2026-10-18"


def test_schema_change_preserves_existing_rows(tmp_path):
    """A pre-6.2 state DB (only processed + meta) auto-migrates on reopen:
    the new table appears and the old rows survive (no data loss)."""
    s1 = _store(tmp_path)
    s1.mark_processed("old1", "backfill")
    s1.record_success("update")
    s1.close()

    s2 = _store(tmp_path)
    assert s2.is_processed("old1")
    assert s2.last_success("update") is not None
    assert s2.archive_stats()["videos"] == 0


def test_episode_ordering_is_numeric_not_lexical(tmp_path):
    """1000 must beat 999 (as integers) even though "1000" < "999" lexically."""
    s = _store(tmp_path)
    s.record_video_stats("v999", episode=999, duration=100, date="2026-01-01")
    s.record_video_stats("v1000", episode=1000, duration=100, date="2026-01-02")
    stats = s.archive_stats()
    assert stats["last_episode"] == 1000
    assert stats["last_episode_date"] == "2026-01-02"


def test_set_video_episode_overwrites_and_clears(tmp_path):
    """The repair write: unlike record_video_stats's COALESCE (which never
    erases), set_video_episode is authoritative — None CLEARS the junk."""
    s = _store(tmp_path)
    s.record_video_stats("v1", duration=100, episode=301415, date="2025-03-16")
    s.set_video_episode("v1", 747)
    assert s.video_facts("v1")["episode"] == 747
    s.set_video_episode("v1", None)
    assert s.video_facts("v1")["episode"] is None
    assert s.video_facts("v1")["duration"] == 100  # untouched
    s.set_video_episode("absent", 5)  # no row → no-op, no crash
    assert s.video_facts("absent") == {"duration": None, "episode": None, "date": None}
