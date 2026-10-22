"""Story 0.3 — CLI entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "mc.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_help_lists_all_subcommands():
    result = _run_cli("--help")
    assert result.returncode == 0
    for cmd in [
        "backfill",
        "update",
        "push",
        "apply",
        "reindex",
        "serve",
        "stats-backfill",
    ]:
        assert cmd in result.stdout, f"missing subcommand {cmd}"


def test_unknown_subcommand_exits_nonzero_with_usage():
    result = _run_cli("definitely-not-a-command")
    assert result.returncode != 0
    assert "usage" in (result.stdout + result.stderr).lower()


def test_stats_backfill_rejects_unknown_flag():
    """Story 6.4: a bad flag on a real subcommand still exits non-zero."""
    result = _run_cli("stats-backfill", "--not-a-real-flag")
    assert result.returncode != 0
    assert "usage" in (result.stdout + result.stderr).lower()


def test_stats_backfill_summary_is_a_complete_line(tmp_path):
    """Story 6.4/6.6: the operator summary must end with a newline, not run
    into the shell prompt (regression: ``print(..., end="")`` on a string that
    has no trailing newline). Runs against a throwaway state dir."""
    env = {
        **os.environ,
        "MC_STATE_DIR": str(tmp_path / "state"),
        "MC_DATA_DIR": str(tmp_path / "data"),
        "MC_TRANSCRIPTS_DIR": str(tmp_path / "transcripts"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "mc.cli", "stats-backfill"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "stats-backfill:" in result.stdout
    assert result.stdout.endswith("\n"), repr(result.stdout[-60:])


def test_stats_backfill_from_db_threads_the_argument(tmp_path):
    """The whole point of --from-db is that the path actually reaches the
    backfill (regression: the flag was parsed but silently ignored). Seed a
    junk episode in a throwaway state DB, point --from-db at a legacy db whose
    title declares the truth, and confirm the row is healed end to end."""
    import sqlite3

    from mc.state import StateStore

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = StateStore(state_dir / "ingest.db")
    state.mark_processed("abc747junk1", "update")
    state.record_video_stats("abc747junk1", duration=100, episode=301415, date="2025-03-16")

    legacy = tmp_path / "mc_transcripts.db"
    con = sqlite3.connect(legacy)
    con.execute("CREATE TABLE transcripts (title TEXT, url TEXT, description TEXT)")
    con.execute(
        "INSERT INTO transcripts VALUES (?, ?, ?)",
        (
            "[ IRL ][ ep 747 ] Shopping na Estonia",
            "https://www.youtube.com/watch?v=abc747junk1&t=0",
            "t",
        ),
    )
    con.commit()
    con.close()

    env = {
        **os.environ,
        "MC_STATE_DIR": str(state_dir),
        "MC_DATA_DIR": str(tmp_path / "data"),
        "MC_TRANSCRIPTS_DIR": str(tmp_path / "transcripts"),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mc.cli",
            "stats-backfill",
            "--from-db",
            str(legacy),
            "--no-index",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "747 episodes re-derived" not in result.stdout  # only 1 row
    assert "1 episodes re-derived" in result.stdout
    assert StateStore(state_dir / "ingest.db").video_facts("abc747junk1")["episode"] == 747
