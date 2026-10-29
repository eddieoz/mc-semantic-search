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


# ---------------------------------------------------------------------------
# Story 8.2 — `stats-backfill --from-tabs`: the tab-driven date hydration pass
# runs BEFORE the classic pass (which mops rows/payloads up from the now-richer
# cache), and the whole command holds the flock RunLock so a cron overlap can
# never run two multi-hour hydration passes against the same index.
# ---------------------------------------------------------------------------


def test_stats_backfill_accepts_from_tabs_flag():
    from mc.cli import build_parser

    args = build_parser().parse_args(["stats-backfill", "--from-tabs"])
    assert args.from_tabs is True


def test_from_tabs_runs_the_tab_pass_before_the_classic_pass(
    tmp_path, monkeypatch, capsys
):
    # lock_path injected like update/apply/backfill tests: the real
    # data/stats-backfill.lock may be held by a live stats-backfill run
    import mc.ingest.stats_backfill as sb_mod
    from mc.cli import _cmd_stats_backfill, build_parser

    calls: list[str] = []
    monkeypatch.setattr(
        sb_mod,
        "run_tab_backfill",
        lambda **kw: calls.append("tabs") or {"summary": "stats-backfill tabs: TABS"},
    )
    monkeypatch.setattr(
        sb_mod,
        "run_stats_backfill",
        lambda **kw: calls.append("classic") or {"summary": "stats-backfill: CLASSIC"},
    )
    args = build_parser().parse_args(["stats-backfill", "--from-tabs", "--no-index"])

    assert _cmd_stats_backfill(args, lock_path=tmp_path / "sb.lock") == 0
    assert calls == ["tabs", "classic"]
    out = capsys.readouterr().out
    assert out.index("TABS") < out.index("CLASSIC")


def test_stats_backfill_refuses_to_overlap_with_a_held_lock(tmp_path, monkeypatch, capsys):
    from mc.cli import _cmd_stats_backfill, build_parser
    from mc.ingest.lock import AlreadyRunning

    def held(self):
        raise AlreadyRunning("another run holds the lock")

    monkeypatch.setattr("mc.ingest.lock.RunLock.acquire", held)
    args = build_parser().parse_args(["stats-backfill"])

    assert _cmd_stats_backfill(args, lock_path=tmp_path / "sb.lock") == 1
    assert "overlap" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# Story 9.4 — `stats-backfill --from-api`: the YouTube Data API fact pass,
# mutually exclusive with --from-tabs (D19 — two writers would interleave on
# the same cache/rows for zero benefit). Missing key and quota exhaustion
# fail LOUD with meaningful exit codes (D18) — an unconfigured key must not
# "succeed" having done nothing.
# ---------------------------------------------------------------------------


def test_stats_backfill_accepts_from_api_flag():
    from mc.cli import build_parser

    args = build_parser().parse_args(["stats-backfill", "--from-api"])
    assert args.from_api is True


def test_from_api_and_from_tabs_are_mutually_exclusive(capsys):
    """D19: one source per run — argparse rejects the pair outright."""
    import pytest

    from mc.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["stats-backfill", "--from-api", "--from-tabs"])
    err = capsys.readouterr().err
    assert "mutually exclusive" in err or "not allowed with" in err


def test_from_api_missing_key_fails_before_the_lock(tmp_path, monkeypatch, capsys):
    """An unconfigured key exits 1 with the runbook hint before contending
    for the lock — and before any module call (D18)."""
    import mc.config as config_mod
    import mc.ingest.stats_backfill as sb_mod
    from mc.cli import _cmd_stats_backfill, build_parser

    monkeypatch.setattr(config_mod, "YOUTUBE_API_KEY", None)

    def forbidden(**kw):
        raise AssertionError("run_api_backfill must not run without a key")

    acquire_calls: list[str] = []

    def recorded_acquire(self):
        acquire_calls.append("acquire")
        from mc.ingest.lock import RunLock

        return RunLock._orig_acquire(self) if hasattr(RunLock, "_orig_acquire") else None

    monkeypatch.setattr(sb_mod, "run_api_backfill", forbidden)
    monkeypatch.setattr(sb_mod, "run_stats_backfill", forbidden)
    args = build_parser().parse_args(["stats-backfill", "--from-api", "--no-index"])

    # the lock must never even be contended: acquire raising proves the
    # command bailed out before it
    def must_not_acquire(self):
        raise AssertionError("the lock must not be acquired without a key")

    monkeypatch.setattr("mc.ingest.lock.RunLock.acquire", must_not_acquire)

    assert _cmd_stats_backfill(args, lock_path=tmp_path / "sb.lock") == 1
    out = capsys.readouterr().out
    assert "YOUTUBE_API_KEY" in out
    assert "deploy-runbook" in out


def test_from_api_runs_the_api_pass_before_the_classic_pass(
    tmp_path, monkeypatch, capsys
):
    import mc.config as config_mod
    import mc.ingest.stats_backfill as sb_mod
    from mc.cli import _cmd_stats_backfill, build_parser

    monkeypatch.setattr(config_mod, "YOUTUBE_API_KEY", "test-key")

    calls: list[str] = []
    monkeypatch.setattr(
        sb_mod,
        "run_api_backfill",
        lambda **kw: calls.append("api") or {"summary": "stats-backfill api: API"},
    )
    monkeypatch.setattr(
        sb_mod,
        "run_stats_backfill",
        lambda **kw: calls.append("classic") or {"summary": "stats-backfill: CLASSIC"},
    )
    args = build_parser().parse_args(["stats-backfill", "--from-api", "--no-index"])

    assert _cmd_stats_backfill(args, lock_path=tmp_path / "sb.lock") == 0
    assert calls == ["api", "classic"]
    out = capsys.readouterr().out
    assert out.index("stats-backfill api:") < out.index("CLASSIC")


def test_from_api_quota_exhaustion_fails_loud(tmp_path, monkeypatch, capsys):
    """D18: quota silence would hide a misconfigured deployment. The partial
    work already committed stays (writes are idempotent) — but the classic
    pass must not run on top of a dead run."""
    import mc.config as config_mod
    import mc.ingest.stats_backfill as sb_mod
    from mc.cli import _cmd_stats_backfill, build_parser
    from mc.ingest.youtube_api import QuotaExceededError

    monkeypatch.setattr(config_mod, "YOUTUBE_API_KEY", "test-key")

    calls: list[str] = []

    def quota_dead(**kw):
        calls.append("api")
        raise QuotaExceededError("quotaExceeded", status=403)

    monkeypatch.setattr(sb_mod, "run_api_backfill", quota_dead)
    monkeypatch.setattr(
        sb_mod,
        "run_stats_backfill",
        lambda **kw: calls.append("classic") or {"summary": "stats-backfill: CLASSIC"},
    )
    args = build_parser().parse_args(["stats-backfill", "--from-api", "--no-index"])

    assert _cmd_stats_backfill(args, lock_path=tmp_path / "sb.lock") == 1
    assert calls == ["api"]  # no classic mop-up on top of a dead run
    assert "quota" in capsys.readouterr().out.lower()


def test_from_api_refuses_to_overlap_with_a_held_lock(tmp_path, monkeypatch, capsys):
    import mc.config as config_mod
    from mc.cli import _cmd_stats_backfill, build_parser
    from mc.ingest.lock import AlreadyRunning

    monkeypatch.setattr(config_mod, "YOUTUBE_API_KEY", "test-key")

    def held(self):
        raise AlreadyRunning("another run holds the lock")

    monkeypatch.setattr("mc.ingest.lock.RunLock.acquire", held)
    args = build_parser().parse_args(["stats-backfill", "--from-api"])

    assert _cmd_stats_backfill(args, lock_path=tmp_path / "sb.lock") == 1
    assert "overlap" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# Story 10.4 — logging is configured at all: with no handler the library
# default (WARNING) swallowed `mc update`'s INFO lines, so the "run summary"
# never reached the cron log and a run that ingested nothing looked healthy.
# Story 10.5 — `mc health` gates the cron on the state DB's last successful
# ingest; the old `find -mmin` log gate could never fire because `>>`
# refreshes the log file on every run.
# ---------------------------------------------------------------------------


def test_help_lists_health_subcommand():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "health" in result.stdout, "missing subcommand health"


def test_logging_is_configured_at_info_on_stderr(capsys):
    import logging

    from mc.cli import _configure_logging

    _configure_logging("INFO")
    logging.getLogger("mc.ingest.update").info("run summary: 0 new videos")

    assert "run summary: 0 new videos" in capsys.readouterr().err


def test_log_level_defaults_to_info_and_accepts_the_flag(monkeypatch):
    from mc.cli import build_parser

    monkeypatch.delenv("MC_LOG_LEVEL", raising=False)
    assert build_parser().parse_args(["health"]).log_level == "INFO"
    assert (
        build_parser().parse_args(["--log-level", "debug", "health"]).log_level
        == "DEBUG"
    )
    monkeypatch.setenv("MC_LOG_LEVEL", "warning")
    assert build_parser().parse_args(["health"]).log_level == "WARNING"


def _health_env(tmp_path):
    return {
        **os.environ,
        "MC_STATE_DIR": str(tmp_path / "state"),
        "MC_DATA_DIR": str(tmp_path / "data"),
        "MC_TRANSCRIPTS_DIR": str(tmp_path / "transcripts"),
    }


def test_health_is_stale_without_a_recorded_success(tmp_path):
    import json

    result = subprocess.run(
        [sys.executable, "-m", "mc.cli", "health", "--max-age-hours", "72"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_health_env(tmp_path),
    )

    assert result.returncode == 1
    body = json.loads(result.stdout)
    assert body["status"] == "stale"
    assert body["last_update"] is None
    assert body["max_age_hours"] == 72


def test_health_ok_after_a_recent_success(tmp_path):
    import json

    from mc.state import StateStore

    StateStore(tmp_path / "state" / "ingest.db").record_success("update")

    result = subprocess.run(
        [sys.executable, "-m", "mc.cli", "health", "--max-age-hours", "72"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_health_env(tmp_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    body = json.loads(result.stdout)
    assert body["status"] == "ok"
    assert body["hours_since_update"] is not None
    assert body["hours_since_update"] < 1
