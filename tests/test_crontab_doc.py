"""Story 5.1 — crontab.example contract (pure file assertions, no network).

The example crontab is part of the deliverable: these tests pin the
properties the epic demands (flock, absolute logs, docker compose run,
snapshot/apply lines, notify hook, install instructions) so future edits
cannot silently drop them.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CRONTAB_PATH = REPO_ROOT / "crontab.example"
CRONTAB = CRONTAB_PATH.read_text()


def _active_lines(text: str) -> list[str]:
    """Non-empty, non-comment lines (env assignments included)."""
    return [
        ln
        for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def _job_lines(text: str) -> list[str]:
    """Active cron job lines (env assignments like NOTIFY_WEBHOOK= excluded)."""
    return [
        ln
        for ln in _active_lines(text)
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", ln)
    ]


def _schedule_of(job_line: str) -> list[str]:
    return job_line.split()[0:5]


def test_every_active_job_line_uses_flock():
    jobs = _job_lines(CRONTAB)
    assert len(jobs) >= 3, "expected update + snapshot + apply job lines"
    for line in jobs:
        assert "flock -n" in line, f"job line without flock -n: {line}"


def test_log_redirects_are_absolute():
    for line in _job_lines(CRONTAB):
        for match in re.finditer(r">>(\S+)", line):
            assert match.group(1).startswith("/"), (
                f"log path must be absolute (cron starts in /): {line}"
            )


def test_update_line_uses_docker_compose_run_and_update_command():
    update_jobs = [ln for ln in _job_lines(CRONTAB) if "update" in ln]
    assert update_jobs, "no daily update line found"
    assert any(
        "docker compose run --rm" in ln and re.search(r"\bupdate\b", ln)
        for ln in update_jobs
    ), "update line must run `docker compose run --rm ... update`"


def test_update_line_runs_daily():
    update_jobs = [ln for ln in _job_lines(CRONTAB) if "update" in ln]
    sched = _schedule_of(update_jobs[0])
    assert sched[0] == "15" and sched[1] == "6", "daily update expected at 06:15"
    assert sched[2] == "*" and sched[3] == "*" and sched[4] == "*"


def test_snapshot_line_references_backup_entry():
    snap_jobs = [ln for ln in _job_lines(CRONTAB) if "mc.backup" in ln]
    assert snap_jobs, "no snapshot line referencing `mc.backup` found"
    assert any(
        "create" in ln and "--keep 7" in ln for ln in snap_jobs
    ), "snapshot line must call the backup create entry with keep=7"


def test_snapshot_line_runs_nightly():
    snap_jobs = [ln for ln in _job_lines(CRONTAB) if "mc.backup" in ln]
    sched = _schedule_of(snap_jobs[0])
    assert sched[0] == "0" and sched[1] == "3", "nightly snapshot expected at 03:00"
    assert sched[2] == "*" and sched[3] == "*" and sched[4] == "*"


def test_apply_line_is_weekly_noop_sanity():
    apply_jobs = [ln for ln in _job_lines(CRONTAB) if re.search(r"\bapply\b", ln)]
    assert apply_jobs, "no weekly apply sanity line found"
    sched = _schedule_of(apply_jobs[0])
    assert sched[4] == "0", "apply sanity must run on Sunday (dow 0)"
    assert sched[2] == "*" and sched[3] == "*", "weekly = every week, not monthly"


def test_notify_hook_documented_with_curl_pattern():
    assert "NOTIFY_WEBHOOK" in CRONTAB, "notify webhook env must be documented"
    assert "curl" in CRONTAB, "notify hook must curl the webhook"
    assert "Telegram" in CRONTAB, "webhook should be documented as Telegram-compatible"


def test_freshness_gate_uses_the_ingest_state_not_the_log():
    """Story 10.5: the gate must ask the state DB when the last ingest
    succeeded. A `find -mmin` gate on the log can never fire — `>>` refreshes
    the file's mtime on every run, however empty the run was."""
    update_jobs = [ln for ln in _job_lines(CRONTAB) if "mc.cli update" in ln]
    assert update_jobs, "no daily update line found"
    line = update_jobs[0]
    assert "mc.cli health --max-age-hours" in line, (
        "update job must end with `mc.cli health --max-age-hours N`"
    )
    assert "-mmin" not in line, "log-mtime gate is meaningless with `>>`"


def test_jobs_use_the_mc_home_variable():
    """One place to edit when the checkout is not /opt/mc: the repo path is an
    env assignment, never hardcoded inside a job line."""
    assert re.search(r"^MC_HOME=", CRONTAB, re.MULTILINE), (
        "crontab must define MC_HOME (the repo checkout)"
    )
    for line in _job_lines(CRONTAB):
        assert "/opt/mc" not in line, f"job must use $MC_HOME, not a hardcoded path: {line}"


def test_install_instructions_present():
    assert "crontab -e" in CRONTAB, "must document installing via crontab -e"
