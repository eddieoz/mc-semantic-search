"""Story 6.6 — deploy-runbook "hydrate archive stats" contract.

Pure file assertions (no network), same shape as test_crontab_doc.py: the
runbook is part of the deliverable, so pin the operator-facing facts the
archive-status epic demands — the command, the fetch flag, the runtime
hint and why fetching defaults to off.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNBOOK = (REPO_ROOT / "docs/tasks/deploy-runbook.md").read_text()


def test_runbook_has_stats_backfill_hydrate():
    assert "stats-backfill" in RUNBOOK, "runbook must document `mc stats-backfill`"


def test_runbook_documents_fetch_missing_flag():
    assert "--fetch-missing" in RUNBOOK, "runbook must cover the fetch flag"


def test_runbook_gives_a_runtime_hint():
    low = RUNBOOK.lower()
    assert any(
        w in low for w in ("segundos", "minutos", "seconds", "minutes", "runtime", "runtime")
    ), "runbook must hint at the expected runtime"


def test_runbook_explains_why_fetch_defaults_off():
    low = RUNBOOK.lower()
    # fetching touches youtube → network + bot-detection exposure, hence off
    assert any(w in low for w in ("bot", "network", "rede", "yt-dlp", "fetch"))


def test_runbook_documents_from_api_flag():
    assert "--from-api" in RUNBOOK, "runbook must document `mc stats-backfill --from-api`"


def test_runbook_documents_youtube_api_key_env():
    assert "YOUTUBE_API_KEY" in RUNBOOK, "runbook must name the YOUTUBE_API_KEY env var"


def test_runbook_mentions_key_restriction_or_quota():
    low = RUNBOOK.lower()
    assert any(
        w in low for w in ("quota", "cota", "restrict", "restringir")
    ), "runbook must cover restricting the key / the API quota"


def test_runbook_says_api_path_needs_no_cookies():
    low = RUNBOOK.lower()
    assert "cookie" in low, "runbook must state the --from-api path involves no cookies"
