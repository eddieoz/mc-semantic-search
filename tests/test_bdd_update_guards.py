"""BDD step definitions for tests/bdd/update-guards.feature (Story 10.2).

Container-free: every guard returns before qdrant is touched, so a plain
``object()`` stands in for the client and ``ensure_collection`` is patched
out. These scenarios pin the difference between "nothing new" (exit 0,
freshness refreshed) and "nothing could be ingested" (exit 1, freshness left
stale so the health gate keeps alerting).
"""

from __future__ import annotations

import logging

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from mc.ingest import update
from mc.ingest.download import IGNORE_LIST_FILENAME, add_to_ignore_list, is_ignored
from mc.state import StateStore

scenarios("bdd/update-guards.feature")

CHANNEL = "https://www.youtube.com/@test"


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    # No scenario may reach qdrant: the guards return before ensure_collection,
    # and the transient scenario stops at the failed download.
    monkeypatch.setattr(update, "ensure_collection", lambda *args, **kwargs: None)
    return {
        "state": StateStore(tmp_path / "state" / "ingest.db"),
        "tmp_path": tmp_path,
        "transcripts_dir": tmp_path / "transcripts",
        "queue_dir": tmp_path / "queue",
        "ignore_file": tmp_path / "transcripts" / IGNORE_LIST_FILENAME,
        "discovered": [],
        "downloader": lambda video_id: None,
        "exit_code": None,
        "log": "",
    }


def _ignore(ctx, video_id: str, reason: str = "No subtitles available") -> None:
    ctx["transcripts_dir"].mkdir(parents=True, exist_ok=True)
    add_to_ignore_list(video_id, ctx["ignore_file"], reason)


@given("channels return no video IDs")
def no_videos(ctx):
    ctx["discovered"] = []


@given(parsers.parse('channels return only video "{video_id}" and it is ignore-listed'))
def one_ignored(ctx, video_id):
    ctx["discovered"] = [video_id]
    _ignore(ctx, video_id)


@given(
    parsers.parse(
        'channels return only video "{video_id}" and its download is rate limited'
    )
)
def one_rate_limited(ctx, video_id):
    from mc.ingest.download import RateLimitedError

    ctx["discovered"] = [video_id]

    def downloader(video_id: str):
        raise RateLimitedError("HTTP Error 429: Too Many Requests")

    ctx["downloader"] = downloader


@given(parsers.parse('channels return only video "{video_id}" and it is already processed'))
def one_processed(ctx, video_id):
    ctx["discovered"] = [video_id]
    ctx["state"].mark_processed(video_id, "update")


@when("`mc update` runs")
def run_update(ctx, caplog):
    with caplog.at_level(logging.INFO, logger="mc.ingest.update"):
        ctx["exit_code"] = update.run_update(
            channels=(CHANNEL,),
            state=ctx["state"],
            client=object(),
            discover=lambda channels: ctx["discovered"],
            downloader=ctx["downloader"],
            metadata_fetch=lambda ids: {},
            lock_path=ctx["tmp_path"] / "update.lock",
            transcripts_dir=ctx["transcripts_dir"],
            queue_dir=ctx["queue_dir"],
        )
    ctx["log"] = caplog.text


@then(parsers.parse("the run exits {code:d}"))
def run_exits(ctx, code):
    assert ctx["exit_code"] == code


@then("the run log reports an empty discovery")
def log_empty_discovery(ctx):
    assert "no videos" in ctx["log"].lower()


@then("the run log reports no progress")
def log_no_progress(ctx):
    assert "no progress" in ctx["log"].lower()


@then("the run log reports a transient failure")
def log_transient(ctx):
    assert "transient" in ctx["log"].lower()


@then("the run log reports nothing to do")
def log_nothing_to_do(ctx):
    assert "nothing to do" in ctx["log"].lower()


@then(parsers.parse('video "{video_id}" is not ignore-listed'))
def not_ignore_listed(ctx, video_id):
    assert not is_ignored(video_id, ctx["ignore_file"])


@then("no successful ingest is recorded")
def no_success(ctx):
    assert ctx["state"].last_success("update") is None


@then("a successful ingest is recorded")
def success_recorded(ctx):
    assert ctx["state"].last_success("update") is not None
