"""Story 10.2 / 10.3 — `mc update` guards against a silently empty run.

Production symptom: the cron job exited 0 and recorded a successful ingest
every day while the archive stopped growing. Two holes made that possible —
an empty discovery looked like a healthy run, and every non-auth failure
(including transient 429s) was permanently ignore-listed, so a discovered
video that could not be downloaded simply vanished from all later runs.

These guards run before any qdrant access, so a plain ``object()`` stands in
for the client and no container is needed.
"""

from __future__ import annotations

import logging

import pytest

from mc.ingest import update
from mc.ingest.download import IGNORE_LIST_FILENAME, add_to_ignore_list, is_ignored
from mc.state import StateStore

CHANNEL = "https://www.youtube.com/@canal"


def _run(tmp_path, discovered, *, state=None, dry_run=False, downloader=None):
    state = state or StateStore(tmp_path / "state" / "ingest.db")
    return update.run_update(
        dry_run=dry_run,
        channels=(CHANNEL,),
        state=state,
        client=object(),
        discover=lambda channels: list(discovered),
        downloader=downloader,
        metadata_fetch=lambda ids: {},
        lock_path=tmp_path / "update.lock",
        transcripts_dir=tmp_path / "transcripts",
        queue_dir=tmp_path / "queue",
    )


def _ignore(tmp_path, video_id: str, reason: str = "No subtitles available") -> None:
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir(parents=True, exist_ok=True)
    add_to_ignore_list(video_id, transcripts / IGNORE_LIST_FILENAME, reason)


def test_empty_discovery_fails_loud_and_records_no_success(tmp_path, caplog):
    """An empty listing is an outage or a broken config, never 'up to date'."""
    state = StateStore(tmp_path / "state" / "ingest.db")
    with caplog.at_level(logging.INFO, logger="mc.ingest.update"):
        code = _run(tmp_path, [], state=state)

    assert code == 1
    assert "no videos" in caplog.text.lower()
    assert state.last_success("update") is None


def test_all_unprocessed_videos_ignored_is_no_progress(tmp_path, caplog):
    """Every discovered-but-unprocessed video being ignore-listed means the
    pipeline cannot make progress; that is a failure, not a clean run."""
    state = StateStore(tmp_path / "state" / "ingest.db")
    _ignore(tmp_path, "stuckvid001")

    with caplog.at_level(logging.INFO, logger="mc.ingest.update"):
        code = _run(tmp_path, ["stuckvid001"], state=state)

    assert code == 1
    assert "no progress" in caplog.text.lower()
    assert state.last_success("update") is None


def test_discovery_summary_counts_are_logged(tmp_path, caplog):
    """The operator must be able to tell 'nothing new' from 'nothing found'
    by reading the log line — the counts go out at INFO, before any return."""
    state = StateStore(tmp_path / "state" / "ingest.db")
    state.mark_processed("oldvid00001", "update")
    _ignore(tmp_path, "ignoredvid1")

    with caplog.at_level(logging.INFO, logger="mc.ingest.update"):
        code = _run(
            tmp_path,
            ["oldvid00001", "ignoredvid1", "newvid00001"],
            state=state,
            dry_run=True,
        )

    assert code == 0
    assert "3 discovered" in caplog.text
    assert "1 already processed" in caplog.text
    assert "1 ignored" in caplog.text
    assert "1 new videos" in caplog.text


def test_nothing_to_do_records_success(tmp_path, caplog):
    """A genuinely up-to-date archive is a success (and keeps /health fresh)."""
    state = StateStore(tmp_path / "state" / "ingest.db")
    state.mark_processed("oldvid00001", "update")

    with caplog.at_level(logging.INFO, logger="mc.ingest.update"):
        code = _run(tmp_path, ["oldvid00001"], state=state)

    assert code == 0
    assert "nothing to do" in caplog.text.lower()
    assert state.last_success("update") is not None


def test_transient_failure_is_not_ignore_listed(tmp_path, caplog, monkeypatch):
    """A 429/timeout must fail the run and be retried tomorrow — never
    written to ignored_videos.txt, which would hide the video forever."""
    from mc.ingest.download import RateLimitedError

    monkeypatch.setattr(update, "ensure_collection", lambda *a, **k: None)
    state = StateStore(tmp_path / "state" / "ingest.db")

    def downloader(video_id: str):
        raise RateLimitedError("HTTP Error 429: Too Many Requests")

    with caplog.at_level(logging.INFO, logger="mc.ingest.update"):
        code = _run(tmp_path, ["newvid00001"], state=state, downloader=downloader)

    assert code == 1
    assert "transient" in caplog.text.lower()
    assert not is_ignored("newvid00001", tmp_path / "transcripts" / IGNORE_LIST_FILENAME)
    assert state.last_success("update") is None


@pytest.mark.parametrize(
    "reason",
    [
        "HTTP Error 429: Too Many Requests",
        "read timed out",
        "Connection reset by peer",
        "HTTP Error 503: Service Unavailable",
        "HTTP Error 502: Bad Gateway",
        "HTTP Error 504: Gateway Timeout",
    ],
)
def test_transient_classification(reason):
    assert update.is_transient_failure(reason)


@pytest.mark.parametrize(
    "reason",
    [
        "No subtitles available in any supported language",
        "video unavailable",
        "HTTP Error 404: Not Found",
    ],
)
def test_permanent_failures_are_not_transient(reason):
    assert not update.is_transient_failure(reason)
