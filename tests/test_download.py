"""Story 3.2 tests — yt-dlp transcript download (all network mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mc.ingest.download import (
    LANGUAGE_PRIORITY,
    DownloadError,
    add_to_ignore_list,
    download_transcript,
    is_ignored,
    parse_json3_subtitles,
)

FIXTURES = Path(__file__).parent / "fixtures" / "download"
VIDEO_ID = "abc123video"


# Expected output hard-coded from the real v3 parse_json3_subtitles
# (verified against /home/eddieoz/Projects/git/live_gerard/extras/
# import_transcriptions_v3.py on this exact fixture).
EXPECTED_SEGMENTS = [{"text": "Hello world", "start": 1.2, "duration": 3.5}]


class Recorder:
    """Callable standing in for the yt-dlp runner; records every call."""

    def __init__(self, behavior=None):
        self.calls: list[tuple[str, dict]] = []
        self._behavior = behavior

    def __call__(self, url: str, ydl_opts: dict) -> None:
        self.calls.append((url, ydl_opts))
        if self._behavior is not None:
            self._behavior(url, ydl_opts)

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _write_cookies(tmp_path: Path) -> str:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    return str(cookies)


def test_language_priority_list_and_per_attempt_options(tmp_path):
    transcripts_dir = tmp_path / "transcripts"
    cookies_path = _write_cookies(tmp_path)
    rec = Recorder()  # never produces subtitle files -> every language tried

    result = download_transcript(
        VIDEO_ID,
        cookies_path=cookies_path,
        transcripts_dir=transcripts_dir,
        runner=rec,
    )

    assert result is None
    assert list(LANGUAGE_PRIORITY) == ["pt", "pt-BR", "en", "en-US"]
    assert rec.call_count == len(LANGUAGE_PRIORITY)

    for (url, ydl_opts), lang in zip(rec.calls, LANGUAGE_PRIORITY, strict=True):
        assert url == f"https://www.youtube.com/watch?v={VIDEO_ID}"
        assert ydl_opts["subtitleslangs"] == [lang]
        assert ydl_opts["cookiefile"] == cookies_path
        assert ydl_opts["subtitlesformat"] == "json3"
        assert ydl_opts["writesubtitles"] is True
        assert ydl_opts["writeautomaticsub"] is True
        assert ydl_opts["skip_download"] is True
        # Subtitle-only runs must survive extraction with zero media formats:
        # when the n-challenge can't be solved (no JS runtime, e.g. the
        # production server), yt-dlp raises a fatal "Requested format is not
        # available" before writing subtitles unless this flag is set.
        assert ydl_opts["ignore_no_formats_error"] is True
        # Failures must raise into the classification path, never be swallowed.
        assert ydl_opts["ignoreerrors"] is False


def test_cached_srt_short_circuits(tmp_path):
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    cached = [{"text": "cached", "start": 0.0, "duration": 1.0}]
    (transcripts_dir / f"{VIDEO_ID}.srt").write_text(
        json.dumps(cached), encoding="utf-8"
    )
    rec = Recorder(behavior=lambda *_: pytest.fail("yt-dlp must not be invoked"))

    result = download_transcript(
        VIDEO_ID,
        cookies_path=_write_cookies(tmp_path),
        transcripts_dir=transcripts_dir,
        runner=rec,
    )

    assert result == cached
    assert rec.call_count == 0


def test_rate_limit_backoff_then_next_language(tmp_path, monkeypatch):
    jitterless = lambda a, b: 0.0  # noqa: E731
    monkeypatch.setattr("random.uniform", jitterless)
    delays: list[float] = []
    rec = Recorder(
        behavior=lambda *_: (_ for _ in ()).throw(RuntimeError("HTTP Error 429: Too Many Requests"))
    )

    result = download_transcript(
        VIDEO_ID,
        cookies_path=_write_cookies(tmp_path),
        transcripts_dir=tmp_path / "transcripts",
        sleeper=delays.append,
        runner=rec,
    )

    assert result is None
    # Per language: 1 initial attempt + 3 backoff retries.
    assert rec.call_count == 4 * (1 + 3)
    # 2**retry * 10 with zero jitter, then move to the next language.
    assert delays == [10, 20, 40] * len(LANGUAGE_PRIORITY)


def test_permanent_failure_appends_ignore_list_once(tmp_path):
    transcripts_dir = tmp_path / "transcripts"
    cookies_path = _write_cookies(tmp_path)

    kwargs = dict(
        cookies_path=cookies_path,
        transcripts_dir=transcripts_dir,
        runner=Recorder(),  # never yields subtitles
    )
    assert download_transcript(VIDEO_ID, **kwargs) is None
    assert download_transcript(VIDEO_ID, **kwargs) is None

    lines = (transcripts_dir / "ignored_videos.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert lines == [f"{VIDEO_ID} # No subtitles available"]


def test_ignore_list_helpers(tmp_path):
    ignore_file = tmp_path / "ignored_videos.txt"

    assert is_ignored(VIDEO_ID, ignore_file) is False

    add_to_ignore_list(VIDEO_ID, ignore_file, "No subtitles available")
    add_to_ignore_list(VIDEO_ID, ignore_file, "No subtitles available")  # idempotent

    content = ignore_file.read_text(encoding="utf-8").splitlines()
    assert content == [f"{VIDEO_ID} # No subtitles available"]
    assert is_ignored(VIDEO_ID, ignore_file) is True
    # Reason suffix does not confuse the id match.
    assert not is_ignored(f"{VIDEO_ID}x", ignore_file)


def test_non_rate_limit_failure_propagates_not_ignored(tmp_path):
    """With ignoreerrors=False a real download failure must raise (so run_update
    can classify it, e.g. stale cookies) instead of masquerading as
    "no subtitles" and silently landing on the ignore list."""
    rec = Recorder(
        behavior=lambda *_: (_ for _ in ()).throw(
            RuntimeError("Sign in to confirm you're not a bot")
        )
    )
    transcripts_dir = tmp_path / "transcripts"

    with pytest.raises(RuntimeError, match="Sign in"):
        download_transcript(
            VIDEO_ID,
            cookies_path=_write_cookies(tmp_path),
            transcripts_dir=transcripts_dir,
            runner=rec,
        )

    assert rec.call_count == 1  # first failure aborts — no language retries
    assert not (transcripts_dir / "ignored_videos.txt").exists()


def test_ignore_list_newlines_collapsed(tmp_path):
    """Embedded newlines in id/reason must not forge extra ignore-list lines."""
    ignore_file = tmp_path / "ignored_videos.txt"

    add_to_ignore_list(f"{VIDEO_ID}\ninjected", ignore_file, "boom\r\nfake-entry")

    lines = ignore_file.read_text(encoding="utf-8").splitlines()
    assert lines == [f"{VIDEO_ID} injected # boom fake-entry"]
    assert is_ignored(VIDEO_ID, ignore_file) is True


def test_group_readable_cookies_are_tightened_to_0600(tmp_path, caplog):
    import logging

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    cookies.chmod(0o644)

    with caplog.at_level(logging.WARNING, logger="mc.ingest.download"):
        download_transcript(
            VIDEO_ID,
            cookies_path=str(cookies),
            transcripts_dir=tmp_path / "transcripts",
            runner=Recorder(),
        )

    assert cookies.stat().st_mode & 0o777 == 0o600
    assert any("0600" in r.message for r in caplog.records)


def test_parse_json3_golden():
    assert parse_json3_subtitles(FIXTURES / "sample.json3") == EXPECTED_SEGMENTS


def test_missing_cookies_raises_before_runner(tmp_path):
    rec = Recorder(behavior=lambda *_: pytest.fail("runner must not be called"))

    with pytest.raises(DownloadError, match="cookies"):
        download_transcript(
            VIDEO_ID,
            cookies_path=str(tmp_path / "missing.txt"),
            transcripts_dir=tmp_path / "transcripts",
            runner=rec,
        )

    assert rec.call_count == 0


def test_successful_download_caches_and_cleans_up(tmp_path):
    transcripts_dir = tmp_path / "transcripts"
    raw = (FIXTURES / "sample.json3").read_text(encoding="utf-8")

    def produces_subtitle(url: str, ydl_opts: dict) -> None:
        lang = ydl_opts["subtitleslangs"][0]
        (transcripts_dir / f"{VIDEO_ID}.{lang}.json3").write_text(
            raw, encoding="utf-8"
        )

    rec = Recorder(produces_subtitle)

    result = download_transcript(
        VIDEO_ID,
        cookies_path=_write_cookies(tmp_path),
        transcripts_dir=transcripts_dir,
        runner=rec,
    )

    assert result == EXPECTED_SEGMENTS
    # First language wins; cached .srt written, temp json3 removed.
    assert rec.call_count == 1
    cached = transcripts_dir / f"{VIDEO_ID}.srt"
    assert cached.exists()
    assert json.loads(cached.read_text(encoding="utf-8")) == EXPECTED_SEGMENTS
    assert not (transcripts_dir / f"{VIDEO_ID}.pt.json3").exists()
