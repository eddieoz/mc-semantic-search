"""Story 2.2 — yt-dlp metadata enrichment with a resumable on-disk cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mc.ingest.metadata import (
    _normalize,
    _parse_episode,
    duration_int,
    enrich,
    episode_int,
)


def _payload(video_id: str, **overrides) -> dict:
    raw = {
        "upload_date": "20260109",
        "title": f"BITCOIN: O PLANO SECRETO #{video_id}",
        "uploader": "Morning Crypto",
        "channel": "Morning Crypto Channel",
    }
    raw.update(overrides)
    return raw


def test_fetch_invoked_once_per_unique_video(tmp_path: Path):
    calls: list[str] = []

    def fetch(video_id: str) -> dict:
        calls.append(video_id)
        return _payload(video_id)

    result = enrich(["a", "b", "a"], cache_dir=tmp_path, fetch=fetch)

    assert calls == ["a", "b"]
    assert set(result) == {"a", "b"}


def test_result_cached_second_call_skips_fetch(tmp_path: Path):
    calls: list[str] = []

    def fetch(video_id: str) -> dict:
        calls.append(video_id)
        return _payload(video_id)

    first = enrich(["a", "b"], cache_dir=tmp_path, fetch=fetch)
    calls.clear()

    second = enrich(["a", "b"], cache_dir=tmp_path, fetch=fetch)

    assert calls == []
    assert second == first


def test_upload_date_normalized_and_channel(tmp_path: Path):
    def fetch(video_id: str) -> dict:
        return {
            "upload_date": "20260109",
            "title": "SOME TITLE #123",
            "uploader": "Morning Crypto",
            "channel": "Channel Field",
        }

    meta = enrich(["abc"], cache_dir=tmp_path, fetch=fetch)["abc"]

    assert meta["date"] == "2026-01-09"
    assert meta["channel"] == "Morning Crypto"
    assert (tmp_path / "abc.json").exists()
    assert json.loads((tmp_path / "abc.json").read_text()) == meta


def test_episode_parsed_from_title_digits(tmp_path: Path):
    def fetch(video_id: str) -> dict:
        return _payload(video_id, title="BITCOIN: O PLANO SECRETO #1234")

    meta = enrich(["x1"], cache_dir=tmp_path, fetch=fetch)["x1"]
    assert meta["episode"] == "1234"


def test_episode_none_when_title_has_no_digits(tmp_path: Path):
    def fetch(video_id: str) -> dict:
        return _payload("x2", title="BITCOIN: O PLANO SECRETO")

    meta = enrich(["x2"], cache_dir=tmp_path, fetch=fetch)["x2"]
    assert meta["episode"] is None


def test_failure_retries_three_times_then_missing(tmp_path: Path):
    calls: list[str] = []

    def fetch(video_id: str) -> dict:
        calls.append(video_id)
        if video_id == "bad":
            raise RuntimeError("network down")
        return _payload(video_id)

    result = enrich(["bad", "good"], cache_dir=tmp_path, fetch=fetch, backoff=(0, 0, 0))

    assert calls.count("bad") == 3
    assert "good" in calls
    assert set(result) == {"good"}
    assert (tmp_path / "missing.txt").read_text().splitlines() == ["bad"]


# --- Story 6.1 — duration flows through the metadata normalizer -------------


def test_normalize_keeps_int_duration():
    meta = _normalize({"title": "EP #1", "duration": 5400})
    assert meta["duration"] == 5400


def test_normalize_float_duration_truncated_to_int():
    meta = _normalize({"title": "EP #1", "duration": 5400.9})
    assert meta["duration"] == 5400


def test_normalize_missing_duration_is_none():
    assert _normalize({"title": "EP #1"})["duration"] is None


def test_normalize_garbage_duration_is_none():
    # yt-dlp sometimes returns "N/A" or a non-numeric string — never crash.
    assert _normalize({"title": "EP #1", "duration": "N/A"})["duration"] is None
    assert _normalize({"title": "EP #1", "duration": None})["duration"] is None


def test_duration_round_trips_through_cache_file(tmp_path: Path):
    def fetch(video_id: str) -> dict:
        return _payload(video_id, duration=3600)

    first = enrich(["vid61"], cache_dir=tmp_path, fetch=fetch)["vid61"]
    assert first["duration"] == 3600
    # the on-disk cache entry carries duration, so a re-fetch loads it back
    cached = json.loads((tmp_path / "vid61.json").read_text())
    assert cached["duration"] == 3600
    calls: list[str] = []

    def fetch_again(video_id: str) -> dict:
        calls.append(video_id)
        return _payload(video_id, duration=999)

    second = enrich(["vid61"], cache_dir=tmp_path, fetch=fetch_again)["vid61"]
    assert calls == []  # served from cache, fetcher never invoked
    assert second["duration"] == 3600


# --- Story 6.2 — episode_int (numeric episode for MAX) ----------------------


def test_episode_int_parses_number_string():
    assert episode_int("971") == 971
    assert episode_int("1000") == 1000
    assert episode_int("  42  ") == 42  # tolerant of surrounding whitespace


def test_episode_int_none_for_garbage():
    assert episode_int("garbage") is None
    assert episode_int("") is None
    assert episode_int(None) is None


# --- Episode parsing — the archive's real title format ----------------------
#
# Titles declare the episode as ``[ Morning Crypto ][ ep 441 ]`` / ``EP. N``
# (sometimes with a doubled space). The old bare-digit fallback turned dates,
# prices and milestones into episodes — "[ IRL ][ ep 567 ] DeSci Nordics
# 2024" became "2024", "Nunca use 123456" became 123456 and poisoned
# MAX(episode) on the landing page. A title must DECLARE its episode; bare
# digit runs are never episodes.


def test_episode_parsed_from_ep_word_marker():
    assert _parse_episode("[ IRL ][ ep 567 ] DeSci Nordics 2024") == "567"
    assert _parse_episode("[ Morning Crypto ][ ep  371 ] Nostr") == "371"  # doubled space
    assert _parse_episode("[ Morning Crypto ][ EP. 372 ] Treta, FTX") == "372"


def test_episode_prefers_hash_marker_over_ep_word():
    assert _parse_episode("EP #971 (also says ep 500)") == "971"


def test_episode_none_for_bare_digit_runs():
    assert _parse_episode("Nunca use 123456  Proteja seu Access Code") is None
    assert _parse_episode("Tinha pool de Bitcoin em 2009?") is None
    assert _parse_episode("Fireside Chat: ETH Tallinn April/2024") is None
    assert _parse_episode("BITCOIN: O PLANO SECRETO") is None


# --- Story 6.1 — duration_int (numeric seconds for storage/aggregates) ------


def test_duration_int_coerces_numbers():
    assert duration_int(5400) == 5400
    assert duration_int(5400.9) == 5400  # float truncates, like yt-dlp's seconds
    assert duration_int("5400") == 5400  # lenient for a stringified queue value


def test_duration_int_none_for_garbage():
    assert duration_int(None) is None
    assert duration_int("") is None
    assert duration_int("N/A") is None
    assert duration_int("not-a-number") is None
    assert duration_int(object()) is None


def test_duration_int_none_for_non_finite_floats():
    # int(float("inf")) raises OverflowError, int(float("nan")) ValueError —
    # both must degrade to None, never crash a stats write.
    assert duration_int(float("inf")) is None
    assert duration_int(float("-inf")) is None
    assert duration_int(float("nan")) is None


# --- _default_fetch — the real yt-dlp subprocess -----------------------------
#
# YouTube's bot gating can leave a video with NO playable formats (only
# storyboards) even though the extractor knows its metadata. The default
# format selector then aborts the whole --dump-single-json with "Requested
# format is not available" — so metadata-only fetches must not require a
# format, and failures must carry yt-dlp's stderr (an bare "exit status 1"
# is undiagnosable for the operator).


def _fake_run(monkeypatch, *, stdout: str | None = None, stderr: str | None = None):
    """Replace subprocess.run with a recorder; returns the captured argv list."""
    import subprocess as sp

    import mc.ingest.metadata as metadata_mod

    argvs: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        argvs.append(list(cmd))
        if stderr is not None:
            raise sp.CalledProcessError(returncode=1, cmd=cmd, stderr=stderr)
        return sp.CompletedProcess(cmd, returncode=0, stdout=stdout or "{}", stderr="")

    monkeypatch.setattr(metadata_mod.subprocess, "run", fake_run)
    return argvs


def test_default_fetch_never_requires_a_playable_format(monkeypatch):
    from mc.ingest.metadata import _default_fetch

    argvs = _fake_run(monkeypatch, stdout='{"title": "EP #88", "duration": 9114}')

    raw = _default_fetch("--5z_jFh-_g")

    assert raw["duration"] == 9114
    (argv,) = argvs
    assert "--ignore-no-formats-error" in argv  # bot-gated videos still yield metadata
    assert "--dump-single-json" in argv
    assert argv[-1] == "https://www.youtube.com/watch?v=--5z_jFh-_g"


def test_default_fetch_failure_carries_stderr(monkeypatch):
    from mc.ingest.metadata import _default_fetch

    _fake_run(
        monkeypatch,
        stderr="ERROR: [youtube] --5z_jFh-_g: Requested format is not available.",
    )

    with pytest.raises(RuntimeError, match="Requested format is not available"):
        _default_fetch("--5z_jFh-_g")


def test_default_fetch_uses_cookies_file_when_present(monkeypatch, tmp_path):
    from mc.ingest.metadata import _default_fetch

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    monkeypatch.setattr("mc.config.COOKIES_PATH", str(cookies))
    argvs = _fake_run(monkeypatch, stdout="{}")

    _default_fetch("vid")

    assert argvs[0][argvs[0].index("--cookies") + 1] == str(cookies)


def test_default_fetch_without_cookies_file_still_fetches(monkeypatch, tmp_path):
    from mc.ingest.metadata import _default_fetch

    monkeypatch.setattr("mc.config.COOKIES_PATH", str(tmp_path / "absent-cookies.txt"))
    argvs = _fake_run(monkeypatch, stdout="{}")

    _default_fetch("vid")

    assert "--cookies" not in argvs[0]  # cookies are optional, not required
