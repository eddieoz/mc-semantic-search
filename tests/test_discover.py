"""Story 3.1 — channel video listing via yt-dlp --flat-playlist.

Discovery-tabs design (2026-10-22): every configured channel is listed
across its /videos, /streams and /shorts tabs — Morning Crypto episodes are
live streams and only appear under /streams. Video ids are exactly 11 chars;
anything else (playlist-type entries from degraded listings, traversal
shapes) is skipped with a warning.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mc.ingest.discover import DiscoveryError, list_channel_videos

FIXTURE = Path(__file__).parent / "fixtures" / "flat_playlist.json"

CHANNEL_URL = "https://www.youtube.com/channel/UCexample12345"


@pytest.fixture
def playlist_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _fake_run(returncode=0, stdout="", stderr=""):
    def run(cmd, **kwargs):
        import types

        proc = types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
        return proc

    return run


def test_lists_video_ids_from_flat_playlist(playlist_fixture, monkeypatch):
    stdout = json.dumps(playlist_fixture)
    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", _fake_run(0, stdout, ""))
    ids = list_channel_videos(["https://www.youtube.com/@canal"])
    assert ids == {"vid1vid1vid", "vid2vid2vid", "vid3vid3vid"}


def test_multiple_channels_union(playlist_fixture, monkeypatch):
    stdout = json.dumps(playlist_fixture)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return _fake_run(0, stdout, "")(cmd, **kwargs)

    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", run)
    ids = list_channel_videos(["https://www.youtube.com/@a", "https://youtube.com/@b"])
    assert len(ids) == 3
    # 2 channels × 3 tabs (videos, streams, shorts)
    assert len(calls) == 6


def test_tabs_appended_per_channel(monkeypatch):
    """Episodes can be live streams and shorts: every configured channel is
    listed across its /videos, /streams and /shorts tabs (Morning Crypto
    #1037 only appears under /streams)."""
    calls = []

    def run(listing_url, **kwargs):
        calls.append(listing_url)
        return _fake_run(0, json.dumps({"entries": []}), "")(listing_url, **kwargs)

    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", run)
    list_channel_videos([CHANNEL_URL])
    assert [u.split("/channel/")[1] for u in calls] == [
        "UCexample12345/videos",
        "UCexample12345/streams",
        "UCexample12345/shorts",
    ]


def test_already_tabbed_url_listed_once(monkeypatch):
    """A config URL that already points at a tab is listed as-is — never
    double-appended (/streams/videos)."""
    calls = []

    def run(listing_url, **kwargs):
        calls.append(listing_url)
        return _fake_run(0, json.dumps({"entries": []}), "")(listing_url, **kwargs)

    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", run)
    list_channel_videos([f"{CHANNEL_URL}/streams"])
    assert calls == [f"{CHANNEL_URL}/streams"]


def test_tabs_union_dedup(monkeypatch):
    """The same video surfaced by two tabs must appear once in the union."""
    by_url = {
        f"{CHANNEL_URL}/videos": {"entries": [{"id": "aaaabbbbbcc"}, {"id": "cccdddddeee"}]},
        f"{CHANNEL_URL}/streams": {"entries": [{"id": "aaaabbbbbcc"}]},
        f"{CHANNEL_URL}/shorts": {"entries": [{"id": "fffeggggghi"}]},
    }

    def run(listing_url, **kwargs):
        return _fake_run(0, json.dumps(by_url[listing_url]), "")(listing_url, **kwargs)

    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", run)
    ids = list_channel_videos([CHANNEL_URL])
    assert ids == {"aaaabbbbbcc", "cccdddddeee", "fffeggggghi"}


def test_channels_come_from_config_not_hardcoded(monkeypatch, tmp_path):
    """Channels must be configurable (env/config.yaml), never baked in code."""
    import mc.config as config

    monkeypatch.setenv("MC_CHANNELS", "https://www.youtube.com/@um, https://x/@dois")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    from mc.ingest.discover import load_channels

    assert load_channels() == (
        "https://www.youtube.com/@um",
        "https://x/@dois",
    )


def test_malformed_output_clear_error(monkeypatch):
    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", _fake_run(0, "isto não é json {", ""))
    with pytest.raises(DiscoveryError, match="flat-playlist"):
        list_channel_videos(["https://www.youtube.com/@x"])


def test_malformed_entries_clear_error(monkeypatch):
    stdout = json.dumps({"entries": ["not-a-dict", None]})
    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", _fake_run(0, stdout, ""))
    with pytest.raises(DiscoveryError):
        list_channel_videos(["https://www.youtube.com/@x"])


def test_ytdlp_failure_raises(monkeypatch):
    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", _fake_run(1, "", "boom"))
    with pytest.raises(DiscoveryError, match="boom"):
        list_channel_videos(["https://www.youtube.com/@x"])


def test_missing_tab_tolerated(monkeypatch):
    """Not every channel has every tab (channel UC0JADez… has no /streams):
    a 'does not have a X tab' listing must be skipped with a warning, never
    abort the whole run — the other tabs still feed discovery."""
    by_url = {
        f"{CHANNEL_URL}/videos": {"entries": [{"id": "aaaabbbbbcc"}]},
        f"{CHANNEL_URL}/streams": {
            "returncode": 1,
            "stderr": "ERROR: [youtube:tab] UCexample12: This channel does not have a streams tab",
        },
        f"{CHANNEL_URL}/shorts": {"entries": [{"id": "fffeggggghi"}]},
    }

    def run(listing_url, **kwargs):
        spec = by_url[listing_url]
        if "returncode" in spec:
            return _fake_run(spec["returncode"], "", spec["stderr"])(listing_url)
        return _fake_run(0, json.dumps(spec), "")(listing_url, **kwargs)

    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", run)
    ids = list_channel_videos([CHANNEL_URL])
    assert ids == {"aaaabbbbbcc", "fffeggggghi"}


def test_other_tab_failures_still_raise(monkeypatch):
    """Only 'no such tab' is tolerable — bot-gates, network errors and any
    other listing failure must still fail loudly."""

    def run(listing_url, **kwargs):
        if listing_url.endswith("/streams"):
            return _fake_run(1, "", "Sign in to confirm you're not a bot")(listing_url)
        return _fake_run(0, json.dumps({"entries": []}), "")(listing_url, **kwargs)

    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", run)
    with pytest.raises(DiscoveryError, match="not a bot"):
        list_channel_videos([CHANNEL_URL])


def test_invalid_video_ids_skipped_with_warning(monkeypatch, caplog):
    """Video ids become path components (f"{id}.srt") — traversal-shaped ids
    AND non-video ids (a degraded listing can surface playlist-type entries
    whose id is the 24-char channel id) must be skipped and logged, never
    propagated into the pipeline. Real video ids are exactly 11 chars."""
    import logging

    stdout = json.dumps(
        {
            "entries": [
                {"id": "../evil"},
                {"id": "a/b"},
                {"id": "has space"},
                {"id": "a" * 24},  # a channel id from a playlist-type entry
                {"id": "a" * 10},  # too short
                {"id": "b" * 12},  # too long
                {"id": "still_vali1"},  # exactly 11 chars — valid
            ]
        }
    )
    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", _fake_run(0, stdout, ""))
    with caplog.at_level(logging.WARNING, logger="mc.ingest.discover"):
        ids = list_channel_videos(["https://www.youtube.com/@x"])

    assert ids == {"still_vali1"}
    # the fake returns the same listing for all three tabs → 6 invalid × 3
    assert sum("invalid video id" in r.message for r in caplog.records) == 18


def test_cookies_flag_passed_when_file_exists(monkeypatch, tmp_path):
    """Listings must carry the cookies: without them YouTube serves degraded
    layouts (playlist-type junk entries instead of videos)."""
    import mc.config as config

    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# netscape cookie jar\n")
    monkeypatch.setattr(config, "COOKIES_PATH", str(cookies))
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _fake_run(0, json.dumps({"entries": []}), "")(cmd, **kwargs)

    monkeypatch.setattr("mc.ingest.discover.subprocess.run", run)
    list_channel_videos([CHANNEL_URL])
    assert "--cookies" in seen["cmd"]
    assert str(cookies) in seen["cmd"]


def test_no_cookies_flag_when_file_missing(monkeypatch, tmp_path):
    import mc.config as config

    monkeypatch.setattr(config, "COOKIES_PATH", str(tmp_path / "absent.txt"))
    seen = {}

    def run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _fake_run(0, json.dumps({"entries": []}), "")(cmd, **kwargs)

    monkeypatch.setattr("mc.ingest.discover.subprocess.run", run)
    list_channel_videos([CHANNEL_URL])
    assert "--cookies" not in seen["cmd"]


def test_ytdlp_timeout_raises_discovery_error(monkeypatch):
    def run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=120)

    monkeypatch.setattr("mc.ingest.discover.subprocess.run", run)
    with pytest.raises(DiscoveryError, match="timed out"):
        list_channel_videos(["https://www.youtube.com/@x"])
