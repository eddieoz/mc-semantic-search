"""Story 7.3 — null-date payload audit + backfill (stub Qdrant, no container).

The temporal sort modes are only as trustworthy as the payload ``date`` field.
Newer episodes can carry an episode but a null date (the Mode B apply path
threads queue-record dates through unvalidated), so this finds the undated
tail, counts it (grouped by episode), and patches the true ISO date in from the
metadata cache / video_stats row — payload-only, idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

from mc.ingest.date_backfill import run_date_backfill
from mc.state import StateStore


class _Point:
    def __init__(self, payload: dict):
        self.payload = payload


class FakeQdrant:
    """Minimal stand-in: ``scroll`` honors the ``is_empty=date`` filter, and
    ``set_payload`` records the patch and applies it in-memory so a re-run is a
    genuine no-op (idempotency is observed, not asserted by count alone)."""

    def __init__(self, points: list[_Point]):
        self._points = list(points)
        self.patches: list[dict] = []

    @staticmethod
    def _empty_date(filter_) -> bool:
        must = getattr(filter_, "must", None)
        if not must:
            return False
        cond = must[0]
        # IsEmptyCondition.is_empty is the PayloadField itself (has `.key`).
        return getattr(getattr(cond, "is_empty", None), "key", None) == "date"

    @staticmethod
    def _excludes_meta(filter_) -> bool:
        must_not = getattr(filter_, "must_not", None) or []
        return any(getattr(cond, "key", None) == "__meta__" for cond in must_not)

    def scroll(
        self, *, collection_name, scroll_filter, limit=1000, offset=None, with_payload=False
    ):
        # Honor the is_empty=date filter per-point: a point is in the pool only
        # when its OWN date is empty (not merely because the filter asks for it).
        if self._empty_date(scroll_filter):
            pool = [p for p in self._points if not str(p.payload.get("date") or "").strip()]
        else:
            pool = list(self._points)
        if self._excludes_meta(scroll_filter):
            pool = [p for p in pool if not p.payload.get("__meta__")]
        start = offset or 0
        page = pool[start : start + limit]
        next_offset = None if start + limit >= len(pool) else start + limit
        return page, next_offset

    def set_payload(self, *, collection_name, payload, points):
        self.patches.append({"payload": payload, "points": points})
        vids = {
            cond.match.value
            for cond in getattr(points, "must", [])
            if getattr(getattr(cond, "match", None), "value", None) is not None
        }
        for p in self._points:
            if p.payload.get("video_id") in vids:
                p.payload["date"] = payload.get("date")


def _point(video_id: str, date, episode: str):
    return _Point({"video_id": video_id, "date": date, "episode": episode, "text": "x"})


def _write_cache(cache_dir: Path, video_id: str, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{video_id}.json").write_text(json.dumps(meta), encoding="utf-8")


def _ctx(tmp_path: Path):
    state = StateStore(tmp_path / "ingest.db")
    return state, tmp_path / "meta"


def test_audit_counts_null_date_points(tmp_path):
    """Undated = None / "" / whitespace; counted as points and as videos."""
    points = [
        _point("v1", "2026-10-20", "1000"),  # dated — ignored
        _point("v2", None, "1034"),  # undated
        _point("v2", "", "1034"),  # undated (blank) — 2nd chunk of v2
        _point("v3", "   ", "928"),  # undated (whitespace)
    ]
    client = FakeQdrant(points)
    state, cache_dir = _ctx(tmp_path)

    report = run_date_backfill(state=state, cache_dir=cache_dir, client=client)

    assert report["undated_points_before"] == 3  # v2 x2 + v3 x1
    assert report["undated_videos"] == 2  # v2, v3
    assert report["dates_backfilled"] == 0  # no sources → nothing patched
    assert report["undated_points_after"] == 3  # all remain unknown
    assert set(report["undated_by_episode"]) == {928, 1034}  # grouped by numeric episode


def test_backfill_patches_from_cache_with_right_filter_and_payload(tmp_path):
    points = [_point("v1", "2026-10-20", "1000"), _point("v2", None, "1034")]
    client = FakeQdrant(points)
    state, cache_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v2", {"episode": "1034", "date": "2026-10-22"})

    report = run_date_backfill(state=state, cache_dir=cache_dir, client=client)

    assert report["dates_backfilled"] == 1
    assert report["undated_points_after"] == 0
    # exactly one patch, scoped by video_id, carrying the ISO date
    assert len(client.patches) == 1
    payload, filter_ = client.patches[0]["payload"], client.patches[0]["points"]
    assert payload == {"date": "2026-10-22"}
    assert filter_.must[0].key == "video_id"
    assert filter_.must[0].match.value == "v2"
    # the in-memory point now carries the date (so a re-run finds nothing)
    assert points[1].payload["date"] == "2026-10-22"


def test_backfill_uses_row_when_cache_missing(tmp_path):
    """The truth chain falls through to the video_stats row when the cache
    has no date for the video."""
    points = [_point("v9", None, "500")]
    client = FakeQdrant(points)
    state, cache_dir = _ctx(tmp_path)
    state.record_video_stats("v9", episode=500, date="2025-01-15")

    report = run_date_backfill(state=state, cache_dir=cache_dir, client=client)

    assert report["dates_backfilled"] == 1
    assert client.patches[0]["payload"] == {"date": "2025-01-15"}


def test_second_run_is_a_noop(tmp_path):
    points = [_point("v2", None, "1034")]
    client = FakeQdrant(points)
    state, cache_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v2", {"episode": "1034", "date": "2026-10-22"})

    first = run_date_backfill(state=state, cache_dir=cache_dir, client=client)
    second = run_date_backfill(state=state, cache_dir=cache_dir, client=client)

    assert first["dates_backfilled"] == 1
    assert second["undated_points_before"] == 0  # nothing left undated
    assert second["dates_backfilled"] == 0  # no new patches
    assert len(client.patches) == 1  # patched exactly once across both runs


def test_missing_sources_leave_point_untouched(tmp_path):
    points = [_point("v7", None, "777")]
    client = FakeQdrant(points)
    state, cache_dir = _ctx(tmp_path)  # no cache, no row → genuinely unknown

    report = run_date_backfill(state=state, cache_dir=cache_dir, client=client)

    assert report["dates_backfilled"] == 0
    assert report["undated_points_after"] == 1
    assert client.patches == []  # never issued a patch for an unknown date
    assert points[0].payload["date"] is None


def test_sentinel_meta_point_is_not_counted_as_undated(tmp_path):
    """The collection fingerprint sentinel (point id 0, ``__meta__`` payload)
    carries no date BY DESIGN — it is collection metadata, not a transcript
    point, and can never be repaired. It must never pollute the undated
    audit (observed in production: "325 points across 7 videos" was really
    324 transcript points + the sentinel)."""
    points = [
        _Point({"__meta__": True, "fingerprint": {"dense": {"dim": 384}}}),
        _point("v1", None, "1034"),
    ]
    client = FakeQdrant(points)
    state, cache_dir = _ctx(tmp_path)

    report = run_date_backfill(state=state, cache_dir=cache_dir, client=client)

    assert report["undated_points_before"] == 1
    assert report["undated_points_after"] == 1
    assert report["undated_videos"] == 1
