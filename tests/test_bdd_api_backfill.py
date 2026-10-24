"""BDD steps for tests/bdd/api_backfill.feature (Story 9.5) — the YouTube
Data API fact pass heals undated payloads end-to-end against the local
qdrant container. Dense vectors are faked (constant e5 of the query word)
so seeding is instant; the sparse BM25 encoder is real, so both seeded
videos match the query word and can be ordered. The api_client is a fake
serving canned facts — no network, ever (the Story 9.2 transport is unit-
tested separately).
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, scenarios, then, when
from qdrant_client import QdrantClient, models

from mc.api.app import create_app
from mc.embeddings import embed_query
from mc.ingest.models import Chunk
from mc.ingest.stats_backfill import run_api_backfill
from mc.qdrant import upsert_video
from mc.state import StateStore

scenarios("bdd/api_backfill.feature")

pytestmark = pytest.mark.integration

QUERY_WORD = "bitcoin"
DATED_VID = "video000020"  # episode #1000, dated 2026-10-20
DATED_EPISODE = "1000"
DATED_DATE = "2026-10-20"
UNDATED_VID = "video000035"  # episode #1035, null payload date
UNDATED_EPISODE = "1035"
API_DATE = "2026-10-22"  # newer than the dated episode
API_DURATION = 5400
MISSING_VID = "video000036"  # the API no longer serves it
SERVED_VID = "video000037"  # undated, but the API does return it
SERVED_DATE = "2026-10-23"


def _by_video(video_id: str):
    return models.Filter(
        must=[models.FieldCondition(key="video_id", match=models.MatchValue(value=video_id))]
    )


def _chunks(video_id: str, *, date, episode: str) -> list[Chunk]:
    return [
        Chunk(
            video_id=video_id,
            text=f"o {QUERY_WORD} reagiu ao boato do dia",
            url=f"https://www.youtube.com/watch?v={video_id}&t=30",
            start_seconds=30,
            date=date,
            episode=episode,
            channel="Morning Crypto",
        )
    ]


@contextmanager
def _fake_dense():
    """Constant dense vectors so seeding skips the e5 model (sparse BM25 ranks)."""
    import mc.qdrant as q

    original = q.embed_passage
    q.embed_passage = lambda texts: [list(embed_query(QUERY_WORD)) for _ in texts]
    try:
        yield
    finally:
        q.embed_passage = original


class _FakeApiClient:
    """The Story 9.3 adapter shape with canned facts — zero network."""

    def __init__(self, facts: dict[str, dict], missing: set[str] | None = None):
        self.facts = facts
        self.missing = set(missing or set())
        self.quota_units = 3  # discovery spent a few units finding nothing new

    def channel_uploads(self, channels):
        return {c: f"UU{i}" for i, c in enumerate(channels)}

    def playlist_videos(self, playlist_id):
        return []  # everything relevant is already in the indexed universe

    def video_facts(self, ids):
        served = {i: dict(self.facts[i]) for i in ids if i in self.facts}
        missing = [i for i in ids if i in self.missing or i not in self.facts]
        return served, missing


@pytest.fixture
def ctx(tmp_path):
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=30)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    collection = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    state = StateStore(tmp_path / "ingest.db")
    ctx = {
        "client": client,
        "collection": collection,
        "state": state,
        "cache_dir": tmp_path / "meta",
        "app": create_app(client=client, collection=collection, state=state),
    }
    yield ctx
    client.delete_collection(collection)


@given("the archive contains an episode dated \"2026-10-20\"")
def seeded_dated(ctx):
    with _fake_dense():
        upsert_video(
            ctx["client"], ctx["collection"], DATED_VID,
            _chunks(DATED_VID, date=DATED_DATE, episode=DATED_EPISODE),
        )
    ctx["state"].mark_processed(DATED_VID, "update")


@given("an undated episode #1035 whose payload date is null")
def seeded_undated(ctx):
    with _fake_dense():
        # date=None → the payload is indexed without a date (the failure mode)
        upsert_video(
            ctx["client"], ctx["collection"], UNDATED_VID,
            _chunks(UNDATED_VID, date=None, episode=UNDATED_EPISODE),
        )
    ctx["state"].mark_processed(UNDATED_VID, "update")


@given(
    "the YouTube API reports upload date \"2026-10-22\" and duration 5400 "
    "for episode #1035's video"
)
def api_reports_facts(ctx):
    ctx["api_client"] = _FakeApiClient(
        {
            UNDATED_VID: {
                "upload_date": "20261022",
                "title": f"[ Morning Crypto ][ ep {UNDATED_EPISODE} ] x",
                "duration": API_DURATION,
            }
        }
    )


@when("the API backfill runs")
def api_backfill_runs(ctx):
    ctx["report"] = run_api_backfill(
        state=ctx["state"],
        cache_dir=ctx["cache_dir"],
        client=ctx["client"],
        collection=ctx["collection"],
        api_client=ctx["api_client"],
    )


@then("the metadata cache for episode #1035's video carries the date \"2026-10-22\"")
def cache_carries_date(ctx):
    stored = json.loads((ctx["cache_dir"] / f"{UNDATED_VID}.json").read_text())
    assert stored["date"] == API_DATE


@then("every indexed point for episode #1035 carries the date \"2026-10-22\"")
def points_carried_date(ctx):
    points, _ = ctx["client"].scroll(
        collection_name=ctx["collection"],
        scroll_filter=_by_video(UNDATED_VID),
        limit=100,
        with_payload=True,
    )
    assert points, "episode #1035 has no indexed points"
    assert all(p.payload.get("date") == API_DATE for p in points)


@then('a "recent" search ranks episode #1035 above the "2026-10-20" episode')
def recent_orders_newest_first(ctx):
    body = TestClient(ctx["app"]).get(
        "/search", params={"query": QUERY_WORD, "sort": "recent", "limit": 50}
    ).json()
    episodes = [r["episode"] for r in body["results"]]
    assert UNDATED_EPISODE in episodes and DATED_EPISODE in episodes
    assert episodes.index(UNDATED_EPISODE) < episodes.index(DATED_EPISODE)
    # and the healed date is now the newest
    assert body["results"][0]["date"] == API_DATE


@given("the archive contains an undated video the API does not return")
def seeded_missing_video(ctx):
    with _fake_dense():
        upsert_video(
            ctx["client"], ctx["collection"], MISSING_VID,
            _chunks(MISSING_VID, date=None, episode="1036"),
        )
    ctx["state"].mark_processed(MISSING_VID, "update")
    ctx["api_client"] = _FakeApiClient({}, missing={MISSING_VID})


@given("another undated video the API does return")
def seeded_served_video(ctx):
    with _fake_dense():
        upsert_video(
            ctx["client"], ctx["collection"], SERVED_VID,
            _chunks(SERVED_VID, date=None, episode="1037"),
        )
    ctx["state"].mark_processed(SERVED_VID, "update")
    ctx["api_client"].facts[SERVED_VID] = {
        "upload_date": "20261023",
        "title": "[ Morning Crypto ][ ep 1037 ] x",
        "duration": 3600,
    }


@then("the run completes with the unserved video counted as still missing")
def unserved_counted(ctx):
    assert ctx["report"]["still_missing"] == 1
    # deleted/private ids never poison the run: it completed with a summary
    assert "1 still missing" in ctx["report"]["summary"]


@then("every other undated video is dated")
def served_video_dated(ctx):
    points, _ = ctx["client"].scroll(
        collection_name=ctx["collection"],
        scroll_filter=_by_video(SERVED_VID),
        limit=100,
        with_payload=True,
    )
    assert points, "the served video has no indexed points"
    assert all(p.payload.get("date") == SERVED_DATE for p in points)
