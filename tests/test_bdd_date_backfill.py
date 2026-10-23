"""BDD steps for tests/bdd/date_backfill.feature (Story 7.3) — null-date
payload audit + backfill against the local qdrant container. Dense vectors are
faked (constant e5 of the query word) so seeding is instant; the sparse BM25
encoder is real, so both seeded videos match the query word and can be ordered.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, scenarios, then, when
from qdrant_client import QdrantClient, models

from mc.api.app import create_app
from mc.embeddings import embed_query
from mc.ingest.date_backfill import run_date_backfill
from mc.ingest.models import Chunk
from mc.qdrant import upsert_video
from mc.state import StateStore

scenarios("bdd/date_backfill.feature")

pytestmark = pytest.mark.integration

QUERY_WORD = "bitcoin"
DATED_VID = "video000020"  # episode #1000, dated 2026-10-20
DATED_EPISODE = "1000"
DATED_DATE = "2026-10-20"
UNDATED_VID = "video000034"  # episode #1034, null payload date
UNDATED_EPISODE = "1034"
CACHE_DATE = "2026-10-22"  # newer than the dated episode


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


@given("an undated episode #1034 whose payload date is null")
def seeded_undated(ctx):
    with _fake_dense():
        # date=None → the payload is indexed without a date (the failure mode)
        upsert_video(
            ctx["client"], ctx["collection"], UNDATED_VID,
            _chunks(UNDATED_VID, date=None, episode=UNDATED_EPISODE),
        )


@given('the metadata cache reports the date "2026-10-22" for episode #1034')
def cache_reports_date(ctx):
    ctx["cache_dir"].mkdir(parents=True, exist_ok=True)
    (ctx["cache_dir"] / f"{UNDATED_VID}.json").write_text(
        '{"episode": "1034", "date": "2026-10-22"}', encoding="utf-8"
    )


@when("the dates backfill runs")
def dates_backfill_runs(ctx):
    ctx["backfill"] = run_date_backfill(
        state=ctx["state"],
        cache_dir=ctx["cache_dir"],
        client=ctx["client"],
        collection=ctx["collection"],
    )


@then('every indexed point for episode #1034 carries the date "2026-10-22"')
def points_carried_date(ctx):
    points, _ = ctx["client"].scroll(
        collection_name=ctx["collection"],
        scroll_filter=_by_video(UNDATED_VID),
        limit=100,
        with_payload=True,
    )
    assert points, "episode #1034 has no indexed points"
    assert all(p.payload.get("date") == CACHE_DATE for p in points)
    assert ctx["backfill"]["dates_backfilled"] == 1
    assert ctx["backfill"]["undated_points_after"] == 0


@then('a "recent" search ranks episode #1034 above the "2026-10-20" episode')
def recent_orders_newest_first(ctx):
    body = TestClient(ctx["app"]).get(
        "/search", params={"query": QUERY_WORD, "sort": "recent", "limit": 50}
    ).json()
    episodes = [r["episode"] for r in body["results"]]
    assert UNDATED_EPISODE in episodes and DATED_EPISODE in episodes
    assert episodes.index(UNDATED_EPISODE) < episodes.index(DATED_EPISODE)
    # and the healed date is now the newest
    assert body["results"][0]["date"] == CACHE_DATE
