"""Story 7.3 — null-date payload audit + backfill (data readiness).

The temporal sort modes (``recent``/``oldest``) are only as trustworthy as the
payload ``date`` field. Newer archive entries can carry an episode but a null
date — the Mode B apply path threads queue-record dates through unvalidated
(``apply.py``) — so a naive "mais recentes" would sink a brand-new dateless
episode (known instances: #928, #1034) to the bottom.

This audits the undated tail (counted, grouped by numeric episode so the
recent-end concentration is visible) and backfills the true ISO date into the
payloads from the highest-fidelity source first: the metadata cache
(``data/meta/<id>.json``, already ISO-normalized by ``metadata._normalize``),
then the ``video_stats`` row. Each patch is a ``set_payload`` scoped by
``video_id`` — payload-only, no re-embedding, and idempotent: a second run
finds nothing to do. Genuinely unknown dates (no cache, no row) are left as-is.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from qdrant_client import QdrantClient, models

from mc import config
from mc.config import COLLECTION_NAME
from mc.ingest.metadata import episode_int
from mc.state import StateStore

logger = logging.getLogger(__name__)

_SCROLL_LIMIT = 1000


def _load_cache(cache_dir: Path, video_id: str) -> dict:
    path = cache_dir / f"{video_id}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _undated_points(client: QdrantClient, collection: str) -> list:
    """Every indexed point whose payload ``date`` is empty (the undated tail).

    One filtered scroll (``is_empty=date``) paged through; the DATETIME index
    makes this cheap rather than a full collection scan. The fingerprint
    sentinel (point id 0, ``__meta__`` payload) has no date BY DESIGN — it is
    collection metadata, not a transcript point, and can never be repaired,
    so it is excluded or it would pollute the audit forever (a production
    run reported "325 points across 7 videos" that were really 324 + the
    sentinel)."""
    filt = models.Filter(
        must=[models.IsEmptyCondition(is_empty=models.PayloadField(key="date"))],
        must_not=[
            models.FieldCondition(key="__meta__", match=models.MatchValue(value=True))
        ],
    )
    points: list = []
    offset = None
    while True:
        scrolled, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=_SCROLL_LIMIT,
            offset=offset,
            with_payload=True,
        )
        points.extend(scrolled)
        if next_offset is None:
            break
        offset = next_offset
    return points


def _resolve_date(video_id: str, state: StateStore, cache_dir: Path) -> str | None:
    """True ISO date for a video: metadata cache first, then the stats row.

    Both sources already hold an ISO ``YYYY-MM-DD`` (the cache via
    ``metadata._normalize``); a value absent from both means genuinely unknown,
    so the point stays undated rather than being assigned a guess."""
    date = _load_cache(cache_dir, video_id).get("date") or None
    if date is None:
        date = state.video_facts(video_id)["date"]
    return date or None


def run_date_backfill(
    *,
    state: StateStore | None = None,
    cache_dir: Path | None = None,
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
) -> dict:
    """Audit + backfill null-date payloads. Returns a report dict (see CLI).

    A point counts as undated when its payload ``date`` is ``None`` or
    whitespace-only. Per undated video, the date is resolved from the cache
    then the stats row and patched into every one of that video's points via a
    ``video_id``-scoped ``set_payload``. Idempotent and resumable: a re-run over
    a fully-dated index is a no-op.
    """
    if client is None:
        raise ValueError("date backfill needs a Qdrant client to read/patch payloads")
    state = state or StateStore(config.INGEST_DB_PATH)
    cache_dir = Path(cache_dir) if cache_dir is not None else config.META_CACHE_DIR

    undated = _undated_points(client, collection)
    points_by_video: dict[str, int] = {}
    episode_of: dict[str, str | None] = {}
    for point in undated:
        video_id = point.payload.get("video_id")
        if video_id is not None:
            points_by_video[video_id] = points_by_video.get(video_id, 0) + 1
            episode_of.setdefault(video_id, point.payload.get("episode"))

    # Audit breakdown: undated videos grouped by their numeric episode (read
    # from the payload, the point's own truth), so the recent-end concentration
    # (the known failure mode — #928, #1034) is visible in the log.
    episodes_by_count: dict[int | None, int] = {}
    for episode in episode_of.values():
        key = episode_int(episode)
        episodes_by_count[key] = episodes_by_count.get(key, 0) + 1

    dates_backfilled = 0
    patched: set[str] = set()
    for video_id in sorted(points_by_video):
        date = _resolve_date(video_id, state, cache_dir)
        if date is None:
            continue  # genuinely unknown — leave the point undated
        client.set_payload(
            collection_name=collection,
            payload={"date": date},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id", match=models.MatchValue(value=video_id)
                    )
                ]
            ),
        )
        dates_backfilled += 1
        patched.add(video_id)

    undated_after = sum(cnt for vid, cnt in points_by_video.items() if vid not in patched)
    ordered_episodes = dict(
        sorted(episodes_by_count.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
    )
    report = {
        "undated_points_before": len(undated),
        "undated_videos": len(points_by_video),
        "dates_backfilled": dates_backfilled,
        "undated_points_after": undated_after,
        "undated_by_episode": ordered_episodes,
        "summary": (
            f"date-backfill: {len(undated)} points across {len(points_by_video)} "
            f"videos were undated; {dates_backfilled} backfilled; "
            f"{undated_after} points remain unknown"
        ),
    }
    logger.info(report["summary"])
    return report
