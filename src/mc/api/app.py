"""FastAPI application factory.

Story 0.2 added the liveness route; Story 4.5 brings GET /search and
Story 4.6 the freshness-aware /health. Everything is dependency-injected
(qdrant client, state store, freshness threshold) so tests can build the
app against local qdrant, unreachable clients or tmp state stores.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from qdrant_client import QdrantClient

from mc.api import agent as agent_docs
from mc.api.health import health_snapshot
from mc.config import (
    COLLECTION_NAME,
    FRESHNESS_MAX_HOURS,
    INGEST_DB_PATH,
    PUBLIC_URL,
    QDRANT_URL,
)
from mc.qdrant import client_kwargs
from mc.search import engine as search_engine
from mc.search.cache import ResultCache
from mc.search.engine import SearchResult
from mc.search.engine import search as run_search
from mc.search.query import parse_query
from mc.search.ranking import apply_boost
from mc.search.snippets import build_snippet
from mc.state import StateStore

# Story 4.5: one request fetches at most 50 hits — bigger client limits are
# clamped (never rejected) so no caller can demand the whole archive.
MAX_SEARCH_LIMIT = 50

# DoS guard: query length is capped at the API edge (and clamped again in
# the engine) so oversized inputs can't occupy the embedder's global lock.
MAX_QUERY_LENGTH = 512

# Stored-XSS defense (poisoned-index payloads): YouTube ids are strictly
# alphanumeric+dash/underscore; anything else is never rendered/linked.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,20}$")

# Keys every search result exposes — /search and the render=json envelope
# share one contract so agents never meet two shapes.
_SEARCH_KEYS = (
    "title",
    "url",
    "start_seconds",
    "date",
    "episode",
    "channel",
    "snippet",
    "score",
)

# DoS guard for the render switch — only the literal value "json" means JSON.
MAX_RENDER_LENGTH = 16

TEMPLATES = Jinja2Templates(
    directory=Path(__file__).parent / "templates", autoescape=True
)


def _tape(start_seconds: int) -> str:
    """Tape-counter rendering of the chunk offset — the design anchor."""
    total = int(start_seconds or 0)
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"T+{hours:02d}:{minutes:02d}:{seconds:02d}"


def _fmt_int_pt(value: int) -> str:
    """pt-BR integer: thousands separator ``.`` → ``197.342``, ``3.556``."""
    return f"{int(value):,}".replace(",", ".")


def _fmt_dec_pt(value: float) -> str:
    """pt-BR decimal: 2 places, comma decimal, ``.`` thousands → ``54,82``,
    ``3.289,03``."""
    us = f"{float(value):,.2f}"  # "54.82" / "3,289.03"
    return us.replace(".", "\x00").replace(",", ".").replace("\x00", ",")


def _duration_line(seconds: int | None) -> str | None:
    """The total duration across s/min/h/days, or None when unknown (Story 6.5)."""
    if seconds is None:
        return None
    parts = [
        f"{_fmt_int_pt(seconds)} segundos",
        f"{_fmt_dec_pt(seconds / 60)} minutos",
        f"{_fmt_dec_pt(seconds / 3600)} horas",
        f"{_fmt_dec_pt(seconds / 86400)} dias",
    ]
    return " · ".join(parts)


def _format_archive_status(raw: dict) -> dict:
    """Build the template ``archive_status`` dict from raw archive numbers.

    Each stat is omitted when its input is unknown (never rendered as 0/—);
    the dict is empty when nothing is known, so the template hides the whole
    section. One source of truth: the same ``archive_stats()`` the JSON
    envelope reads (Story 6.6).
    """
    stats: dict = {}
    episode, date = raw.get("last_episode"), raw.get("last_episode_date")
    if episode is not None:
        # Render the episode even when its date is unknown, so the HTML fact
        # agrees with the JSON envelope (which reports last_episode with a
        # null date) instead of hiding something the archive knows.
        stats["last_episode"] = f"#{episode} · {date}" if date else f"#{episode}"
    if raw.get("videos"):
        stats["videos"] = _fmt_int_pt(raw["videos"])
    duration = _duration_line(raw.get("seconds"))
    if duration is not None:
        stats["duration"] = duration
    return stats


def _raw_archive_status(raw: dict) -> dict:
    """Machine-readable archive numbers for the ``render=json`` envelope
    (Story 6.6): raw ints/floats, derived stats rounded to 2 decimals, and
    ``None`` (JSON ``null``) for anything unknown. One source of truth with
    the HTML surface — the same ``archive_stats()`` dict.
    """
    seconds = raw.get("seconds")
    return {
        "videos": raw.get("videos"),
        "seconds": seconds,
        "minutes": round(seconds / 60, 2) if seconds is not None else None,
        "hours": round(seconds / 3600, 2) if seconds is not None else None,
        "days": round(seconds / 86400, 2) if seconds is not None else None,
        "last_episode": raw.get("last_episode"),
        "last_episode_date": raw.get("last_episode_date"),
    }


def _freshness_line(last_update: str | None) -> str:
    """Portuguese archive-freshness line shown in the footer/empty state."""
    if not last_update:
        return "acervo atualizado agora mesmo."
    date = last_update[:10]
    return f"acervo atualizado até {date}"


def _safe_url(url: str | None, video_id: str | None) -> str:
    """Only https:// YouTube URLs survive a poisoned-index payload.

    Restricting the host to YouTube means a poisoned collection (or a Mode B
    queue record) can't make results link to an arbitrary https:// site —
    phishing, not just stored XSS.
    """
    if url and url.startswith("https://"):
        host = urlparse(url).netloc.lower()
        if host in ("www.youtube.com", "youtube.com", "youtu.be"):
            return url
    if video_id and _VIDEO_ID_RE.fullmatch(video_id):
        return f"https://www.youtube.com/watch?v={video_id}"
    return ""


def _search_results(results: list[dict]) -> list[dict]:
    return [{key: r[key] for key in _SEARCH_KEYS} for r in results]


def _video_result(result: SearchResult, query: str) -> dict:
    video_id = (
        result.video_id
        if result.video_id and _VIDEO_ID_RE.fullmatch(result.video_id)
        else None
    )
    try:
        start_seconds = max(int(result.start_seconds or 0), 0)
    except (TypeError, ValueError):
        # Poisoned payload with a non-numeric start must not 500 the page.
        start_seconds = 0
    return {
        "title": result.episode or f"vídeo {video_id}",
        "episode": result.episode,
        "date": result.date,
        "channel": result.channel,
        "snippet": build_snippet(result.text or "", query).highlighted,
        "url": _safe_url(result.url, result.video_id),
        "video_id": video_id,
        "start_seconds": start_seconds,
        "tape": _tape(start_seconds),
        "score": result.score,
    }



def create_app(
    *,
    qdrant_url: str = QDRANT_URL,
    collection: str = COLLECTION_NAME,
    state_path: Path = INGEST_DB_PATH,
    freshness_max_hours: int = FRESHNESS_MAX_HOURS,
    client: QdrantClient | None = None,
    state: StateStore | None = None,
    cache: object | None = None,
) -> FastAPI:
    qdrant = (
        client if client is not None else QdrantClient(url=qdrant_url, **client_kwargs())
    )
    # StateStore holds a thread-bound sqlite connection: keep only the path and
    # open a short-lived store per request (same pattern as ResultCache).
    _state_db = Path(state.path) if state is not None else Path(state_path)

    @contextmanager
    def _state():
        store = StateStore(_state_db)
        try:
            yield store
        finally:
            store.close()

    app = FastAPI(title="mc-semantic-search", version="0.1.0")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        # 'unsafe-inline' in style/script is required: htmx/Alpine inject
        # inline attributes; frame-src pins the embedded player host.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; frame-src https://www.youtube-nocookie.com; "
            "img-src 'self' https://i.ytimg.com; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )
    # Story 4.7 wiring (review fix): every app gets the embedder-level LRU by
    # default — the engine consults it per request, so a crawler can't starve
    # the embedder (NFR-1). Tests inject their own cache to spy on it.
    cache = cache if cache is not None else ResultCache(
        lambda query: search_engine.embed_query(query), _state_db
    )

    def _freshness() -> dict:
        try:
            with _state() as store:
                return {"last_update": store.last_success("update")}
        except Exception:  # unreadable state must never break search
            return {"last_update": None}

    def _archive_stats() -> dict:
        # Same failure isolation as _freshness: an empty/missing/unreadable
        # video_stats table hides the section and leaves the page intact.
        try:
            with _state() as store:
                return _format_archive_status(store.archive_stats())
        except Exception:  # noqa: BLE001 — stats must never break the page
            return {}

    def _raw_archive_stats() -> dict:
        # Raw numbers for the render=json envelope; same isolation as above
        # so a broken stats store yields nulls, never a 500 (Story 6.6).
        try:
            with _state() as store:
                return _raw_archive_status(store.archive_stats())
        except Exception:  # noqa: BLE001 — stats must never break the API
            return _raw_archive_status({})

    def _video_results(query: str, effective_limit: int) -> list[dict]:
        try:
            results: list[SearchResult] = run_search(
                qdrant,
                query,
                limit=effective_limit,
                collection=collection,
                cache=cache,
            )
        except Exception as exc:  # transport failure → 503, never 500
            raise HTTPException(
                status_code=503, detail="qdrant unreachable"
            ) from exc
        results = apply_boost(results, parse_query(query))
        return [_video_result(r, query) for r in results]

    @app.get("/search")
    def search_endpoint(
        query: str = Query(min_length=1, max_length=MAX_QUERY_LENGTH),
        limit: int = Query(default=10, ge=1),
    ) -> dict:
        effective_limit = min(limit, MAX_SEARCH_LIMIT)
        started = time.perf_counter()
        results = _video_results(query, effective_limit)
        took_ms = (time.perf_counter() - started) * 1000
        return {
            "query": query,
            "took_ms": took_ms,
            "freshness": _freshness(),
            "results": _search_results(results),
        }

    @app.get("/results", response_class=HTMLResponse)
    def results_partial(
        query: str = Query(default="", max_length=MAX_QUERY_LENGTH)
    ) -> str:
        results = _video_results(query, MAX_SEARCH_LIMIT) if query.strip() else []
        return TEMPLATES.get_template("results.html").render(
            results=results,
            query=query,
            freshness_line=_freshness_line(_freshness()["last_update"]),
        )

    @app.get("/", response_class=HTMLResponse)
    def search_page(
        request: Request,
        query: str = Query(default="", max_length=MAX_QUERY_LENGTH),
        render: str = Query(default="", max_length=MAX_RENDER_LENGTH),
    ):
        stripped = query.strip()
        started = time.perf_counter()
        results = _video_results(query, MAX_SEARCH_LIMIT) if stripped else []
        if render.strip().lower() == "json":
            # The agent door (moltbook pattern): the landing page answers in
            # machine-readable form. No query → Muriel's self-description;
            # a query → the /search envelope plus the agent block, so every
            # JSON response is self-describing.
            took_ms = (time.perf_counter() - started) * 1000
            return JSONResponse(
                {
                    "agent": agent_docs.manual(PUBLIC_URL),
                    "query": stripped,
                    "took_ms": took_ms,
                    "freshness": _freshness(),
                    "archive_status": _raw_archive_stats(),
                    "results": _search_results(results),
                }
            )
        return TEMPLATES.TemplateResponse(
            request=request,
            name="search.html",
            context={
                "query": stripped,
                "results": results,
                "freshness_line": _freshness_line(_freshness()["last_update"]),
                "archive_status": _archive_stats(),
                "agent_box": agent_docs.landing_box(),
            },
        )

    @app.get("/llms.txt", response_class=PlainTextResponse)
    def llms_txt_endpoint() -> PlainTextResponse:
        return PlainTextResponse(agent_docs.llms_txt(PUBLIC_URL))

    @app.get("/health")
    def health_endpoint() -> JSONResponse:
        with _state() as store:
            code, body = health_snapshot(
                qdrant, store, freshness_max_hours, collection=collection
            )
        return JSONResponse(status_code=code, content=body)

    return app


app = create_app()
