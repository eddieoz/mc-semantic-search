"""Agent-facing surface (unit, no qdrant): render=json envelope, the
no-query manual, /llms.txt, and the instructions section on the landing
page. run_search is faked like test_api_search.py does."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from mc.api import app as app_module
from mc.api.app import create_app
from mc.search.engine import SearchResult
from mc.state import StateStore

RESULT_KEYS = {
    "title",
    "url",
    "start_seconds",
    "date",
    "episode",
    "channel",
    "snippet",
    "score",
}


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(
        app_module,
        "run_search",
        lambda *args, **kwargs: [
            SearchResult(
                video_id="video000001",
                text="o bitcoin reagiu ao boato do dia",
                url="https://www.youtube.com/watch?v=video000001&t=30",
                start_seconds=30,
                date="2026-01-01",
                episode="1035",
                channel="mc",
                score=1.0,
            )
        ],
    )
    state = StateStore(tmp_path / "ingest.db")
    return TestClient(create_app(client=object(), state=state))


def test_render_json_with_query_matches_search_contract(client):
    response = client.get("/", params={"query": "bitcoin", "render": "json"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert set(body) == {
        "agent",
        "query",
        "took_ms",
        "freshness",
        "archive_status",
        "sort",
        "results",
    }
    assert body["query"] == "bitcoin"
    assert body["sort"] == "relevance"  # effective mode echoed, always present
    assert isinstance(body["took_ms"], int | float)
    assert "last_update" in body["freshness"]
    assert len(body["results"]) == 1
    assert set(body["results"][0]) == RESULT_KEYS
    assert body["results"][0]["url"].startswith("https://www.youtube.com/watch?v=")


def test_render_json_envelope_is_self_describing(client):
    """Every JSON answer carries Muriel's card so the agent can chain calls."""
    agent = client.get("/", params={"query": "bitcoin", "render": "json"}).json()[
        "agent"
    ]
    assert agent["name"] == "Muriel"
    assert any("/?query=SEU_TERMO&render=json" in line for line in agent["how_to_search"])
    assert any("/search?query=SEU_TERMO&limit=10" in line for line in agent["how_to_search"])
    assert set(agent["result_fields"]) == RESULT_KEYS


def test_render_json_without_query_returns_manual(client):
    response = client.get("/", params={"render": "json"})
    body = response.json()
    assert body["query"] == ""
    assert body["results"] == []
    assert body["sort"] == "relevance"  # manual branch still echoes the effective mode
    assert body["agent"]["role"] == "bibliotecária do acervo Morning Crypto"


def test_render_is_case_insensitive_and_defaults_to_html(client):
    json_body = client.get("/", params={"query": "bitcoin", "render": "JSON"})
    assert json_body.headers["content-type"].startswith("application/json")
    html_body = client.get("/", params={"query": "bitcoin", "render": "xml"})
    assert html_body.headers["content-type"].startswith("text/html")
    default_body = client.get("/", params={"query": "bitcoin"})
    assert default_body.headers["content-type"].startswith("text/html")


def test_oversized_render_param_rejected(client):
    assert client.get("/", params={"render": "a" * 17}).status_code == 422
    assert client.get("/", params={"render": "a" * 16}).status_code == 200


def test_llms_txt_document(client):
    response = client.get("/llms.txt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert text.startswith("# Muriel")
    # canonical link-list items: sections only accept `- [name](url): notes`
    # (strict validators reject plain text lines inside H2 sections)
    assert (
        "- [Busca render=json](http://testserver/?query=SEU_TERMO&render=json):"
        in text
    )
    assert (
        "- [API /search](http://testserver/search?query=SEU_TERMO&limit=10):"
        in text
    )
    assert "start_seconds" in text  # result fields are documented
    assert "<html" not in text  # plain text, never the page


def test_llms_txt_sections_are_canonical_link_lists(client):
    """Regression for llmstxtvalidator.org: inside an H2 section the only
    allowed lines are `- [name](url): optional notes`; anything else
    (GET recipes, bare field docs) fails the validator as plain text."""
    in_section = False
    for line in client.get("/llms.txt").text.splitlines():
        if line.startswith("## "):
            in_section = True
        elif in_section and line.strip():
            assert re.match(r"^- \[[^\]]+\]\([^)\s]+\)(?:: .+)?$", line), line


def test_llms_txt_honors_proxy_scheme(client):
    response = client.get("/llms.txt", headers={"x-forwarded-proto": "https"})
    assert "https://testserver/llms.txt" in response.text


def test_llms_txt_documents_sort_modes(client):
    """Story 7.7 — the /search sort knob and its date-order semantics are in
    the manual: the three modes, the relevance default, and the undated-last
    rule that keeps dateless episodes from masquerading as the newest."""
    text = client.get("/llms.txt").text
    assert "sort" in text
    assert "relevance" in text and "recent" in text and "oldest" in text
    assert "sem data" in text  # undated payloads sort last under date modes


def test_agent_card_documents_sort_and_score_semantics(client):
    """The render=json agent block names the sort param and records that,
    under date modes, per-result score stays a per-result relevance and is
    non-monotonic with position (so an agent never re-ranks by score)."""
    agent = client.get("/", params={"query": "x", "render": "json"}).json()["agent"]
    assert "sort" in agent["params"]
    assert "recent" in agent["params"]["sort"] and "oldest" in agent["params"]["sort"]
    assert "recent" in agent["result_fields"]["score"]


@pytest.fixture
def stats_client(tmp_path, monkeypatch) -> TestClient:
    """A client whose state store carries the full archive numbers:
    3556 videos totalling 197342 seconds, last episode #971 (Story 6.6).
    run_search is faked like the ``client`` fixture so the with-query
    envelope path doesn't need a live Qdrant."""
    monkeypatch.setattr(
        app_module,
        "run_search",
        lambda *args, **kwargs: [
            SearchResult(
                video_id="video000001",
                text="o bitcoin reagiu ao boato do dia",
                url="https://www.youtube.com/watch?v=video000001&t=30",
                start_seconds=30,
                date="2026-01-01",
                episode="1035",
                channel="mc",
                score=1.0,
            )
        ],
    )
    state = StateStore(tmp_path / "ingest.db")
    per, count = 50, 3556
    for i in range(count - 1):
        state.record_video_stats(f"sv{i:06d}", duration=per)
    state.record_video_stats(
        "sv-last",
        duration=197342 - per * (count - 1),
        episode=971,
        date="2026-10-20",
    )
    return TestClient(create_app(client=object(), state=state))


def test_render_json_includes_archive_status(stats_client):
    """The agent envelope carries the raw archive numbers (Story 6.6)."""
    status = stats_client.get(
        "/", params={"query": "bitcoin", "render": "json"}
    ).json()["archive_status"]
    assert status["videos"] == 3556
    assert status["seconds"] == 197342
    assert status["minutes"] == round(197342 / 60, 2)
    assert status["hours"] == round(197342 / 3600, 2)
    assert status["days"] == round(197342 / 86400, 2)
    assert status["last_episode"] == 971
    assert status["last_episode_date"] == "2026-10-20"


def test_render_json_unknown_stats_are_null(client):
    """A fresh archive exposes 0 videos and null for every derived stat."""
    status = client.get("/", params={"render": "json"}).json()["archive_status"]
    assert status["videos"] == 0
    assert status["seconds"] is None
    assert status["minutes"] is None
    assert status["hours"] is None
    assert status["days"] is None
    assert status["last_episode"] is None
    assert status["last_episode_date"] is None


def test_llms_txt_mentions_archive_status(client):
    assert "archive_status" in client.get("/llms.txt").text


def test_landing_page_carries_agent_instructions(client):
    html = client.get("/").text
    assert 'class="agent-note"' in html
    assert "Fale com a Muriel" in html
    # the paste-ready phrase points the agent at the machine-readable manual
    assert "quando o eddie falou sobre ilha das américas?" in html
    assert "https://muriel.eddieoz.com/llms.txt" in html
    assert 'href="/llms.txt"' in html
    # copy affordance: hidden button revealed by /static/js/copy.js only
    assert 'data-copy="#agent-phrase"' in html
    assert 'aria-label="copiar frase para o agente"' in html
    assert re.search(r'class="copy-btn"[^>]*hidden', html)
    # the human surface carries no endpoint jargon — agents get the full
    # recipe in /llms.txt and the render=json envelope instead
    assert "SEU_TERMO" not in html
    assert "GET /search" not in html
    # the box survives a query too — agents land on result URLs
    assert "agent-note" in client.get("/", params={"query": "bitcoin"}).text
