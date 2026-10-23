"""BDD steps for tests/bdd/ui.feature (Story 4.8) — search page + htmx.

Parses the rendered HTML with the stdlib parser; assets are asserted
vendored (no external CDN requests, works offline and without JS).
"""

from __future__ import annotations

import re
import uuid

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenarios, then, when
from qdrant_client import QdrantClient

from mc.api.app import (
    _fmt_dec_pt,
    _fmt_int_pt,
    _format_archive_status,
    create_app,
)
from mc.ingest.models import Chunk
from mc.qdrant import ensure_collection, upsert_video

scenarios("bdd/ui.feature")


# --- Story 6.5 — pt-BR formatting helpers + archive-status context ----------


def test_fmt_int_pt_thousands_separator():
    assert _fmt_int_pt(197342) == "197.342"
    assert _fmt_int_pt(3556) == "3.556"
    assert _fmt_int_pt(0) == "0"


def test_fmt_dec_pt_comma_and_rounding():
    assert _fmt_dec_pt(54.8172) == "54,82"  # rounds at 2 decimals
    assert _fmt_dec_pt(3289.0333) == "3.289,03"  # thousands sep + comma
    assert _fmt_dec_pt(2.2840) == "2,28"


def test_format_archive_status_from_seeded_numbers():
    raw = {
        "videos": 3556,
        "seconds": 197342,
        "last_episode": 971,
        "last_episode_date": "2026-10-20",
    }
    status = _format_archive_status(raw)
    assert status["last_episode"] == "#971 · 2026-10-20"
    assert status["videos"] == "3.556"
    assert status["duration"] == (
        "197.342 segundos · 3.289,03 minutos · 54,82 horas · 2,28 dias"
    )


def test_format_archive_status_omits_unknown_stats():
    # no episode/date → last-episode line omitted; None seconds → no duration
    assert _format_archive_status(
        {"videos": 0, "seconds": None, "last_episode": None, "last_episode_date": None}
    ) == {}
    partial = _format_archive_status(
        {"videos": 12, "seconds": None, "last_episode": None, "last_episode_date": None}
    )
    assert partial == {"videos": "12"}  # only the known stat survives


def test_format_archive_status_episode_without_date_still_shows():
    """A known episode renders even when its upload date is unknown, so the
    HTML stat agrees with the JSON envelope (which reports last_episode with a
    null date) instead of hiding a fact the archive actually knows."""
    status = _format_archive_status(
        {"videos": 3, "seconds": None, "last_episode": 971, "last_episode_date": None}
    )
    assert status["last_episode"] == "#971"


def test_format_archive_status_episode_and_date_together():
    status = _format_archive_status(
        {
            "videos": 3,
            "seconds": None,
            "last_episode": 971,
            "last_episode_date": "2026-10-20",
        }
    )
    assert status["last_episode"] == "#971 · 2026-10-20"

pytestmark = pytest.mark.integration

QUERY_WORD = "bolha"


def _seed(ctx, query_word: str = QUERY_WORD) -> None:
    import mc.qdrant as q

    original = q.embed_passage
    q.embed_passage = lambda texts: [[0.5] * 384 for _ in texts]
    try:
        chunks = [
            Chunk(
                video_id="video000001",
                text=f"o mercado em crise e a pergunta se bitcoin é {query_word} volto já",
                url="https://www.youtube.com/watch?v=video000001&t=30",
                start_seconds=30,
                date="2026-10-09",
                episode="1245",
                channel="morning crypto",
            ),
            Chunk(
                video_id="video000002",
                text="a carteira fria guardou as chaves com segurança",
                url="https://www.youtube.com/watch?v=video000002&t=60",
                start_seconds=60,
                date="2026-10-08",
                episode="1244",
                channel="morning crypto",
            ),
        ]
        upsert_video(ctx["client"], ctx["collection"], "video000001", chunks[:1])
        upsert_video(ctx["client"], ctx["collection"], "video000002", chunks[1:])
        ctx["state"].record_success("update")
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
    ensure_collection(client, collection)
    ctx = {"client": client, "collection": collection}
    from mc.state import StateStore

    ctx["state"] = StateStore(tmp_path / "ingest.db")
    _seed(ctx)
    ctx["app"] = create_app(
        client=client, collection=collection, state=ctx["state"]
    )
    ctx["client_http"] = TestClient(ctx["app"])
    ctx["empty_collection"] = None
    ctx["sort_collection"] = None  # set by the sort-ordering scenarios
    yield ctx
    client.delete_collection(collection)
    if ctx["empty_collection"]:
        client.delete_collection(ctx["empty_collection"])
    if ctx["sort_collection"]:
        client.delete_collection(ctx["sort_collection"])


@given("I am on the search page")
def open_page(ctx):
    ctx["response"] = ctx["client_http"].get("/")
    assert ctx["response"].status_code == 200
    ctx["html"] = ctx["response"].text


@when(parsers.parse('I type "{text}" in the search box'))
def type_query(ctx, text):
    # htmx live-swaps the /results partial as the user types
    ctx["partial"] = ctx["client_http"].get(
        "/results", params={"query": text}
    ).text
    # the no-JS fallback is a plain form GET rendering the full page
    ctx["response"] = ctx["client_http"].get("/", params={"query": text})
    ctx["html"] = ctx["response"].text


@then("results appear without a full page reload (htmx)")
def htmx_live_swap(ctx):
    assert "htmx.min.js" in ctx["html"]
    assert 'hx-get="/results"' in ctx["html"]
    assert 'hx-target="#resultados"' in ctx["html"]
    # the partial carries only the ledger — no <html> wrapper
    assert "<html" not in ctx["partial"]
    assert "tape-counter" in ctx["partial"]


@then("each result shows title, episode, date, channel and snippet")
def result_fields(ctx):
    for html in (ctx["html"], ctx["partial"]):
        assert 'class="hit-title"' in html
        assert "ep 1245" in html
        assert "2026-10-09" in html
        assert "morning crypto" in html
        assert f"<mark>{QUERY_WORD}</mark>" in html


@when(parsers.parse("I search for something with no match"))
def search_no_match(ctx):
    # a mismatch only happens on an empty archive — the dense branch of the
    # hybrid engine always returns best-effort ANN matches otherwise
    from pathlib import Path

    from mc.state import StateStore as SS

    empty_collection = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    ctx["empty_collection"] = empty_collection
    ensure_collection(ctx["client"], empty_collection)
    app = create_app(
        client=ctx["client"],
        collection=empty_collection,
        state=SS(Path(ctx["state"].path)),
    )
    ctx["response"] = TestClient(app).get("/", params={"query": "zzz-nada"})
    ctx["html"] = ctx["response"].text


@then(
    parsers.parse(
        'the page shows "{headline}" with the archive freshness line'
    )
)
def empty_state(ctx, headline):
    assert headline in ctx["html"]
    assert "acervo atualizado até" in ctx["html"]


@then("the page does not look identical to a loading state")
def empty_vs_loading(ctx):
    # loading shows the htmx indicator; the empty card is a dashed panel
    assert 'class="empty-head"' in ctx["html"]
    assert "console-busy" in ctx["html"]


@given(parsers.parse("the archive was last updated {date}"))
def archive_updated(ctx, date):
    ctx["state"].record_key("last_success:update", f"{date}T06:15:00")


@when("I open the search page")
def open_page_again(ctx):
    ctx["response"] = ctx["client_http"].get("/")
    ctx["html"] = ctx["response"].text


@then(parsers.parse('the footer shows "acervo atualizado até {date}"'))
def footer_freshness(ctx, date):
    assert f"acervo atualizado até {date}" in ctx["html"]


@then("the page shows how to point an agent at Muriel")
def agent_panel(ctx):
    html = ctx["html"]
    assert 'class="agent-note"' in html
    assert "Fale com a Muriel" in html
    # the paste-ready phrase points the agent at the machine-readable manual
    assert "quando o eddie falou sobre ilha das américas?" in html
    assert "https://muriel.eddieoz.com/llms.txt" in html
    assert 'href="/llms.txt"' in html
    # copy affordance ships hidden (no-JS safe): copy.js reveals it
    assert 'data-copy="#agent-phrase"' in html
    assert 'aria-label="copiar frase para o agente"' in html
    assert re.search(r'class="copy-btn"[^>]*hidden', html)


def test_no_external_cdn_requests(ctx):
    """Vendored assets only — the page must render fully offline."""
    html = ctx["client_http"].get("/").text
    for match in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert not match.startswith(("http://", "https://")), match
    for asset in (
        "/static/js/htmx.min.js",
        "/static/js/alpine.min.js",
        "/static/js/copy.js",
        "/static/css/pico.min.css",
        "/static/css/mc.css",
    ):
        assert ctx["client_http"].get(asset).status_code == 200


def test_page_renders_without_javascript(ctx):
    """No-JS fallback: plain form GET renders the full results inline."""
    html = ctx["client_http"].get("/", params={"query": QUERY_WORD}).text
    assert "<html" in html and "tape-counter" in html
    assert 'action="/" method="get"' in html


# --- Story 7.4 — HTML doors thread `sort` (routes, context, ordering) ----------
# The seed (see _seed) has video000001 dated 2026-10-09 and video000002 dated
# 2026-10-08, both matching QUERY_WORD, so a date sort has a deterministic order.


def test_sort_recent_orders_newest_first(ctx):
    html = ctx["client_http"].get(
        "/", params={"query": QUERY_WORD, "sort": "recent"}
    ).text
    assert html.index("2026-10-09") < html.index("2026-10-08")


def test_sort_oldest_orders_oldest_first(ctx):
    html = ctx["client_http"].get(
        "/", params={"query": QUERY_WORD, "sort": "oldest"}
    ).text
    assert html.index("2026-10-08") < html.index("2026-10-09")


def test_invalid_sort_coerces_to_relevance(ctx):
    """A bad sort on the human door is coerced (200, no reorder), never a 422."""
    resp = ctx["client_http"].get("/", params={"query": QUERY_WORD, "sort": "weird"})
    assert resp.status_code == 200
    assert "ordenado por" not in resp.text  # relevance has no ordering caption


def test_partial_results_honors_sort(ctx):
    """The /results partial honors the mode too (htmx swap target)."""
    html = ctx["client_http"].get(
        "/results", params={"query": QUERY_WORD, "sort": "recent"}
    ).text
    assert "<html" not in html  # it is the partial, not the full page
    assert html.index("2026-10-09") < html.index("2026-10-08")


def test_caption_reflects_mode(ctx):
    recent = ctx["client_http"].get(
        "/", params={"query": QUERY_WORD, "sort": "recent"}
    ).text
    assert "2 resultados · ordenado por mais recentes" in recent
    relevance = ctx["client_http"].get("/", params={"query": QUERY_WORD}).text
    assert "2 resultados" in relevance
    assert "ordenado por" not in relevance  # relevance shows the bare count


# --- Story 7.5 — the sort bar control (markup, htmx wiring, no-JS) -------------
# These assert the SERVER-rendered wiring (the radios, the form attributes, the
# console hx-include); the client-side behavior is exercised by hand in the PR.


def _sortbar(html: str) -> str:
    start = html.index('class="sortbar"')
    end = html.index("</form>", start) + len("</form>")
    return html[start:end]


def test_sortbar_selects_current_mode_per_order(ctx):
    for mode, value in (("recent", "recent"), ("oldest", "oldest")):
        html = ctx["client_http"].get(
            "/", params={"query": QUERY_WORD, "sort": mode}
        ).text
        bar = _sortbar(html)
        assert f'<option value="{value}" selected>' in bar
        assert '<option value="relevance"' in bar


def test_sortbar_default_is_relevance(ctx):
    bar = _sortbar(ctx["client_http"].get("/", params={"query": QUERY_WORD}).text)
    assert '<option value="relevance" selected>' in bar
    # the pt-BR labels, lowercase house style
    for label in ("mais relevantes", "mais recentes", "mais antigos"):
        assert label in bar
    assert '<label class="visually-hidden" for="sort-mode">ordenar por:</label>' in bar


def test_sortbar_no_js_contract(ctx):
    """A real GET form: action/method, hidden query, a sort <select> with
    three options — and no submit button: selection reorders via htmx, and
    without JS the sorted screen stays reachable by URL (decision D12)."""
    bar = _sortbar(ctx["client_http"].get("/", params={"query": QUERY_WORD}).text)
    assert 'action="/" method="get"' in bar
    assert '<input type="hidden" name="query" value="' in bar
    assert '<select' in bar and 'name="sort"' in bar
    assert bar.count("<option value=") == 3
    assert "<button" not in bar


def test_sortbar_htmx_contract(ctx):
    """The form fetches the full page, swaps #resultados, pushes the URL."""
    bar = _sortbar(ctx["client_http"].get("/", params={"query": QUERY_WORD}).text)
    assert 'hx-get="/"' in bar
    assert 'hx-select="#resultados"' in bar
    assert 'hx-target="#resultados"' in bar
    assert 'hx-push-url="true"' in bar
    assert 'hx-trigger="change"' in bar


def test_sortbar_announces_reorder_to_screen_readers(ctx):
    bar = _sortbar(ctx["client_http"].get("/", params={"query": QUERY_WORD, "sort": "recent"}).text)
    assert 'role="status"' in bar
    assert "resultados reordenados por mais recentes" in bar


def test_console_input_carries_the_sort_select(ctx):
    """Live typing on the console preserves the active sort via hx-include."""
    html = ctx["client_http"].get("/", params={"query": QUERY_WORD}).text
    start = html.index('class="console"')
    console = html[start : html.index("</form>", start)]
    assert 'hx-include=' in console
    assert "select[name='sort']" in console


def test_sortbar_absent_in_empty_state(ctx):
    """No results → no sort bar (it lives inside the results branch). An empty
    collection returns zero points, so the empty state renders without it."""
    import mc.qdrant as q

    empty = f"mc-bdd-empty-{uuid.uuid4().hex[:8]}"
    ensure_collection(ctx["client"], empty)
    original = q.embed_passage
    q.embed_passage = lambda texts: [[0.5] * 384 for _ in texts]
    try:
        app = create_app(client=ctx["client"], collection=empty, state=ctx["state"])
        html = TestClient(app).get("/", params={"query": "zzz-nada"}).text
    finally:
        q.embed_passage = original
        ctx["client"].delete_collection(empty)
    assert 'class="sortbar"' not in html


# --- Story 6.5 — "status do acervo" below everything ------------------------


@given("the archive has 3556 catalogued videos totalling 197342 seconds")
def archive_has_videos_and_seconds(ctx):
    state = ctx["state"]
    total, count, per = 197342, 3556, 50
    for i in range(count - 1):
        state.record_video_stats(f"statvid{i:06d}", duration=per)
    # record the final video here too, so this given stands alone with exactly
    # 3556 videos / 197342s (the "Agents read the same status" scenario uses it
    # without last_episode_fact). last_episode_fact then upgrades this same
    # final row with episode/date — idempotent, count unchanged.
    ctx["last_video"] = f"statvid{count - 1:06d}"
    ctx["remainder"] = total - per * (count - 1)
    state.record_video_stats(ctx["last_video"], duration=ctx["remainder"])


@given(parsers.parse("the last episode loaded is #{episode} from {date}"))
def last_episode_fact(ctx, episode, date):
    ctx["state"].record_video_stats(
        ctx["last_video"],
        duration=ctx["remainder"],
        episode=int(episode),
        date=date,
    )


@given("the archive has no recorded stats")
def archive_has_no_stats(ctx):
    pass  # a fresh state store has an empty video_stats table by default


@given("the state store raises on stats reads")
def state_raises_on_stats(ctx, monkeypatch):
    from mc.state import StateStore

    def boom(self):
        raise RuntimeError("stats db corrupt")

    monkeypatch.setattr(StateStore, "archive_stats", boom)


@then("the status block is the last content inside the footer")
def status_last_in_footer(ctx):
    html = ctx["html"]
    footer = html[html.index('<footer class="colophon">'):]
    footer = footer[: footer.index("</footer>") + len("</footer>")]
    assert 'class="archive-status"' in footer
    # below the colophon note paragraphs — the last thing on the page
    assert footer.index('class="archive-status"') > footer.rindex("colophon-note")


@then(parsers.parse('it shows "{label}" "{value}"'))
def status_shows(ctx, label, value):
    assert f"<dt>{label}</dt>" in ctx["html"], label
    assert f"<dd>{value}</dd>" in ctx["html"], value


@then("the page renders without the status block")
def no_status_block(ctx):
    assert 'class="archive-status"' not in ctx["html"]


@then("the freshness line and colophon notes still show")
def freshness_and_colophon_still_show(ctx):
    assert "acervo atualizado" in ctx["html"]
    assert "colophon-note" in ctx["html"]


@then("the page still returns 200 with the search console and agent note intact")
def page_intact_on_broken_stats(ctx):
    assert ctx["response"].status_code == 200
    assert 'class="console"' in ctx["html"]
    assert 'class="agent-note"' in ctx["html"]


@when('I request "/" with render=json')
def request_render_json(ctx):
    ctx["response"] = ctx["client_http"].get("/", params={"render": "json"})
    ctx["body"] = ctx["response"].json()


@then(
    parsers.parse(
        'the envelope includes "archive_status" with seconds {seconds} '
        "and videos {videos}"
    )
)
def envelope_has_archive_status(ctx, seconds, videos):
    status = ctx["body"]["archive_status"]
    assert status["seconds"] == int(seconds)
    assert status["videos"] == int(videos)


# --- Story 7.4 — sorting the current results by date (integration) -------------
# A dedicated 3-video collection (isolated from the default seed) so the count
# and order are exact: two dated episodes plus one undated (#1034), all matching
# QUERY_WORD.


def _seed_sort_three(ctx, d1: str, d2: str, episode: str) -> None:
    import mc.qdrant as q

    collection = f"mc-bdd-srt-{uuid.uuid4().hex[:8]}"
    ensure_collection(ctx["client"], collection)
    original = q.embed_passage  # constant dense so seeding is instant
    q.embed_passage = lambda texts: [[0.5] * 384 for _ in texts]
    try:
        specs = [
            ("rs1", d1, "1000", f"o {QUERY_WORD} reagiu ao boato"),
            ("rs2", d2, "500", f"o {QUERY_WORD} subiu de novo"),
            ("rs3", None, episode, f"o {QUERY_WORD} caiu hoje"),  # undated
        ]
        for vid, date, ep, text in specs:
            upsert_video(
                ctx["client"],
                collection,
                vid,
                [
                    Chunk(
                        video_id=vid,
                        text=text,
                        url=f"https://www.youtube.com/watch?v={vid}&t=10",
                        start_seconds=10,
                        date=date,
                        episode=ep,
                        channel="morning crypto",
                    )
                ],
            )
    finally:
        q.embed_passage = original
    ctx["sort_collection"] = collection
    ctx["sort_app"] = create_app(
        client=ctx["client"],
        collection=collection,
        state=ctx["state"],
    )
    ctx["sort_http"] = TestClient(ctx["sort_app"])


@given(
    parsers.parse(
        'the archive contains episodes dated "{d1}" and "{d2}" '
        "and an undated episode #{episode}"
    )
)
def seed_sort_three_episodes(ctx, d1, d2, episode):
    _seed_sort_three(ctx, d1, d2, episode)


@when(parsers.parse('I search "{query}" with sort "{sort_mode}"'))
def search_with_sort(ctx, query, sort_mode):
    ctx["response"] = ctx["sort_http"].get(
        "/", params={"query": query, "sort": sort_mode}
    )
    ctx["html"] = ctx["response"].text


@then(parsers.parse('the first result shows date "{date}"'))
def first_result_shows_date(ctx, date):
    # newest-first under "recent": this date precedes the older dated episode
    assert ctx["html"].index(date) < ctx["html"].index("2021-03-01")


@then('the "sem data" result appears last')
def undated_result_appears_last(ctx):
    pos = ctx["html"].index("sem data")
    assert ctx["html"].index("2026-10-20") < pos  # both dated results come first
    assert ctx["html"].index("2021-03-01") < pos


@then(parsers.parse('the caption reads "{caption}"'))
def caption_reads(ctx, caption):
    assert caption in ctx["html"], caption


@then("the page renders 200 with relevance order")
def page_200_relevance(ctx):
    assert ctx["response"].status_code == 200
    assert "3 resultados" in ctx["html"]
    assert "ordenado por" not in ctx["html"]  # "weird" coerced to relevance


# --- Story 7.5 — sort control reflects / changes the active order (wiring) -----


@given('I searched "bitcoin" with the default order')
def searched_default_order(ctx):
    ctx["response"] = ctx["client_http"].get("/", params={"query": "bitcoin"})
    ctx["html"] = ctx["response"].text


@then('the sort bar shows "mais relevantes" selected')
def bdd_sortbar_default_selected(ctx):
    assert 'class="sortbar"' in ctx["html"]
    assert '<option value="relevance" selected>' in ctx["html"]


@when('I activate "mais antigos"')
def bdd_activate_oldest(ctx):
    pass  # the assertions below verify the server-rendered wiring that drives it


@then('the request URL carries "sort=oldest" and the page URL is "/?query=bitcoin&sort=oldest"')
def bdd_url_carries_sort_oldest(ctx):
    bar = ctx["html"][ctx["html"].index('class="sortbar"') :]
    assert 'hx-get="/"' in bar  # fetches the full page
    assert 'hx-push-url="true"' in bar  # pushes the shareable URL
    assert 'value="oldest"' in bar  # the "mais antigos" option → ?sort=oldest
    assert '<input type="hidden" name="query" value="bitcoin">' in bar


@then("the console input's htmx wiring includes the sort select")
def bdd_console_includes_sort(ctx):
    start = ctx["html"].index('class="console"')
    console = ctx["html"][start : ctx["html"].index("</form>", start)]
    assert 'hx-include=' in console
    assert "select[name='sort']" in console


# --- Story 7.6 — sort bar styling lives in mc.css (string-test pattern) --------


def test_sortbar_styles_present_in_css_and_template():
    """The .sortbar grammar is authored in mc.css and referenced by the
    template; focus-visible + reduced-motion are honored and no new asset is
    introduced (the CDN test below stays green)."""
    from pathlib import Path

    repo = Path(__file__).parent.parent
    css = (repo / "src/mc/api/static/css/mc.css").read_text(encoding="utf-8")
    template = (repo / "src/mc/api/templates/results.html").read_text(encoding="utf-8")
    assert ".sortbar" in css  # the control's own styles
    assert ".sortbar-select" in css  # the dropdown's own styles
    assert "focus-visible" in css  # visible focus at every stop (existing rule)
    assert "prefers-reduced-motion" in css  # motion respected
    assert 'class="sortbar"' in template  # the template wires it up
