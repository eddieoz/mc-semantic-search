"""Story 4.3 — quote-aware ranking boost (pure unit tests, no Qdrant).

Property-style invariant (DoD): the boost NEVER changes the result count,
only the order/scores — verified deterministically over ~50 seeded
permutations with `random.Random(42)` (hypothesis is not installed).
"""

from __future__ import annotations

import random

import pytest

from mc import config
from mc.search.engine import SearchResult
from mc.search.query import normalize, parse_query
from mc.search.ranking import apply_boost, apply_sort

PHRASE = "bitcoin é bolha"
PHRASE_NORM = normalize(PHRASE)  # "bitcoin e bolha" (diacritic-folded)

FACTOR = 0.5


def _result(vid: str, score: float, *, literal: bool) -> SearchResult:
    if literal:
        text = "Todo mundo repete: bitcoin é bolha, mas eu vejo escassez e demanda."
    else:
        text = f"Hoje falamos da regulamentação de criptoativos e da custódia ({vid})."
    return SearchResult(
        video_id=vid,
        text=text,
        url=f"https://www.youtube.com/watch?v={vid}",
        start_seconds=0,
        date=None,
        episode=None,
        channel=None,
        score=score,
    )


def _is_literal(r: SearchResult) -> bool:
    return PHRASE_NORM in normalize(r.text)


VERBATIM = parse_query(f'"{PHRASE}"')
PLAIN = parse_query("regulamentação de criptoativos")


# --- Unquoted queries: literal-term coverage demotion --------------------------
# A chunk containing every content term keeps its score; each missing term
# halves it. The fixture's non-literal text contains BOTH terms of PLAIN, so
# it keeps its score — on-topic chunks are never penalized.


def test_non_verbatim_full_coverage_keeps_scores():
    """The regulation chunk contains both query terms → untouched; the
    bitcoin chunk shares no term → ×factor². Order is preserved (3.0 > 0.25)
    and count never changes."""
    results = [_result("a", 3.0, literal=False), _result("b", 1.0, literal=True)]
    boosted = apply_boost(results, PLAIN, factor=FACTOR)
    assert [r.video_id for r in boosted] == ["a", "b"]
    assert [r.score for r in boosted] == [3.0, 1.0 * FACTOR**2]


def test_non_verbatim_partial_coverage_demotes_by_missing_terms():
    """One of two content terms missing → ×factor; both missing → ×factor²."""
    query = parse_query("bitcoin é bolha")
    both = SearchResult(
        video_id="both",
        text="o analista repete que bitcoin é bolha e o mercado ignora",
        url="https://www.youtube.com/watch?v=both",
        start_seconds=0,
        date=None,
        episode=None,
        channel=None,
        score=1.0,
    )
    only_bitcoin = SearchResult(
        video_id="one",
        text="o bitcoin fechou a semana em alta",
        url="https://www.youtube.com/watch?v=one",
        start_seconds=0,
        date=None,
        episode=None,
        channel=None,
        score=1.0,
    )
    neither = _result("none", 1.0, literal=False)  # regulamentação text
    boosted = apply_boost([both, only_bitcoin, neither], query, factor=FACTOR)
    by_id = {r.video_id: r for r in boosted}
    assert by_id["both"].score == pytest.approx(1.0)  # full coverage untouched
    assert by_id["one"].score == pytest.approx(FACTOR)  # 1 of 2 missing
    assert by_id["none"].score == pytest.approx(FACTOR**2)  # 2 of 2 missing
    assert [r.video_id for r in boosted] == ["both", "one", "none"]


def test_non_verbatim_no_match_scales_uniformly_order_unchanged():
    """Pure-semantic queries: nothing matches → all scaled equally → same order."""
    query = parse_query("ferrari vermelha")
    results = [_result("a", 3.0, literal=False), _result("b", 1.0, literal=True)]
    boosted = apply_boost(results, query, factor=FACTOR)
    assert [r.video_id for r in boosted] == ["a", "b"]
    assert [r.score for r in boosted] == [3.0 * FACTOR**2, 1.0 * FACTOR**2]


def test_non_verbatim_single_result_never_rescaled():
    """With one result there is nothing to reorder — pass through untouched."""
    results = [_result("a", 2.0, literal=False)]
    assert apply_boost(results, PLAIN, factor=0.0) == results


def test_unquoted_literal_query_surfaces_literal_chunk():
    """The user-facing fix: 'bitcoin é bolha' (no quotes) ranks the on-topic
    chunk above chunks sharing no term. Scores mirror the real RRF spread
    measured on the corpus (tight clustering: 0.5 vs 0.3556) — with the wide
    synthetic gaps of the other fixtures the demotion could not reorder."""
    query = parse_query("bitcoin é bolha")
    junk1 = SearchResult(
        video_id="junk1",
        text="aos que ainda tem que ter isso em relação a lei e chegaram né",
        url="https://www.youtube.com/watch?v=junk1",
        start_seconds=0,
        date=None,
        episode=None,
        channel=None,
        score=0.5,
    )
    junk2 = SearchResult(
        video_id="junk2",
        text="o celular falei com é tentar numa atacada e deixa eu ver se é esse",
        url="https://www.youtube.com/watch?v=junk2",
        start_seconds=0,
        date=None,
        episode=None,
        channel=None,
        score=0.5,
    )
    on_topic = SearchResult(
        video_id="topic",
        text="muita gente fala sobre cripto em geral ainda é uma bolha gigante",
        url="https://www.youtube.com/watch?v=topic",
        start_seconds=0,
        date=None,
        episode=None,
        channel=None,
        score=0.3556,  # engine ranked it third
    )
    boosted = apply_boost([junk1, junk2, on_topic], query, factor=FACTOR)
    assert boosted[0].video_id == "topic"
    # missing "bitcoin" → ×0.5; the junk chunks miss both terms → ×0.25
    assert boosted[0].score == pytest.approx(0.3556 * FACTOR)
    assert [r.video_id for r in boosted[1:]] == ["junk1", "junk2"]


# --- Verbatim: demote, never drop ---------------------------------------------


def test_verbatim_demotes_results_lacking_phrase():
    results = [
        _result("para", 2.0, literal=False),
        _result("lit", 1.0, literal=True),
    ]
    boosted = apply_boost(results, VERBATIM, factor=FACTOR)
    assert len(boosted) == 2  # never dropped
    by_id = {r.video_id: r for r in boosted}
    assert by_id["lit"].score == pytest.approx(1.0)  # literal untouched
    assert by_id["para"].score == pytest.approx(2.0 * FACTOR)


def test_verbatim_reorders_descending():
    results = [
        _result("para", 10.0, literal=False),
        _result("lit", 6.0, literal=True),
        _result("para2", 8.0, literal=False),
    ]
    boosted = apply_boost(results, VERBATIM, factor=FACTOR)
    assert [r.video_id for r in boosted] == ["lit", "para", "para2"]
    assert [r.score for r in boosted] == [6.0, 5.0, 4.0]


def test_verbatim_all_literal_keeps_scores_desc():
    results = [
        _result("lit1", 1.0, literal=True),
        _result("lit2", 5.0, literal=True),
    ]
    boosted = apply_boost(results, VERBATIM, factor=FACTOR)
    assert [r.video_id for r in boosted] == ["lit2", "lit1"]
    assert [r.score for r in boosted] == [5.0, 1.0]


def test_count_never_changes_and_input_not_mutated():
    results = [
        _result("a", 3.0, literal=False),
        _result("b", 2.0, literal=True),
        _result("c", 1.0, literal=False),
    ]
    snapshot = list(results)
    boosted = apply_boost(results, VERBATIM, factor=FACTOR)
    assert len(boosted) == len(results) == 3
    assert results == snapshot  # input list untouched
    assert all(isinstance(r, SearchResult) for r in boosted)


def test_empty_list_stays_empty():
    assert apply_boost([], VERBATIM, factor=FACTOR) == []
    assert apply_boost([], PLAIN) == []


def test_result_without_text_does_not_crash():
    """Poisoned-index payload without `text` must demote, not raise (500)."""
    result = SearchResult(
        video_id="v1",
        text=None,
        url="https://www.youtube.com/watch?v=v1",
        start_seconds=0,
        date=None,
        episode=None,
        channel=None,
        score=1.0,
    )
    boosted = apply_boost([result], VERBATIM, factor=FACTOR)
    assert boosted[0].score == pytest.approx(FACTOR)  # demoted, not crashed


def test_factor_parameter_overrides_default():
    results = [_result("a", 10.0, literal=False)]
    (only,) = apply_boost(results, VERBATIM, factor=0.25)
    assert only.score == pytest.approx(2.5)


# --- Config default ------------------------------------------------------------


def test_factor_defaults_to_config_constant(monkeypatch):
    """`factor=None` reads config.SEARCH_QUOTE_BOOST at CALL time.

    Note: config.py reads the SEARCH_QUOTE_BOOST env var at import time
    into a module constant, so an env override can only take effect
    before interpreter start — the dynamic-read path is tested here by
    monkeypatching the module attribute instead.
    """
    monkeypatch.setattr(config, "SEARCH_QUOTE_BOOST", 0.3)
    results = [_result("a", 10.0, literal=False)]
    (only,) = apply_boost(results, VERBATIM)
    assert only.score == pytest.approx(3.0)


# --- Property test (DoD): count preserved, scores are a permutation -----------


def test_property_boost_preserves_count_and_permutes_scores():
    """Over 50 seeded permutations: same count, scores sorted desc, and every
    output score is exactly the original or the original * factor."""
    rng = random.Random(42)
    for _ in range(50):
        n = rng.randint(0, 12)
        results = [
            _result(f"v{i:02d}", rng.random() * 10 + 0.001, literal=rng.random() < 0.5)
            for i in range(n)
        ]
        boosted = apply_boost(results, VERBATIM, factor=FACTOR)

        assert len(boosted) == n  # count NEVER changes

        scores = [r.score for r in boosted]
        assert scores == sorted(scores, reverse=True)  # descending order

        # multiset of video ids preserved (pure permutation)
        assert sorted(r.video_id for r in boosted) == sorted(
            r.video_id for r in results
        )

        for out in boosted:
            orig = next(r for r in results if r.video_id == out.video_id)
            expected = orig.score if _is_literal(orig) else orig.score * FACTOR
            assert out.score == pytest.approx(expected)
            assert _is_literal(out) == _is_literal(orig)  # text untouched


# --- Story 7.1 — apply_sort(): relevance / recent / oldest --------------------
# Post-retrieval temporal reorder of the already-ranked set. Relevance is the
# identity (boost order preserved); recent/oldest reorder by payload date with
# undated sinking last and episode_int as the numeric tie-break. Count NEVER
# changes — sort is a pure permutation like the boost above.


def _s(video_id, *, date, episode, score, text="transcript text"):
    return SearchResult(
        video_id=video_id,
        text=text,
        url=f"https://www.youtube.com/watch?v={video_id}",
        start_seconds=0,
        date=date,
        episode=episode,
        channel="morning crypto",
        score=score,
    )


def test_apply_sort_relevance_is_identity_and_returns_new_list():
    results = [
        _s("a", date="2026-01-01", episode="1", score=0.9),
        _s("b", date="2026-02-01", episode="2", score=0.5),
        _s("c", date="2026-03-01", episode="3", score=0.1),
    ]
    out = apply_sort(results, "relevance")
    assert out is not results  # a new list, input untouched
    assert [r.video_id for r in out] == ["a", "b", "c"]


def test_apply_sort_default_mode_is_relevance():
    results = [_s("a", date=None, episode=None, score=1.0)]
    assert [r.video_id for r in apply_sort(results)] == ["a"]


def test_apply_sort_does_not_mutate_input():
    results = [
        _s("a", date="2026-01-01", episode="1", score=0.5),
        _s("b", date="2026-05-01", episode="2", score=0.9),
    ]
    snapshot = list(results)
    apply_sort(results, "recent")
    assert results == snapshot


def test_apply_sort_relevance_ignores_dates():
    """relevance mode never reorders by date — an input in non-date order stays
    put (date-desc would flip it to newest-first; relevance does not)."""
    results = [
        _s("old", date="2021-01-01", episode="1", score=0.9),
        _s("new", date="2026-10-20", episode="999", score=0.3),
    ]
    assert [r.video_id for r in apply_sort(results, "relevance")] == ["old", "new"]


def test_apply_sort_relevance_matches_boost_order():
    """The DoD regression: apply_sort(relevance) is the identity on the boost
    output — feeding it apply_boost()'s result returns it unchanged."""
    results = [
        _s("a", date="2026-01-01", episode="1", score=0.5),
        _s("b", date="2026-02-01", episode="2", score=2.0),
        _s("c", date="2026-03-01", episode="3", score=1.0),
    ]
    boosted = apply_boost(results, PLAIN, factor=FACTOR)  # re-sorted by score desc
    reordered = apply_sort(boosted, "relevance")
    assert [r.video_id for r in reordered] == [r.video_id for r in boosted]
    assert [r.video_id for r in boosted] == ["b", "c", "a"]  # score 2.0, 1.0, 0.5


def test_apply_sort_recent_dates_descending():
    results = [
        _s("old", date="2021-03-01", episode="500", score=0.7),
        _s("new", date="2026-10-20", episode="1000", score=0.5),
        _s("mid", date="2024-06-15", episode="700", score=0.9),
    ]
    out = apply_sort(results, "recent")
    assert [r.video_id for r in out] == ["new", "mid", "old"]


def test_apply_sort_oldest_dates_ascending():
    results = [
        _s("old", date="2021-03-01", episode="500", score=0.7),
        _s("new", date="2026-10-20", episode="1000", score=0.5),
        _s("mid", date="2024-06-15", episode="700", score=0.9),
    ]
    out = apply_sort(results, "oldest")
    assert [r.video_id for r in out] == ["old", "mid", "new"]


def test_apply_sort_undated_sink_last_in_both_modes():
    dated = _s("d", date="2026-10-20", episode="1000", score=0.5)
    undated_hi = _s("u1", date=None, episode="2000", score=0.9)
    undated_blank = _s("u2", date="   ", episode="1500", score=0.1)
    # recent: dated first, then undated by episode desc (2000 before 1500)
    assert [r.video_id for r in apply_sort([undated_hi, dated, undated_blank], "recent")] == [
        "d",
        "u1",
        "u2",
    ]
    # oldest: dated first, then undated by episode asc (1500 before 2000)
    assert [r.video_id for r in apply_sort([undated_hi, dated, undated_blank], "oldest")] == [
        "d",
        "u2",
        "u1",
    ]


def test_apply_sort_episode_is_numeric_not_lexical():
    """episode "1000" must beat "999" — the lexicographic trap ("999" > "1000")."""
    thousand = _s("k", date="2026-10-20", episode="1000", score=0.1)
    nineh = _s("n", date="2026-10-20", episode="999", score=0.9)
    assert [r.video_id for r in apply_sort([nineh, thousand], "recent")] == ["k", "n"]


def test_apply_sort_tie_same_date_higher_episode_then_score():
    """Same date → higher episode first; equal episode → higher score first."""
    a = _s("a", date="2026-10-20", episode="500", score=0.9)
    b = _s("b", date="2026-10-20", episode="500", score=0.1)  # same ep, lower score
    c = _s("c", date="2026-10-20", episode="499", score=0.99)  # lower ep
    assert [r.video_id for r in apply_sort([b, c, a], "recent")] == ["a", "b", "c"]


def test_apply_sort_tie_oldest_episode_ascending():
    a = _s("a", date="2026-10-20", episode="500", score=0.9)
    c = _s("c", date="2026-10-20", episode="499", score=0.99)
    assert [r.video_id for r in apply_sort([a, c], "oldest")] == ["c", "a"]


def test_apply_sort_poisoned_values_never_raise():
    """Malformed date strings sort as strings; "" and None are undated. No crash."""
    results = [
        _s("slash", date="20/10/2026", episode="1", score=0.5),
        _s("bad", date="2026-13-45", episode="2", score=0.6),
        _s("blank", date="", episode="3", score=0.7),
        _s("none", date=None, episode="4", score=0.8),
    ]
    for mode in ("recent", "oldest"):
        out = apply_sort(results, mode)  # must not raise
        assert len(out) == 4  # count preserved
        assert {r.video_id for r in out[:2]} == {"slash", "bad"}  # dated first
        assert {r.video_id for r in out[-2:]} == {"blank", "none"}  # undated last
    # relevance keeps input order, poisoned values untouched
    assert [r.video_id for r in apply_sort(results, "relevance")] == [
        "slash",
        "bad",
        "blank",
        "none",
    ]


def test_apply_sort_empty_list():
    assert apply_sort([], "recent") == []
    assert apply_sort([], "oldest") == []
    assert apply_sort([], "relevance") == []
