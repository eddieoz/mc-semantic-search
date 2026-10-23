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
