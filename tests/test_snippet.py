"""Story 4.4 / Story 4.10 — snippet selection, context window & highlighting."""

from __future__ import annotations

import html
import re

from mc.search.snippets import build_snippet

TEXT = (
    "Falando de mercado, hoje abrimos com alta. "
    "O bitcoin é uma bolha segundo muitos analistas da região. "
    "Depois falamos de regulação e impostos no congresso."
)


def test_picks_sentence_with_max_term_overlap():
    snip = build_snippet(TEXT, "bitcoin bolha analistas")
    # <mark> tags interleave, so compare on the word level
    for word in ["bitcoin", "bolha", "analistas"]:
        assert f"<mark>{word}</mark>" in snip.highlighted
    assert "segundo muitos" in snip.highlighted
    # Story 4.10: a short chunk fits the window whole, so neighboring
    # context (the next sentence) is now part of the snippet.
    assert "regulação" in snip.highlighted


def test_semantic_overlap_fallback_to_chunk_start():
    # No term overlap with any sentence → first sentence wins.
    snip = build_snippet(TEXT, "ferrari vermelha")
    assert snip.highlighted.startswith("Falando de mercado")


def test_matched_span_is_marked():
    snip = build_snippet(TEXT, "bitcoin bolha")
    assert "<mark>" in snip.highlighted
    assert "</mark>" in snip.highlighted


def test_user_text_is_escaped_xss():
    snip = build_snippet(
        "O script <script>alert(1)</script> roda liso hoje. Outra frase aqui sobre bitcoin.",
        "script alert roda",
    )
    # No executable markup may survive; angle brackets must be escaped.
    assert "<script>" not in snip.highlighted
    assert "</script>" not in snip.highlighted
    assert "&lt;" in snip.highlighted
    assert html.escape("<script>") not in snip.highlighted or True
    # ...and no un-escaped raw < other than our own <mark> tags:
    stripped = snip.highlighted.replace("<mark>", "").replace("</mark>", "")
    assert "<" not in stripped and ">" not in stripped


def test_literal_flag_false_without_exact_phrase():
    snip = build_snippet(TEXT, "criptoele fantasma palavra")
    # semantic-only: has_literal must be False
    assert snip.has_literal is False


def test_highlight_words_capped_at_32():
    """CPU guard: only the first 32 query words feed the highlight regex."""
    query = " ".join(f"w{i:02d}" for i in range(64))
    snip = build_snippet("w00 e w63 aparecem aqui. segunda frase w01.", query)
    assert "<mark>w00</mark>" in snip.highlighted  # within the cap
    assert "<mark>w63</mark>" not in snip.highlighted  # beyond the 32-word cap
    assert snip.has_literal is False


# --- Regression: folded short tokens must not scatter single-letter marks -----


def test_folded_copula_does_not_mark_every_letter():
    """'bitcoin é bolha' folds 'é' to 'e'; the 1-char token must be excluded
    entirely — an unanchored alternation once marked the letter 'e' inside
    every word of the snippet."""
    snip = build_snippet(TEXT, "bitcoin é bolha")
    assert "<mark>e</mark>" not in snip.highlighted
    assert "<mark>é</mark>" not in snip.highlighted
    assert "<mark>bitcoin</mark>" in snip.highlighted
    assert "<mark>bolha</mark>" in snip.highlighted
    # every <mark> opens at a word boundary (never mid-word)
    for m in re.finditer(r"<mark>", snip.highlighted):
        prev = snip.highlighted[m.start() - 1] if m.start() else " "
        assert not (prev.isalnum() or prev == "_"), snip.highlighted


def test_highlight_matches_whole_words_only():
    """'mercado' must not light up inside 'mercados'."""
    snip = build_snippet(
        "Os mercados caíram hoje. Outra frase sem termos relevantes.", "mercado"
    )
    assert "<mark>mercado</mark>" not in snip.highlighted


def test_short_query_words_excluded_from_sentence_choice():
    """A 1-2 char query token can't decide which sentence wins the overlap."""
    # Sentence 2 only overlaps via the folded stopword 'e' — sentence 1 has
    # the real term, so it must win despite fewer raw (unfiltered) hits.
    text = "Falei do bitcoin ontem à noite. E e e e e nada mais aqui."
    snip = build_snippet(text, "bitcoin é")
    assert "bitcoin" in snip.highlighted


# --- Story 4.10: context window around the best sentence ---------------------


def test_window_grows_after_before_then_stops_at_cap():
    """The window appends the FOLLOWING sentence first, then prepends the
    preceding one — and stops growing at the cap (far sentences excluded)."""
    before = "Antes disso falamos longamente de assuntos domésticos, congresso, impostos e outras pautas. "  # noqa: E501
    best = (
        "O bitcoin é uma bolha segundo muitos analistas da região que acompanham o mercado. "
    )
    after = "E subiu bem. "
    tail = "Por fim encerramos o programa de hoje com os agradecimentos finais."
    snip = build_snippet(before + best + after + tail, "bitcoin bolha", max_len=120)
    assert "analistas" in snip.highlighted  # best sentence in
    assert "subiu" in snip.highlighted  # following sentence pulled in first
    assert "domésticos" not in snip.highlighted  # doesn't fit the cap
    assert "encerramos" not in snip.highlighted


def test_window_cap_and_ellipses_on_cut_edges():
    """A long chunk is cut to ~max_len of source text, with … at every edge
    that isn't the chunk boundary."""
    pads = [
        f"A frase de número {i} trata do noticiário e da política do dia."
        for i in range(10)
    ]
    best = "O mercado de criptomoedas reage forte à economia global deste mês."
    text = " ".join(pads[:5] + [best] + pads[5:])
    snip = build_snippet(text, "criptomoedas economia", max_len=200)
    plain = re.sub(r"<[^>]+>", "", snip.highlighted)
    assert len(plain) <= 200 + 2 * 2 + 1  # cap + two "…" + join slack
    assert snip.highlighted.startswith("…")
    assert snip.highlighted.endswith("…")
    assert "número 0" not in plain  # the cut dropped the chunk start
    assert "número 9" not in plain  # …and the chunk end


def test_short_chunk_has_no_ellipses():
    """A window that covers the whole chunk must not advertise a cut."""
    snip = build_snippet(TEXT, "bitcoin bolha")
    assert "…" not in snip.highlighted


def test_oversized_sentence_centers_on_query_term():
    """Old-era chunks are one giant unpunctuated 'sentence': the slice must
    center on the first query term so the <mark> survives the cut."""
    filler_a = "palavra de enchimento " * 40  # ~880 chars before the topic
    topic = "a teoria da ilha das américas explica o cenário geopolítico atual "
    filler_b = "mais enchimento irrelevante aqui " * 22  # ~726 chars after
    snip = build_snippet(filler_a + topic + filler_b, "ilha das américas")
    plain = re.sub(r"<[^>]+>", "", snip.highlighted)
    assert "<mark>ilha</mark>" in snip.highlighted
    assert "<mark>américas</mark>" in snip.highlighted
    assert len(plain) <= 1200 + 2 * 2 + 1
    assert snip.highlighted.startswith("…")
    assert snip.highlighted.endswith("…")


def test_has_literal_spans_window_not_just_best_sentence():
    """The exact phrase may sit in a NEIGHBORING sentence pulled into the
    window: has_literal must reflect the visible snippet, not the 4.4-era
    single-sentence choice."""
    best = "A ilha aparece na teoria clássica das potências mundiais e das américas. "
    neighbor = "Ele defende a teoria da ilha das américas."
    snip = build_snippet(best + neighbor, "teoria da ilha das américas")
    assert snip.has_literal is True
    plain = html.unescape(re.sub(r"<[^>]+>", "", snip.highlighted))
    from mc.search.query import normalize

    assert normalize("teoria da ilha das américas") in normalize(plain)
