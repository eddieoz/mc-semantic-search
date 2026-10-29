"""BDD step definitions for tests/bdd/snippet-window.feature (Story 4.10).

Pure-unit BDD: the snippet builder is a pure function of (chunk text,
query), so no qdrant container or embedder is needed — the scenarios pin
the user-visible contract for both caption eras (punctuated vs one giant
unpunctuated sentence).
"""

from __future__ import annotations

import re

from pytest_bdd import given, scenarios, then, when

from mc.search.snippets import MAX_SNIPPET_CHARS, build_snippet

scenarios("bdd/snippet-window.feature")

QUERY = "ilha das américas"

_PAD = "Os apresentadores comentam o noticiário do dia entre si por um instante. "

# Each side alone must exceed the window cap, otherwise the whole chunk fits
# inside the snippet and nothing is cut (the cap moved to 1200 in c088db8).
_SIDE = MAX_SNIPPET_CHARS // len(_PAD) + 2

_FILL = "palavra de enchimento do noticiário "
_REPEAT = MAX_SNIPPET_CHARS // len(_FILL) + 4


def _plain(snippet_html: str) -> str:
    return re.sub(r"<[^>]+>", "", snippet_html)


@given("a punctuated chunk with the query topic in the middle", target_fixture="chunk")
def punctuated_chunk():
    # Pad sentences each side keep the topic sentence mid-chunk; the
    # surrounding sentences are what the window must surface.
    return _PAD * _SIDE + "A teoria da ilha das américas organiza o poder global. " + _PAD * _SIDE


@given("an unpunctuated chunk with the query deep inside", target_fixture="chunk")
def unpunctuated_chunk():
    # Old-era captions: no punctuation, one giant "sentence" longer than the
    # cap, query buried past the halfway mark.
    return (
        _FILL * _REPEAT
        + "a teoria da ilha das américas explica o cenário geopolítico atual "
        + _FILL * _REPEAT
    )


@when("I build its snippet for \"ilha das américas\"", target_fixture="snippet")
def build(chunk):
    return build_snippet(chunk, QUERY)


@then("the snippet contains sentences around the best match")
def snippet_has_context(snippet):
    plain = _plain(snippet.highlighted)
    assert "ilha das américas" in plain
    # context BEFORE and AFTER the topic sentence must be visible
    assert "noticiário" in plain


@then("the query terms are marked")
def snippet_marks_terms(snippet):
    assert "<mark>ilha</mark>" in snippet.highlighted
    assert "<mark>américas</mark>" in snippet.highlighted


@then("the snippet is cut with ellipses at the edges")
def snippet_has_ellipses(snippet):
    assert snippet.highlighted.startswith("…")
    assert snippet.highlighted.endswith("…")


@then("the snippet stays within the context cap")
def snippet_within_cap(snippet):
    # The cap bounds the source text; <mark> tags and the two "…" add a
    # few chars on top, hence the small slack.
    assert len(_plain(snippet.highlighted)) <= MAX_SNIPPET_CHARS + len("…") * 2 + 2
