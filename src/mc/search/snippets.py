"""Snippet selection & highlighting (Story 4.4), context window (Story 4.10).

Picks the sentence with maximum term overlap with the query, falls back
to the chunk start on pure-semantic matches, and highlights matched
spans. All user-facing text is HTML-escaped BEFORE any <mark> is added,
so query/content can never inject markup (XSS).

Story 4.10: the snippet is a ~``MAX_SNIPPET_CHARS`` window grown around
the best sentence (following sentence first, then the preceding one)
instead of that sentence alone. Why: YouTube's newer captions are heavily
punctuated, so the 4.4 single-sentence rule collapsed snippets to one
short line, while the older unpunctuated captions rendered the whole
~1100-char chunk as "one sentence". The capped window gives both caption
eras the same amount of readable context, and a giant single sentence is
sliced centered on the first query term so the <mark> survives the cut.
The cap bounds the source text; <mark> tags and the two "…" add a few
chars on top.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

from mc.search.query import normalize

# Capture group keeps each sentence's trailing punctuation so the window
# re-joins into faithful running text.
_SENTENCE_SPLIT_RE = re.compile(r"([.!?]+)")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# CPU guard: the per-result highlight regex is an alternation of every
# distinct query word (× up to 50 results/request) — cap the word list so a
# pathological query can't amplify regex work.
MAX_HIGHLIGHT_WORDS = 32

# Tokens shorter than 3 chars carry no highlight signal in Portuguese: the
# diacritic fold turns the copula "é" into the conjunction "e", and an
# unanchored 1-char pattern would mark the letter "e" inside every word.
MIN_HIGHLIGHT_LEN = 3

# Context window cap (Story 4.10). Bounds the SOURCE text handed to the
# highlighter — the rendered snippet is a few chars longer (<mark> tags,
# two "…") by design.
MAX_SNIPPET_CHARS = 1200


@dataclass(frozen=True)
class Snippet:
    highlighted: str
    has_literal: bool


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(normalize(text))


def _content_words(text: str) -> list[str]:
    """Query words worth matching: folded words of length >= MIN_HIGHLIGHT_LEN."""
    return [w for w in _words(text) if len(w) >= MIN_HIGHLIGHT_LEN]


def _split_sentences(text: str) -> list[str]:
    """Sentences as running text, each keeping its trailing punctuation."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    sentences = []
    for i in range(0, len(parts), 2):
        sent = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        sent = sent.strip()
        if sent:
            sentences.append(sent)
    return sentences


def _best_sentence_index(sentences: list[str], qwords: set[str]) -> int:
    """Index of the sentence with maximum query-term overlap (first wins
    on ties, so a pure-semantic match falls back to the chunk start)."""
    best_idx = 0
    best_score = 0
    for i, sentence in enumerate(sentences):
        score = len(qwords & set(_words(sentence)))
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def _expand_window(
    sentences: list[str], best_idx: int, max_len: int
) -> tuple[int, int]:
    """Grow the contiguous window [lo, hi] around ``best_idx``.

    Priority alternates after → before → after…, honoring the "context
    after the hit reads more naturally" rule; a side that can't fit (or is
    exhausted) yields to the other, and the window stops as soon as neither
    side fits. Only adjacent sentences are ever added — skipping a
    too-long neighbor would splice non-adjacent text into a fake narrative.
    """
    lo = hi = best_idx
    length = len(sentences[best_idx])
    prefer_after = True
    while True:
        idx = hi + 1 if prefer_after else lo - 1
        if 0 <= idx < len(sentences) and length + 1 + len(sentences[idx]) <= max_len:
            if prefer_after:
                hi = idx
            else:
                lo = idx
            length += 1 + len(sentences[idx])
            prefer_after = not prefer_after
            continue
        # Preferred side is full: try the other side once before stopping.
        other = lo - 1 if prefer_after else hi + 1
        if 0 <= other < len(sentences) and length + 1 + len(sentences[other]) <= max_len:
            if prefer_after:
                lo = other
            else:
                hi = other
            length += 1 + len(sentences[other])
            continue
        return lo, hi


def _highlight_pattern(qwords: set[str]) -> re.Pattern[str]:
    """Whole-word alternation of the query words (word boundaries are
    mandatory: without them a folded 1-char token ("é" → "e") matches the
    letter "e" inside every word)."""
    return re.compile(
        r"(?<!\w)(" + "|".join(re.escape(w) for w in sorted(qwords)) + r")(?!\w)",
        re.IGNORECASE,
    )


def _fold_with_map(text: str) -> tuple[str, list[int]]:
    """Diacritic-stripped copy of ``text`` plus an index map back into it.

    Query words are stored FOLDED (normalize strips accents), so the
    highlight pattern can only ever match a folded haystack — before
    Story 4.10 that meant accented terms ("Américas") were never marked.
    The map lets us match on the folded copy while <mark> wraps the
    ORIGINAL spelling (NFKD folding can change length, hence per-char
    mapping rather than a fixed offset)."""
    chars: list[str] = []
    idx_map: list[int] = []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKD", ch):
            if not unicodedata.combining(c):
                chars.append(c)
                idx_map.append(i)
    return "".join(chars), idx_map


def _center_on_term(
    text: str, qwords: set[str], max_len: int
) -> tuple[str, bool, bool]:
    """Slice an oversized "sentence" (old unpunctuated captions) to
    ``max_len`` chars centered on the first query-term occurrence, so the
    mark survives the cut; falls back to the text start on pure-semantic
    matches. Edges are trimmed to word boundaries. Returns
    (slice, cut_start, cut_end)."""
    start = 0
    if qwords:
        folded, idx_map = _fold_with_map(text)
        match = _highlight_pattern(qwords).search(folded)
        if match:
            center = (idx_map[match.start()] + idx_map[match.end() - 1] + 1) // 2
            start = max(0, min(center - max_len // 2, len(text) - max_len))
    end = min(len(text), start + max_len)
    if start > 0:  # never open a snippet mid-word
        space = text.find(" ", start)
        if space != -1 and space < end:
            start = space + 1
    if end < len(text):  # …nor close one
        space = text.rfind(" ", start, end)
        if space > start:
            end = space
    return text[start:end], start > 0, end < len(text)


def build_snippet(text: str, query: str, *, max_len: int = MAX_SNIPPET_CHARS) -> Snippet:
    qwords = set(_content_words(query)[:MAX_HIGHLIGHT_WORDS])
    sentences = _split_sentences(text)
    if not sentences:
        return Snippet(highlighted=html.escape(text[:max_len]), has_literal=False)

    best_idx = _best_sentence_index(sentences, qwords)
    lo, hi = _expand_window(sentences, best_idx, max_len)
    window = " ".join(sentences[lo : hi + 1])

    cut_start, cut_end = lo > 0, hi < len(sentences) - 1
    if len(window) > max_len:
        # One giant "sentence" (unpunctuated captions): center the slice on
        # the first query term instead of cutting the match away.
        window, s_cut, e_cut = _center_on_term(window, qwords, max_len)
        cut_start, cut_end = cut_start or s_cut, cut_end or e_cut

    # has_literal reflects the VISIBLE window (pre-ellipses): the exact
    # phrase may sit in a neighboring sentence pulled into the window.
    has_literal = _has_literal_phrase(window, query)
    source = ("… " if cut_start else "") + window + (" …" if cut_end else "")
    return Snippet(highlighted=_highlight(source, qwords), has_literal=has_literal)


def _has_literal_phrase(text: str, query: str) -> bool:
    if not query.strip():
        return False
    return normalize(query) in normalize(text)


def _highlight(sentence: str, qwords: set[str]) -> str:
    """Escape first, then add <mark> around whole-word query matches.

    The pattern is matched against a diacritic-folded copy of the escaped
    text (with an index map) so accented terms still light up, while the
    <mark> wraps the ORIGINAL spelling. The regex operates on word
    characters only, so escaping first is safe; "…" is not a word
    character and passes through untouched.
    """
    if not qwords:
        return html.escape(sentence)
    escaped = html.escape(sentence)
    folded, idx_map = _fold_with_map(escaped)
    pattern = _highlight_pattern(qwords)
    pieces: list[str] = []
    last = 0
    for match in pattern.finditer(folded):
        start = idx_map[match.start()]
        end = idx_map[match.end() - 1] + 1
        pieces.append(escaped[last:start])
        # escaped[start:end] is already HTML-escaped — no second escape
        pieces.append(f"<mark>{escaped[start:end]}</mark>")
        last = end
    pieces.append(escaped[last:])
    return "".join(pieces)
