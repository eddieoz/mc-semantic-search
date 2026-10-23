"""Literal-aware ranking boost + temporal sort (Story 4.3 / Story 7.1).

Deterministic post-processing of engine results. Two rules, both
order-only — the result count NEVER changes, recall is preserved:

- Verbatim (quoted phrase): results whose normalized text lacks the
  phrase are DEMOTED — score multiplied by SEARCH_QUOTE_BOOST.
- Natural language (unquoted): results are demoted by
  SEARCH_QUOTE_BOOST ** (number of query content words missing from the
  chunk). A chunk containing every content word keeps its score; each
  missing word halves it. Words shorter than 3 characters are ignored as
  content terms — after diacritic folding the copula "é" becomes the
  conjunction "e", which appears in nearly every Portuguese sentence and
  carries no signal (this also mirrors snippets.MIN_HIGHLIGHT_LEN).

Results are re-sorted by score descending (stable: ties keep their
original relative order). If no result matches any term, every score is
scaled by the same factor and the order is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Literal

from mc import config
from mc.ingest.metadata import episode_int
from mc.search.engine import SearchResult
from mc.search.query import Query, normalize

SortMode = Literal["relevance", "recent", "oldest"]

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Content-term floor, mirroring snippets.MIN_HIGHLIGHT_LEN: 1-2 char tokens
# ("e", "é", "de", "a") are function words with no discrimination power.
MIN_TERM_LEN = 3


def _content_terms(normalized_query: str) -> list[str]:
    """Distinct folded query words long enough to carry signal."""
    terms: list[str] = []
    for word in _WORD_RE.findall(normalized_query):
        if len(word) >= MIN_TERM_LEN and word not in terms:
            terms.append(word)
    return terms


def _coverage_missing(haystack: str, patterns: list[re.Pattern[str]]) -> int:
    """How many term patterns are absent from the (normalized) chunk text."""
    return sum(1 for pattern in patterns if not pattern.search(haystack))


def apply_boost(
    results: list[SearchResult],
    query: Query,
    factor: float | None = None,
) -> list[SearchResult]:
    """Demote non-literal results; count never changes.

    `factor=None` reads `config.SEARCH_QUOTE_BOOST` at CALL time so an env
    override (or a test monkeypatching the module attribute) is honored
    without re-importing anything.
    """
    if factor is None:
        factor = config.SEARCH_QUOTE_BOOST
    if not results:
        return []

    if query.is_verbatim:
        phrase = query.normalized_phrase
        if not phrase:
            return list(results)
        scaled = [
            r if phrase in normalize(r.text or "") else replace(r, score=r.score * factor)
            for r in results
        ]
    else:
        terms = _content_terms(query.normalized)
        if not terms or len(results) <= 1:
            return list(results)
        patterns = [
            re.compile(rf"(?<!\w){re.escape(term)}(?!\w)") for term in terms
        ]
        scaled = []
        for r in results:
            missing = _coverage_missing(normalize(r.text or ""), patterns)
            scaled.append(
                r if not missing else replace(r, score=r.score * factor**missing)
            )

    # Stable sort: ties keep their original relative order.
    scaled.sort(key=lambda r: r.score, reverse=True)
    return scaled


def _is_undated(result: SearchResult) -> bool:
    """True when the payload ``date`` is absent or whitespace-only.

    A naive date sort would sink a genuinely-new dateless episode (e.g. #1034)
    to the bottom of "mais recentes"; Story 7.3 keeps this share ≈ 0, so this is
    a rare legacy tail that must sort last in BOTH date modes, never first.
    """
    date = result.date
    return date is None or not str(date).strip()


def _date_key(result: SearchResult) -> str:
    """Raw payload date for dated points; a constant "" for undated ones.

    Collapsing every undated date to "" is what lets the undated group keep its
    episode ordering through the date pass — otherwise a whitespace-only date
    ("   ") would sort above "" and scramble the dateless tail."""
    return "" if _is_undated(result) else (result.date or "")


def apply_sort(
    results: list[SearchResult],
    mode: SortMode = "relevance",
) -> list[SearchResult]:
    """Reorder the already-ranked set temporally; count NEVER changes.

    A pure permutation like :func:`apply_boost` — returns a new list, the input
    is untouched. ``relevance`` is the identity (the boost order, byte-identical,
    so today's behavior is a regression test). ``recent``/``oldest`` reorder by
    the raw payload ``date`` string (ISO ``YYYY-MM-DD`` sorts chronologically, so
    the comparison is a plain lexicographic one — never parsed, so a poisoned
    value demotes rather than raises).

    Ordering, most-significant first: dated before undated; date desc (recent) /
    asc (oldest); within a date, ``episode_int`` (same direction, unparseable
    episode last); within an equal date+episode, ``score`` descending. The sort
    is stable throughout, achieved by a chain of least-significant-first stable
    passes — the undated partition is the final, direction-independent pass so
    dateless points sink last in both modes.
    """
    if mode not in ("recent", "oldest"):
        return list(results)  # relevance (and any unexpected value) → identity
    reverse = mode == "recent"
    out = list(results)
    # Least-significant first; each .sort() is stable so earlier order survives.
    out.sort(key=lambda r: r.score, reverse=True)  # score desc (tie-break, both modes)
    out.sort(key=lambda r: episode_int(r.episode) or 0, reverse=reverse)  # episode value
    out.sort(key=lambda r: episode_int(r.episode) is None)  # unparseable episode last
    out.sort(key=_date_key, reverse=reverse)  # date; undated collapse to "" (see _date_key)
    out.sort(key=_is_undated)  # most-significant: undated always last (direction-independent)
    return out
