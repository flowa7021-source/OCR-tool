"""Post-OCR rescue via fuzzy match against ``user-words.rus``.

Tesseract's DAWG bias via ``user_words`` at primary OCR time nudges
the LSTM toward vocabulary entries, but the nudge is soft — on
faded / noisy scans the line-level pass still emits tokens that
look close-but-wrong versus a known dictionary entry. Examples
from the user's transport-invoice corpus:

    line-level output  user-words.rus has   (Levenshtein)
    ``ИНЦ``            ``ИНН``                     1
    ``КППI``           ``КПП``                     1
    ``Скаnia``         ``Scania``                  2
    ``грузощправитель`` ``грузоотправитель``       1
    ``Гексаформа``     ``ГЕКСАФОРМ``               case+ending, treat via casefold

This module takes the already-OCR'd text + confidence and, for
borderline tokens, finds the closest dictionary entry and swaps in
the canonical spelling. No second Tesseract call — a dict lookup +
Levenshtein comparison is ~100× cheaper than re-OCR and handles the
class of errors where the CROP is good but the LSTM's vocabulary
wasn't biased strongly enough.

Complements the per-word image rescues in
:mod:`src.core.per_word_image_rescue` (which re-OCR faded or
tiny-glyph crops) — use both when available.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


#: Confidence band in which the rescue fires. Same lower bound as
#: :mod:`per_word_image_rescue` (below 30 is pure noise, dict lookup
#: would chase random string matches), but the upper bound is a bit
#: tighter at 75 — above that the line-level output is trustworthy
#: enough that swapping to a dictionary entry risks false positives
#: (a correctly-read surname that happens to be Levenshtein-1 from
#: ``СКАНИЯ``).
_RESCUE_MIN_ORIGINAL_CONF: float = 30.0
_RESCUE_MAX_ORIGINAL_CONF: float = 75.0

#: Conf bump applied when the rescue swaps. The substitution passed
#: a dictionary-membership check, which is a stronger signal than
#: either the single-pass Tesseract confidence OR the image-rescue
#: lift threshold — so we bump to max(original, 85) to ensure the
#: downstream adaptive threshold can't drop the rescued word.
_RESCUE_CONF_FLOOR: float = 85.0

#: Maximum Levenshtein distance accepted as a match. Scales with
#: the candidate's length — short tokens (3-5 chars) must match
#: within 1 edit, longer tokens allow 2. Prevents ``КПП`` from
#: resolving to every 3-letter dictionary entry at distance 2.
_MAX_EDIT_SHORT: int = 1
_MAX_EDIT_LONG: int = 2
_LENGTH_THRESHOLD_FOR_LONG: int = 6


def _levenshtein(a: str, b: str) -> int:
    """Iterative two-row Levenshtein edit distance.

    Pure Python, no dependencies. For the 350-entry user-words
    dictionary the inner loop runs once per ``(candidate, entry)``
    pair — ~300 comparisons per OCR token, negligible vs a Tesseract
    call.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # Ensure a is the shorter string so the work-row size is min(la, lb)+1.
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        cur = [j] + [0] * la
        for i in range(1, la + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[i] = min(
                cur[i - 1] + 1,
                prev[i] + 1,
                prev[i - 1] + cost,
            )
        prev = cur
    return prev[la]


def _load_word_list(path: Path) -> list[str]:
    """Read one word per line from ``path``; return non-empty entries.

    Blank lines and pure-whitespace entries are dropped. The caller
    passes the full path — usually
    ``resources/tessdata/user-words.rus`` — so the same file the
    primary OCR pass bundles into Tesseract's DAWG also feeds this
    post-OCR rescue, keeping the two vocabulary views consistent.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("user_words_rescue: cannot read %s (%s)", path, exc)
        return []
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


class UserWordsCatalog:
    """Dictionary of known-good tokens, indexed by length for fast
    fuzzy lookup.

    Lookup cost is ``O(entries_of_similar_length)`` per query — for
    a 350-entry dictionary that's ~50 comparisons per word, which
    is cheap compared to Tesseract re-OCR. No indexing required.
    """

    def __init__(self, words: Iterable[str]) -> None:
        # Group by exact length for fast same-length lookups. Also
        # keep a set for O(1) membership checks to short-circuit
        # exact matches.
        self._by_length: dict[int, list[str]] = {}
        self._exact: set[str] = set()
        self._exact_casefold: dict[str, str] = {}
        for w in words:
            self._exact.add(w)
            self._exact_casefold[w.casefold()] = w
            self._by_length.setdefault(len(w), []).append(w)

    @classmethod
    def from_file(cls, path: Path) -> UserWordsCatalog:
        return cls(_load_word_list(path))

    def __len__(self) -> int:
        return len(self._exact)

    def contains(self, word: str) -> bool:
        """Exact match (case-sensitive) OR case-fold match."""
        return word in self._exact or word.casefold() in self._exact_casefold

    def closest(self, word: str) -> tuple[str, int] | None:
        """Return ``(best_match, distance)`` under the length-scaled
        edit-distance cap, or ``None`` when nothing qualifies.

        Considers dictionary entries whose length differs from the
        candidate's by at most the same edit budget — a Levenshtein
        distance of 2 requires at least 2 character differences, so
        entries of length |len(word) - 2| or nearer are the only
        possible candidates.
        """
        if not word:
            return None
        max_edits = (
            _MAX_EDIT_LONG
            if len(word) >= _LENGTH_THRESHOLD_FOR_LONG
            else _MAX_EDIT_SHORT
        )
        best_match: str | None = None
        best_dist: int = max_edits + 1
        # Iterate lengths within the edit budget.
        for length in range(
            max(1, len(word) - max_edits), len(word) + max_edits + 1,
        ):
            for entry in self._by_length.get(length, ()):
                d = _levenshtein(word, entry)
                if d < best_dist:
                    best_dist = d
                    best_match = entry
                    if d == 0:
                        return entry, 0
        if best_match is None:
            return None
        return best_match, best_dist


def _is_rescuable(word: str, conf: float) -> bool:
    """Gate: word is borderline-conf and long enough to look up.

    Short tokens (< 3 chars) are excluded — a 2-char candidate at
    Levenshtein 1 from every 3-char dictionary entry is a false-
    positive factory.
    """
    if conf < _RESCUE_MIN_ORIGINAL_CONF:
        return False
    if conf > _RESCUE_MAX_ORIGINAL_CONF:
        return False
    stripped = word.strip()
    return len(stripped) >= 3 and any(ch.isalpha() for ch in stripped)


def user_words_fuzzy_rescue(
    word: str,
    confidence: float,
    catalog: UserWordsCatalog,
) -> tuple[str, float]:
    """Fuzzy-match ``word`` against ``catalog`` and return the corrected
    pair.

    Returns the ORIGINAL ``(word, confidence)`` on any of:

      * Catalog is empty.
      * Word is outside the rescuable conf band / too short.
      * Word is ALREADY an exact (or case-fold) catalog entry — the
        line-level OCR is already correct, a fuzzy pass would risk
        swapping to a same-length near-neighbour (``ИНН`` → ``ООО``
        at Levenshtein 2).
      * No catalog entry within the length-scaled edit budget.

    On a successful match returns ``(catalog_entry, max(orig, 85))``
    — the confidence is floored at 85 because a dictionary match is
    a stronger signal than either language's single-pass
    self-reported confidence and shouldn't be droppable by the
    adaptive-threshold filter downstream.
    """
    if len(catalog) == 0:
        return word, confidence
    if not _is_rescuable(word, confidence):
        return word, confidence
    if catalog.contains(word):
        # Already canonical; nothing to rescue.
        return word, max(confidence, _RESCUE_CONF_FLOOR)
    result = catalog.closest(word)
    if result is None:
        return word, confidence
    match, distance = result
    logger.debug(
        "user_words_rescue: %r (%.1f) → %r (distance %d)",
        word, confidence, match, distance,
    )
    return match, max(confidence, _RESCUE_CONF_FLOOR)


@lru_cache(maxsize=4)
def _cached_catalog(path_str: str) -> UserWordsCatalog:
    """Memoised catalog loader — the pipeline hits ``_compute_confidences``
    once per page but the dictionary is identical across pages, so
    we read+parse once and reuse."""
    return UserWordsCatalog.from_file(Path(path_str))


def load_user_words_catalog(path: Path | None) -> UserWordsCatalog | None:
    """Return a catalog for ``path`` or ``None`` when the file is
    missing / unreadable.
    """
    if path is None:
        return None
    if not path.is_file():
        return None
    return _cached_catalog(str(path))


__all__ = [
    "UserWordsCatalog",
    "load_user_words_catalog",
    "user_words_fuzzy_rescue",
]
