"""Post-OCR rescue via fuzzy match against ``resources/ru_lexicon.txt``.

The per-word output of the EasyOCR CRNN recognizer occasionally emits
tokens that look close-but-wrong versus a known dictionary entry —
even when the visual crop is clean, Cyrillic-Latin cluster confusion
(``КППI`` vs ``КПП``) and single-edit OCR typos are common on noisy
scans. Examples from the transport-invoice corpus:

    line-level output  lexicon has          (Levenshtein)
    ``ИНЦ``            ``ИНН``                     1
    ``КППI``           ``КПП``                     1
    ``Скаnia``         ``Scania``                  2
    ``грузощправитель`` ``грузоотправитель``       1
    ``Гексаформа``     ``ГЕКСАФОРМ``               case+ending, treat via casefold

This module takes the already-OCR'd text + confidence and, for
borderline tokens, finds the closest dictionary entry and swaps in
the canonical spelling. No second OCR call — rapidfuzz's C-native
indexed search over ``ru_lexicon.txt`` returns in single-millisecond
per token regardless of dictionary size, so 2M-form lookup is the
same cost as the former 350-entry user-words.rus.
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


def _load_word_list(path: Path) -> list[str]:
    """Read one word per line from ``path``; return non-empty entries.

    Blank lines and pure-whitespace entries are dropped. The caller
    passes the full path — usually ``resources/ru_lexicon.txt`` — so
    both primary rescue and fuzzy_corrector share the same vocabulary.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("user_words_rescue: cannot read %s (%s)", path, exc)
        return []
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


class UserWordsCatalog:
    """Dictionary of known-good tokens with rapidfuzz-backed lookup.

    Indexes lower-cased forms into per-length buckets so the
    rapidfuzz ``process.extractOne`` call is bounded to the slice of
    candidates that could possibly match at the length-scaled edit
    budget — on a 2M-form lexicon this keeps per-token rescue latency
    at ~1-2 ms regardless of dictionary size.

    Falls back to a pure-Python exact-match lookup when rapidfuzz is
    unavailable (development / stripped builds) — rescue becomes a
    no-op for non-exact tokens but never raises.
    """

    def __init__(self, words: Iterable[str]) -> None:
        # Per-length buckets hold the casefolded form so closest()
        # runs rapidfuzz over the exact strings it will compare —
        # casefolding on every lookup on a 2M-form dict wasted ~50ms
        # per token. The original-case form is kept alongside so we
        # can return surface-level canonical when the input preserves
        # mixed casing.
        self._by_length: dict[int, list[tuple[str, str]]] = {}
        self._by_length_folded: dict[int, list[str]] = {}
        self._exact: set[str] = set()
        self._exact_casefold: dict[str, str] = {}
        for w in words:
            folded = w.casefold()
            self._exact.add(w)
            self._exact_casefold[folded] = w
            self._by_length.setdefault(len(folded), []).append((folded, w))
            self._by_length_folded.setdefault(len(folded), []).append(folded)

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

        Uses rapidfuzz's C-native ``process.extractOne`` over the
        per-length candidate slice so lookup scales to 2M-form
        dictionaries without the O(N) pure-Python Levenshtein walk.
        """
        if not word:
            return None
        word_fold = word.casefold()
        max_edits = (
            _MAX_EDIT_LONG
            if len(word_fold) >= _LENGTH_THRESHOLD_FOR_LONG
            else _MAX_EDIT_SHORT
        )
        try:
            from rapidfuzz import process
            from rapidfuzz.distance import Levenshtein
        except ImportError:  # pragma: no cover — rapidfuzz is a hard dep
            return None

        best_match: str | None = None
        best_dist: int = max_edits + 1
        for length in range(
            max(1, len(word_fold) - max_edits),
            len(word_fold) + max_edits + 1,
        ):
            folded_bucket = self._by_length_folded.get(length)
            if not folded_bucket:
                continue
            # ``Levenshtein.distance`` with ``score_cutoff`` lets
            # rapidfuzz short-circuit any candidate whose distance is
            # above the remaining budget — effectively a BK-tree-like
            # prune implemented in native C.
            cutoff = best_dist - 1 if best_dist <= max_edits else max_edits
            try:
                hit = process.extractOne(
                    word_fold,
                    folded_bucket,
                    scorer=Levenshtein.distance,
                    score_cutoff=cutoff,
                )
            except Exception:  # noqa: BLE001
                hit = None
            if hit is None:
                continue
            _folded, dist, idx = hit
            if dist < best_dist:
                best_dist = int(dist)
                # Map folded-bucket index back to the original-case form.
                best_match = self._by_length[length][idx][1]
                if best_dist == 0:
                    return best_match, 0
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
