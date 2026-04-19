"""Line-level garbage filter for OCR output.

Tesseract on low-quality scans frequently emits "garbage" lines —
ruled borders it mistook for text, speckle noise segmented as glyphs,
table frames it couldn't separate from text runs. These lines pollute
the final searchable PDF (Ctrl-F finds false positives) and copy-
paste (user gets ``!!!@#$%`` between real paragraphs).

The filter runs on the recognised text post-OCR, not on the image.
Two levels of aggressiveness trade user perception of completeness
against cleanliness:

  * :attr:`GarbageStrictness.LENIENT` — default for production
    profiles. Drops only the clearly-mechanical garbage: symbol
    walls, ruler lines made of repeated ``-``/``=``/``_``. Leaves
    anything remotely word-like alone.

  * :attr:`GarbageStrictness.STRICT` — tighter thresholds for messy
    scans. Additionally drops single-letter orphan lines and any
    line where non-letters account for ≥ 40 % of characters.

  * :attr:`GarbageStrictness.DISABLED` — pass-through. Useful during
    profile development when the user wants to SEE what Tesseract
    emitted without any cleanup.

Design invariants:
  * Never touches whitespace or line structure within kept lines —
    the filter is strictly line-level drop, not a reflow.
  * Never drops a line with digits-and-whitespace-only (invoice
    totals like ``1 250 000`` are legitimate content).
  * Never drops a line whose letters dominate (any line where
    ``letter_count >= non_letter_count`` survives lenient).
"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class GarbageStrictness(Enum):
    """How aggressively to drop garbage lines."""

    DISABLED = "disabled"
    LENIENT = "lenient"
    STRICT = "strict"


#: Character class threshold for "ruler-like" lines. A line built of
#: three or more repeats of one punctuation char is a ruler.
_RULER_MIN_REPEAT: int = 3

#: Characters that, when repeated, form decorative rulers /
#: underlines rather than text. Extended list catches the common
#: shapes Tesseract generates from noisy horizontal rules.
_RULER_CHARS: frozenset[str] = frozenset("-=_*~`.•·")


def _is_ruler_line(stripped: str) -> bool:
    """True if the line is a repeated-punctuation ruler.

    ``--------``, ``========``, ``~~~~~``, ``... ...`` all qualify.
    Whitespace between ruler characters is tolerated so Tesseract's
    segmented-dot-ruler output still counts.
    """
    condensed = "".join(stripped.split())
    if len(condensed) < _RULER_MIN_REPEAT:
        return False
    ruler_chars_in_line = [c for c in condensed if c in _RULER_CHARS]
    return (
        len(ruler_chars_in_line) >= _RULER_MIN_REPEAT
        and len(ruler_chars_in_line) == len(condensed)
    )


def _char_stats(stripped: str) -> tuple[int, int, int]:
    """Return ``(letter_count, digit_count, other_count)``.

    Whitespace is excluded from ``other_count`` — whitespace is
    structural, not content. ``other`` captures the characters that
    drive the garbage heuristic (symbols, punctuation, Tesseract
    artefacts like ``|``, ``\\``, ``«»``).
    """
    letter = 0
    digit = 0
    other = 0
    for ch in stripped:
        if ch.isspace():
            continue
        if ch.isalpha():
            letter += 1
        elif ch.isdigit():
            digit += 1
        else:
            other += 1
    return letter, digit, other


def _is_garbage_lenient(stripped: str) -> bool:
    """Return True if ``stripped`` is mechanical garbage under lenient
    rules. ``stripped`` must already be ``line.strip()``."""
    if not stripped:
        return False  # blank line handled separately (preserved)

    if _is_ruler_line(stripped):
        return True

    letter, digit, other = _char_stats(stripped)
    total_content = letter + digit + other
    if total_content == 0:
        return False

    # Pure digits-and-whitespace (invoice totals) never counts as
    # garbage, regardless of how long the run is.
    if letter == 0 and other == 0 and digit >= 1:
        return False

    # Symbol-wall heuristic: ≥ 70 % non-letters AND non-digits means
    # the line has almost no content Tesseract could have recognised
    # legitimately.
    symbol_ratio = other / total_content
    return symbol_ratio >= 0.7


def _is_garbage_strict(stripped: str) -> bool:
    """Strict rules: lenient + orphan letters + more aggressive
    symbol ratio (40% instead of 70%)."""
    if _is_garbage_lenient(stripped):
        return True
    if not stripped:
        return False

    letter, digit, other = _char_stats(stripped)
    total_content = letter + digit + other
    if total_content == 0:
        return False

    # Orphan single-letter line like ``l`` (from a thin ruler
    # Tesseract confused with a letter). Dropped in strict,
    # preserved in lenient because legitimate short tokens
    # (``No``, ``ИП``) have ≥ 2 characters.
    if total_content == 1 and letter == 1:
        return True

    # More aggressive symbol threshold. Keeps anything letter-
    # dominated; drops lines where letters are the minority.
    return total_content >= 3 and (other / total_content) >= 0.4


def filter_garbage_lines(
    text: str, strictness: GarbageStrictness,
) -> str:
    """Drop garbage lines from ``text`` according to ``strictness``.

    Preserves blank lines and never touches per-line whitespace — the
    filter is line-granular. Returned text has the same line endings
    as the input.

    Args:
        text: OCR'd text, typically post-whitespace-normalisation.
        strictness: :class:`GarbageStrictness` level.

    Returns:
        Cleaned text. Empty input returns empty output.
    """
    if strictness is GarbageStrictness.DISABLED:
        return text
    if not text:
        return text

    is_garbage = (
        _is_garbage_strict
        if strictness is GarbageStrictness.STRICT
        else _is_garbage_lenient
    )

    kept: list[str] = []
    dropped = 0
    for line in text.splitlines(keepends=False):
        stripped = line.strip()
        if is_garbage(stripped):
            dropped += 1
            continue
        kept.append(line)

    if dropped:
        logger.debug(
            "garbage_filter: dropped %d line(s) at strictness=%s",
            dropped, strictness.value,
        )
    # Rejoin using ``\n`` — OCR pipeline runs on normalised text
    # where platform-specific line endings don't matter.
    return "\n".join(kept)


__all__ = ["GarbageStrictness", "filter_garbage_lines"]
