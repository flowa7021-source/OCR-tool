"""Rebuild OCR text from Tesseract's per-word TSV, dropping low-conf entries.

The searchable text layer that OCRmyPDF stamps into the output PDF is the
union of every word Tesseract emits, regardless of confidence. On
mixed-content scans (forms + stamps + signatures + logo areas) that
layer ends up littered with 10–40 %-confidence guesses Tesseract made at
noise — the user sees ``нe / Taw / Fam / а / ба`` next to the legitimate
body text and perceives the whole run as garbage.

This module rebuilds the per-page string shown in the results panel and
used by TXT / DOCX export from Tesseract's ``image_to_data`` TSV output,
keeping only words whose confidence meets a minimum threshold. Line
boundaries from the TSV's ``(block_num, par_num, line_num)`` hierarchy
are preserved, so paragraph structure survives the filter.

The filter is opt-in through ``OCRConfig.drop_low_conf_words`` so
existing profiles and tests see no behavioural change. The
``universal_accurate`` profile — explicitly the "maximum perceived
accuracy" preset — turns it on by default.

Scope:
    * Affects ``PageResult.text`` (the string callers see via the
      results panel, TXT export, DOCX export, downstream search).
    * Does NOT rewrite the PDF's invisible text layer. Regenerating
      that requires surgery on OCRmyPDF's hOCR → PDF step and lives
      in a separate follow-up (Step 2+).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)


#: Absolute floor for the CAPS-Cyrillic-preservation heuristic. A
#: word classified as a likely-company-name (all-caps Cyrillic, 3-7
#: chars) is still dropped below this floor — Tesseract reports
#: 10-20 % confidence on pure-noise regions where even a caps-only
#: pattern match would be false positive. 25 % is tight enough to
#: filter noise while keeping legitimate faded-ink company names
#: the ``drop_low_conf_words`` filter would otherwise lose.
_CAPS_COMPANY_MIN_CONFIDENCE: float = 25.0

#: Pattern for "almost certainly a company / brand / agency name".
#: 3-7 upper-case Cyrillic letters (with optional ``Ё``), no digits,
#: no lowercase, no punctuation. On Russian transport / invoice /
#: contract scans these tokens are almost always real — ``БЕКАМ``,
#: ``ДСК``, ``АВТОРЕСУРС``, ``ОАО``, ``ООО`` etc. — and the
#: surrounding-word confidence filter tends to drop them because
#: the layout analyser put them on stamp-overlay lines where the
#: per-word confidence comes back low. Preserving this shape of
#: token lifts Russian-business recall without re-introducing the
#: stamp noise the filter is there to kill.
_CAPS_COMPANY_RE: re.Pattern[str] = re.compile(
    r"^[А-ЯЁ]{3,7}$",
)


def _should_keep_despite_low_conf(word: str, conf: float) -> bool:
    """Return True if this (word, conf) pair should be preserved even
    though ``conf`` is below the caller's ``min_confidence``.

    Currently the only preservation heuristic is the all-caps Cyrillic
    3-7 character company-name pattern; any other word at low
    confidence falls through to the default "drop" branch. Exposed as
    a module-private helper so the tests can exercise the rule in
    isolation.
    """
    if conf < _CAPS_COMPANY_MIN_CONFIDENCE:
        return False
    return bool(_CAPS_COMPANY_RE.match(word))


def reconstruct_text_from_tsv(
    data: Mapping[str, Sequence[Any]],
    *,
    min_confidence: float,
) -> str:
    """Rebuild text from a pytesseract ``image_to_data`` dict.

    Args:
        data: The dict returned by ``pytesseract.image_to_data(...,
            output_type=Output.DICT)``. Expected keys: ``text``, ``conf``,
            ``block_num``, ``par_num``, ``line_num``, ``word_num``.
            Missing keys are tolerated — the function treats them as
            parallel zero-lists.
        min_confidence: Words with ``conf < min_confidence`` are dropped.
            Tesseract emits ``-1`` for placeholder rows (page / block /
            paragraph / line markers with no word); those are dropped
            unconditionally. Pass the same number the caller uses for
            ``PageResult.low_confidence_words`` bookkeeping so the two
            views agree.

    Returns:
        Text with one output line per TSV ``(block, par, line)`` group,
        words joined by single spaces in ascending ``word_num`` order.
        Empty when no word passes the threshold — callers should treat
        that as "no replacement, keep whatever you already had" to
        avoid blanking out a result just because confidence scoring
        was unreliable.
    """
    texts = list(data.get("text", []))
    confs = list(data.get("conf", []))
    blocks = list(data.get("block_num", []))
    pars = list(data.get("par_num", []))
    lines = list(data.get("line_num", []))
    word_nums = list(data.get("word_num", []))

    n = len(texts)
    if n == 0:
        return ""

    # Group accepted words by (block, par, line) so we can rebuild with
    # the same paragraph structure Tesseract observed. Using a dict keeps
    # insertion order by group key, then we sort explicitly by the tuple
    # so concurrent blocks / paragraphs serialise top-down reliably.
    grouped: dict[tuple[int, int, int], list[tuple[int, str]]] = {}

    for i in range(n):
        word = texts[i]
        if not isinstance(word, str):
            continue
        word = word.strip()
        if not word:
            continue  # blank rows are block/par/line markers, not words
        try:
            conf = float(confs[i]) if i < len(confs) else -1.0
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        if conf < min_confidence and not _should_keep_despite_low_conf(
            word, conf,
        ):
            continue

        def _safe_int(seq: list[Any], idx: int) -> int:
            if idx >= len(seq):
                return 0
            try:
                return int(seq[idx])
            except (TypeError, ValueError):
                return 0

        key = (
            _safe_int(blocks, i),
            _safe_int(pars, i),
            _safe_int(lines, i),
        )
        word_num = _safe_int(word_nums, i)
        grouped.setdefault(key, []).append((word_num, word))

    if not grouped:
        logger.debug(
            "reconstruct_text_from_tsv: no words passed conf>=%.1f — "
            "returning empty string, caller should keep existing text",
            min_confidence,
        )
        return ""

    out_lines: list[str] = []
    # Sort by (block, par, line) so output reflects reading order. Words
    # inside a line keep their word_num order — Tesseract emits them
    # left-to-right, which matches word_num ascending.
    for key in sorted(grouped.keys()):
        words = sorted(grouped[key], key=lambda w: w[0])
        out_lines.append(" ".join(w for _, w in words))
    return "\n".join(out_lines)


__all__ = ["reconstruct_text_from_tsv"]
