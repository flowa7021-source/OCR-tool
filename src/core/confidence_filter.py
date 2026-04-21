"""Rebuild OCR text from per-word data, dropping low-confidence entries.

The engine produces a per-word list. The pipeline synthesises a
TSV-like dict with keys ``text``, ``conf``, ``left``, ``top``,
``width``, ``height``, ``block_num``, ``par_num``, ``line_num``,
``word_num`` for compatibility with this filter and related callers.

This module rebuilds the per-page string shown in the results panel
and used by TXT / DOCX export, keeping only words whose confidence
meets a minimum threshold. Line boundaries from
``(block_num, par_num, line_num)`` are preserved.

Soft-rescue (opt-in via ``soft_rescue=True``) keeps borderline tokens
whose shape looks legitimate — short digit runs (amounts, ИНН/КПП,
dates) and clean all-caps acronyms that the engine underweights.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Soft-rescue band: words with ``conf`` in ``[SOFT_FLOOR_ABS, threshold)``
# AND above ``threshold - SOFT_MARGIN`` may be rescued if they're
# lexically clean. The absolute floor protects against wild thresholds.
SOFT_MARGIN: float = 15.0
SOFT_FLOOR_ABS: float = 45.0

_CYRILLIC_RE_STR = r"[Ѐ-ӿёЁ]"
_LATIN_RE_STR = r"[A-Za-z]"
_DIGIT_TOKEN_RE_STR = r"^[+\-]?[0-9][0-9.,\-/ ]*[0-9]$|^[0-9]$"


def _is_lexically_valid_rescue(word: str) -> bool:
    """Return True when ``word`` looks like a legitimate OCR read.

    Validity = length ≥ 3 AND single-script (all Cyrillic letters, or
    all Latin letters, or a digit/separator token).
    """
    if not word:
        return False
    w = word.strip()
    if len(w) < 3:
        return False

    if re.match(_DIGIT_TOKEN_RE_STR, w):
        return True

    if re.fullmatch(rf"{_CYRILLIC_RE_STR}+", w):
        return True
    return bool(re.fullmatch(rf"{_LATIN_RE_STR}+", w))


#: Absolute floor for the CAPS-Cyrillic-preservation heuristic.
_CAPS_COMPANY_MIN_CONFIDENCE: float = 25.0

#: Pattern for all-caps Cyrillic 3-7 letter tokens (company / brand names).
_CAPS_COMPANY_RE: re.Pattern[str] = re.compile(r"^[А-ЯЁ]{3,7}$")


def _should_keep_despite_low_conf(word: str, conf: float) -> bool:
    """Return True if this ``(word, conf)`` survives despite being below
    the caller's ``min_confidence`` (CAPS-Cyrillic company-name rule).
    """
    if conf < _CAPS_COMPANY_MIN_CONFIDENCE:
        return False
    return bool(_CAPS_COMPANY_RE.match(word))


#: Marker inserted in place of a handwritten-looking block's text.
HANDWRITTEN_MARKER: str = "⟨рукописный текст⟩"


def detect_handwritten_blocks(
    data: Mapping[str, Sequence[Any]],
    *,
    max_mean_confidence: float = 40.0,
    min_words: int = 3,
) -> set[tuple[int, int]]:
    """Return ``(block_num, par_num)`` keys for groups whose text looks
    like handwriting.

    A block qualifies when its mean per-word confidence is below
    ``max_mean_confidence`` AND it contains at least ``min_words`` valid
    words.
    """
    texts = list(data.get("text", []))
    confs = list(data.get("conf", []))
    blocks = list(data.get("block_num", []))
    pars = list(data.get("par_num", []))

    per_block_total: dict[tuple[int, int], int] = {}
    per_block_sum: dict[tuple[int, int], float] = {}
    for i, word in enumerate(texts):
        if not isinstance(word, str) or not word.strip():
            continue
        try:
            conf = float(confs[i]) if i < len(confs) else -1.0
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        try:
            b = int(blocks[i]) if i < len(blocks) else 0
            p = int(pars[i]) if i < len(pars) else 0
        except (TypeError, ValueError):
            continue
        key = (b, p)
        per_block_total[key] = per_block_total.get(key, 0) + 1
        per_block_sum[key] = per_block_sum.get(key, 0.0) + conf

    suspects: set[tuple[int, int]] = set()
    for key, count in per_block_total.items():
        if count < min_words:
            continue
        mean = per_block_sum[key] / count
        if mean < max_mean_confidence:
            suspects.add(key)
    return suspects


def reconstruct_text_from_tsv(
    data: Mapping[str, Sequence[Any]],
    *,
    min_confidence: float,
    soft_rescue: bool = False,
    handwritten_blocks: set[tuple[int, int]] | None = None,
    handwritten_marker: str = HANDWRITTEN_MARKER,
) -> str:
    """Rebuild text from a TSV-shaped ``image_to_data``-style dict.

    Expected keys: ``text``, ``conf``, ``block_num``, ``par_num``,
    ``line_num``, ``word_num``. Missing keys are tolerated.

    Words with ``conf < min_confidence`` are dropped unless they pass
    the CAPS-Cyrillic company rule or (when ``soft_rescue=True``) the
    lexical-validity check on the soft-rescue band.

    Returns the grouped text, one output line per ``(block, par, line)``.
    Flagged handwritten blocks appear as a single marker line each.
    """
    hw_blocks = handwritten_blocks if handwritten_blocks else set()
    texts = list(data.get("text", []))
    confs = list(data.get("conf", []))
    blocks = list(data.get("block_num", []))
    pars = list(data.get("par_num", []))
    lines = list(data.get("line_num", []))
    word_nums = list(data.get("word_num", []))

    n = len(texts)
    if n == 0:
        return ""

    if soft_rescue:
        rescue_min = max(min_confidence - SOFT_MARGIN, SOFT_FLOOR_ABS)
    else:
        rescue_min = min_confidence + 1.0  # disabled: no word can match

    grouped: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    rescued_count = 0

    for i in range(n):
        word = texts[i]
        if not isinstance(word, str):
            continue
        word = word.strip()
        if not word:
            continue
        try:
            conf = float(confs[i]) if i < len(confs) else -1.0
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        if conf < min_confidence:
            if _should_keep_despite_low_conf(word, conf):
                pass
            elif conf < rescue_min:
                # Below the soft-rescue floor — drop.
                continue
            elif not _is_lexically_valid_rescue(word):
                # In band but token shape isn't credible — drop.
                continue
            else:
                rescued_count += 1

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

    if not grouped and not hw_blocks:
        logger.debug(
            "reconstruct_text_from_tsv: no words passed conf>=%.1f — "
            "returning empty string, caller should keep existing text",
            min_confidence,
        )
        return ""

    if rescued_count:
        logger.debug(
            "reconstruct_text_from_tsv: soft-rescue kept %d borderline "
            "tokens in band [%.1f, %.1f)",
            rescued_count, rescue_min, min_confidence,
        )

    out_lines: list[str] = []
    emitted_hw_blocks: set[tuple[int, int]] = set()
    all_keys = set(grouped.keys())
    for b, p in hw_blocks:
        all_keys.add((b, p, 0))
    for key in sorted(all_keys):
        block_par = (key[0], key[1])
        if block_par in hw_blocks:
            if block_par in emitted_hw_blocks:
                continue
            out_lines.append(handwritten_marker)
            emitted_hw_blocks.add(block_par)
            continue
        if key not in grouped:
            continue
        words = sorted(grouped[key], key=lambda w: w[0])
        out_lines.append(" ".join(w for _, w in words))
    return "\n".join(out_lines)


__all__ = [
    "HANDWRITTEN_MARKER",
    "detect_handwritten_blocks",
    "reconstruct_text_from_tsv",
]
