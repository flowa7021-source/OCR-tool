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

Soft-rescue (opt-in via ``soft_rescue=True``):
    Tesseract systematically underweights confidence on short numeric
    runs (amounts, ИНН/КПП, dates) and clean all-caps acronyms — a
    legitimate ``7813266190`` or ``ООО`` often reads at 55–59 % on a
    noisy form even though it's unambiguous. A strict drop at
    threshold=60 then eats real content along with the stamp garbage.
    Soft-rescue keeps words in band ``[max(threshold-15, 45),
    threshold)`` when the token passes a lexical-validity check:
    length ≥ 3, single-script (all Cyrillic letters OR all Latin
    letters OR all-digit with optional separators). Mixed-script
    short tokens like ``нe`` / ``Taw`` still drop because the OCR
    typically IS wrong there — the only words we rescue are the ones
    where the shape of the token makes the OCR read credible.

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

# Soft-rescue band: words with ``conf`` in ``[SOFT_FLOOR_ABS, threshold)``
# AND above ``threshold - SOFT_MARGIN`` may be rescued if they're
# lexically clean. The absolute floor protects against wild thresholds
# (e.g., ``threshold=30`` would otherwise rescue 15-conf noise).
SOFT_MARGIN: float = 15.0
SOFT_FLOOR_ABS: float = 45.0

# Unicode ranges for single-script validity. We stay narrow on purpose —
# digits + Cyrillic letters + Latin letters covers ≥99% of Russian
# business-doc content; adding Greek / extended Latin / Cyrillic
# Supplement just widens the attack surface for noise.
_CYRILLIC_RE_STR = r"[\u0400-\u04FFёЁ]"
_LATIN_RE_STR = r"[A-Za-z]"
# Digit tokens admit common separators: 1,200.50 / 7813-2661-90 /
# 12.08.2024 / +7 / 1 200. Leading/trailing separators are handled by
# ``.strip`` on the original word; the token we test here is what
# remains after whitespace strip.
_DIGIT_TOKEN_RE_STR = r"^[+\-]?[0-9][0-9.,\-/ ]*[0-9]$|^[0-9]$"


def _is_lexically_valid_rescue(word: str) -> bool:
    """Return True when ``word`` looks like a legitimate OCR read.

    Used to decide whether a borderline-confidence token should be
    rescued. Rescuing ``нe`` / ``Taw`` / ``Fam`` / ``ба`` is exactly
    what we want to AVOID — these are where Tesseract guessed at
    signature / stamp scribble and the low conf is correct. Rescuing
    ``ООО`` / ``7813266190`` / ``ГЕКСАФОРМ`` / ``12.08.2024`` is what
    we want to DO — these are real content that Tesseract underweights
    on short-form tokens.

    Validity = length ≥ 3 AND single-script (all Cyrillic letters, or
    all Latin letters, or a digit/separator token). Single-char tokens
    and mixed-script (``нe``, ``Скаnia``) are never rescued.
    """
    import re

    if not word:
        return False
    w = word.strip()
    if len(w) < 3:
        return False

    # Pure digit token (with optional separators): amounts, ИНН, KPP,
    # dates, phone numbers. Almost always legitimate when Tesseract
    # returns it at 45+ conf — the short-run underweighting problem.
    if re.match(_DIGIT_TOKEN_RE_STR, w):
        return True

    # Letter-only tokens: must be all Cyrillic OR all Latin, no digits,
    # no cross-script mixing. ``re.fullmatch`` so the whole token is
    # same script — a single stray Latin ``a`` in ``Скаnia`` kills
    # the rescue even though the rest is Cyrillic.
    if re.fullmatch(rf"{_CYRILLIC_RE_STR}+", w):
        return True
    return bool(re.fullmatch(rf"{_LATIN_RE_STR}+", w))


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


#: Marker inserted in place of a handwritten-looking block's text.
#: The angle brackets are ``U+27E8`` / ``U+27E9`` (mathematical angle
#: brackets) rather than ASCII ``<>`` or French ``‹›`` — they're
#: uncommon in Russian prose so they won't be produced by a
#: legitimate Tesseract run, which lets downstream consumers
#: unambiguously recognise the marker.
HANDWRITTEN_MARKER: str = "⟨рукописный текст⟩"


def detect_handwritten_blocks(
    data: Mapping[str, Sequence[Any]],
    *,
    max_mean_confidence: float = 40.0,
    min_words: int = 3,
) -> set[tuple[int, int]]:
    """Return ``(block_num, par_num)`` keys for groups whose text looks
    like handwriting.

    Tesseract's Russian LSTM was trained on printed text; handwritten
    regions produce long runs of 10–40 %-confidence noise that the
    word-level filter correctly drops, leaving the user staring at a
    silent gap in the results panel with no indication that there
    was content they need to transcribe manually. The heuristic here
    picks up that signal at the BLOCK level (Tesseract's layout
    analyser clusters contiguous handwritten ink into one block) and
    returns the set of block keys so the caller can surface a marker
    (``⟨рукописный текст⟩``) on the user-facing text.

    Args:
        data: ``image_to_data`` DICT shape.
        max_mean_confidence: A block qualifies when its mean per-word
            confidence is below this value. 40 % is empirically where
            printed-text blocks stop (well above that) and handwritten
            blocks cluster (well below). Tighter thresholds lose
            faded printed text to false positives; looser ones miss
            cleanly-written handwriting.
        min_words: A block needs at least this many valid words to
            be considered. Short 1–2 word fragments are usually
            stamp-overlay noise or the page-header date — leave
            those to the drop_low_conf_words filter rather than
            swallowing them with the marker.

    Returns:
        Set of ``(block_num, par_num)`` tuples to treat as
        handwritten. Empty when no block qualifies.
    """
    texts = list(data.get("text", []))
    confs = list(data.get("conf", []))
    blocks = list(data.get("block_num", []))
    pars = list(data.get("par_num", []))

    # Accumulate per-(block, par) word counts and confidence sums.
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
        soft_rescue: When True, borderline-conf tokens (band
            ``[max(min_confidence - 15, 45), min_confidence)``) that
            pass the lexical-validity check in
            :func:`_is_lexically_valid_rescue` are kept instead of
            dropped. Targets the Tesseract under-confidence behaviour
            on short digit runs (amounts, ИНН, dates) and clean
            all-caps acronyms — rescues real content without opening
            the door to signature / stamp scribble. Default ``False``
            preserves legacy behaviour; ``universal_accurate`` opts in.
            Composes with the always-on CAPS-company preservation in
            :func:`_should_keep_despite_low_conf` — either check
            passing keeps the word.
        handwritten_blocks: Optional set of ``(block_num, par_num)``
            keys to replace with a marker. Use
            :func:`detect_handwritten_blocks` to compute this set.
            Each flagged block contributes ONE marker line to the
            output (not one per line inside the block) so the user
            gets a single unambiguous placeholder per handwritten
            region.
        handwritten_marker: Text to insert for each flagged block.
            Default :data:`HANDWRITTEN_MARKER`.

    Returns:
        Text with one output line per TSV ``(block, par, line)`` group,
        words joined by single spaces in ascending ``word_num`` order.
        Flagged handwritten blocks appear as a single marker line
        instead of their per-line words. Empty when no word passes
        the threshold — callers should treat that as "no replacement,
        keep whatever you already had" to avoid blanking out a result
        just because confidence scoring was unreliable.
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

    # Pre-compute the rescue band once. When ``soft_rescue`` is off the
    # band is unreachable (rescue_min set above the threshold) so the
    # inner loop's cheap ``conf >= rescue_min`` check folds to False
    # without ever invoking the regex validator.
    if soft_rescue:
        rescue_min = max(min_confidence - SOFT_MARGIN, SOFT_FLOOR_ABS)
    else:
        rescue_min = min_confidence + 1.0  # disabled: no word can match

    # Group accepted words by (block, par, line) so we can rebuild with
    # the same paragraph structure Tesseract observed. Using a dict keeps
    # insertion order by group key, then we sort explicitly by the tuple
    # so concurrent blocks / paragraphs serialise top-down reliably.
    grouped: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    rescued_count = 0

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
        if conf < min_confidence:
            # Below threshold — try rescue layers in order of cost.
            # CAPS-company preservation is always on (cheap regex, no
            # flag); soft-rescue is opt-in via the ``soft_rescue``
            # kwarg. Either passing keeps the word; both failing drops
            # it. We short-circuit on the cheaper check first.
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
    # Sort by (block, par, line) so output reflects reading order. Words
    # inside a line keep their word_num order — Tesseract emits them
    # left-to-right, which matches word_num ascending.
    #
    # Flagged handwritten blocks contribute ONE marker line each
    # (the FIRST time we see that (block, par) key); subsequent lines
    # in the same block are skipped so the output doesn't repeat the
    # marker per paragraph line.
    all_keys = set(grouped.keys())
    # Include handwritten blocks even if they had zero words passing
    # the threshold — their marker still has to surface so the user
    # sees there WAS content in that region.
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
