"""Per-word Tesseract re-OCR to disambiguate mixed-script tokens.

The line-level OCR pass picks the wrong script for some glyphs when
a Russian document contains a Latin code (``INV-12345``) or a Latin
document contains a rare Russian token (``Москва-1``). The paragraph-
majority and numeric-context heuristics in
:mod:`src.core.text_postprocessor` fix many of these after the fact,
but they can only work with the text Tesseract handed them — once a
glyph has been mis-classified as Cyrillic ``Н`` instead of Latin ``H``,
the visual information to correct it is gone.

This module recovers the lost information by cropping the word's
pixel bounding box out of the preprocessed page PNG and re-running
Tesseract on that crop twice — once with ``-l rus``, once with
``-l eng``. The higher-confidence result wins. Only fires for words
that :func:`src.core.text_postprocessor._classify_word_script`
returns ``"mixed"`` for; pure-script tokens are never touched.

Design notes:

* **Lazy pytesseract import.** Keeps the module importable in CI
  lint jobs that don't install tesseract.
* **Fail open.** Any exception (missing binary, image decode error,
  timeout, OOM) returns the original word unchanged — we never
  make things worse than the line-level pass.
* **Padding.** The crop is padded by :data:`_BBOX_PADDING_PX` on all
  sides because Tesseract's line-level bounding boxes are
  intentionally tight and a sub-pixel crop loses glyph extremes
  (descenders on "р", ascenders on "b"). Padding gives the LSTM a
  few pixels of margin without changing which glyphs are in frame.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pytesseract  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - CI lint jobs without tesseract deps
    pytesseract = None  # type: ignore[assignment]


#: Pixel padding added around each word's bounding box before the
#: per-word re-OCR crop. 6 px at 300 DPI is ~2 pt — covers ascenders
#: and descenders without picking up adjacent-word ink.
_BBOX_PADDING_PX: int = 6

#: Minimum confidence lift required before we accept a disambiguated
#: result. A tie or a sub-5-point win on what Tesseract already
#: produced is measurement noise; requiring a clear margin prevents
#: churn on tokens the line-level pass already had right.
_MIN_CONFIDENCE_LIFT: float = 5.0


def is_latin_brand_suspect(word: str) -> bool:
    """Return True for tokens that might be a Latin brand mis-read as
    Cyrillic (``TENSAR`` → ``Тапваг``, ``SCANIA`` → garbage).

    The look-alike classifier in :mod:`src.core.text_postprocessor`
    returns ``"cyr"`` (unambiguous Cyrillic) for such tokens because
    Tesseract's LSTM happily renders Latin letters as Cyrillic-
    EXCLUSIVE characters when forced through ``-l rus+eng`` on a
    Russian-dominant page — not a look-alike swap but a fresh
    hallucination biased by the dominant script. The classifier
    can't recover that without image-level re-OCR.

    The heuristic runs against every alphabetic SEGMENT of the
    input (split on non-letter characters) — Tesseract routinely
    glues a CAPS brand to its neighbour with ``/`` or ``,`` ("т.м.
    ТЕМЗАК/скотч," on real transport invoices), and the composite
    token stretches past the length ceiling a naïve whole-word
    check would use. Any CAPS 3-10 letter segment is enough to
    flag the whole token as suspect:

      * 3-10 chars per segment (short enough to be a brand / code
        abbrev; longer alphabetic runs are prose).
      * All uppercase — Russian prose has very few all-caps
        tokens; invoices are full of them (ИНН, КПП, ОГРН,
        brand names, product codes).

    Russian all-caps acronyms (``ИНН``, ``ОГРН`` etc) trigger the
    heuristic too, but that's fine: the disambiguator runs both
    ``-l rus`` and ``-l eng`` re-OCR and Russian wins the
    confidence race for a genuinely-Cyrillic token. Cost is one
    extra Tesseract call per candidate, typically 5-15 tokens per
    page on Russian business documents.
    """
    if not word:
        return False
    # Digit-mixed codes (``INV-5``, ``USD123``) are handled by the
    # numeric-context path in the text postprocessor; skip them here
    # to avoid double-processing.
    if any(ch.isdigit() for ch in word):
        return False
    # Split on non-alphabetic characters so ``ТЕМЗАК/скотч,`` yields
    # ``["ТЕМЗАК", "скотч"]``. Any CAPS 3-10-letter segment flags
    # the whole token as suspect.
    segments: list[str] = []
    buf: list[str] = []
    for ch in word:
        if ch.isalpha():
            buf.append(ch)
        elif buf:
            segments.append("".join(buf))
            buf = []
    if buf:
        segments.append("".join(buf))

    return any(3 <= len(seg) <= 10 and seg.isupper() for seg in segments)


def _run_single_lang_ocr(
    crop: Any, lang: str, tess_config: str,
) -> tuple[str, float] | None:
    """Run ``pytesseract.image_to_data`` on a crop with one language.

    Returns ``(best_word, best_confidence)`` on success or ``None``
    when the crop produced no recognised words.

    "Best" is the highest-confidence recognised word from the crop,
    NOT the mean across all words. Rationale: Tesseract routinely
    splits a bbox-wide crop into a real-word token plus nearby
    junk (punctuation, fragments of adjacent words). Averaging
    drags the measurement down — a ``TENSAR`` correctly recovered
    at 80 % confidence looks like 60 % mean when Tesseract also
    emits a low-conf ``cxory,`` artefact next to it. Picking the
    single best token keeps the comparison apples-to-apples with
    the primary pass's per-word confidence.
    """
    if pytesseract is None:  # pragma: no cover
        return None
    try:
        data = pytesseract.image_to_data(
            crop,
            lang=lang,
            config=tess_config,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "per-word re-OCR failed for lang=%s: %s", lang, exc,
        )
        return None

    best_word: str | None = None
    best_conf: float = -1.0
    for word, conf in zip(
        data.get("text", []), data.get("conf", []), strict=False,
    ):
        if not isinstance(word, str) or not word.strip():
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < 0:
            continue
        if c > best_conf:
            best_conf = c
            best_word = word
    if best_word is None:
        return None
    return best_word, best_conf


def disambiguate_word(
    image: Any,
    bbox: tuple[int, int, int, int],
    original_word: str,
    original_confidence: float,
    tess_config: str = "",
) -> tuple[str, float]:
    """Re-OCR a single word's bounding box under rus + eng and pick the
    higher-confidence result.

    Args:
        image: Full-page image as numpy array (BGR or grayscale).
            The module crops into this directly — never mutates.
        bbox: ``(left, top, width, height)`` of the word in image
            pixel coordinates (Tesseract ``image_to_data`` shape).
        original_word: Token produced by the line-level OCR pass.
            Returned unchanged on any failure or when no alternative
            clears :data:`_MIN_CONFIDENCE_LIFT`.
        original_confidence: The confidence the line-level pass
            assigned to ``original_word`` (0–100 range).
        tess_config: Extra Tesseract config string to forward to both
            re-OCR calls (e.g. ``"--psm 8 --oem 1"``). Empty string
            uses the pytesseract default.

    Returns:
        ``(word, confidence)`` — either the disambiguated pair or the
        original pair when disambiguation didn't clear the lift
        threshold.
    """
    try:
        h, w = image.shape[:2]
    except (AttributeError, IndexError, ValueError):
        return original_word, original_confidence

    left, top, width, height = bbox
    if width <= 0 or height <= 0:
        return original_word, original_confidence

    # Pad and clamp to image bounds. Negative ``left``/``top`` after
    # padding would slice from the wrong end with Python's negative
    # indexing; the ``max(0, ...)`` guards that.
    pad = _BBOX_PADDING_PX
    x0 = max(0, left - pad)
    y0 = max(0, top - pad)
    x1 = min(w, left + width + pad)
    y1 = min(h, top + height + pad)
    if x1 <= x0 or y1 <= y0:
        return original_word, original_confidence

    crop = image[y0:y1, x0:x1]
    # Single-word PSM tells Tesseract to stop looking for layout and
    # treat the crop as one continuous token. OEM passes through
    # whatever the outer caller set.
    config = (tess_config + " --psm 8").strip()

    rus_result = _run_single_lang_ocr(crop, "rus", config)
    eng_result = _run_single_lang_ocr(crop, "eng", config)

    # No single-language pass recovered anything — nothing to do.
    if rus_result is None and eng_result is None:
        return original_word, original_confidence

    # Compare the two SINGLE-LANGUAGE readings against each other, not
    # against the primary pass's confidence. Primary runs with
    # ``-l rus+eng`` and gets to pick the easier interpretation, which
    # inflates its self-reported confidence — we've measured 64 % on
    # a flat-out wrong ``ТЕМЗАК`` reading of ``TENSAR`` while
    # ``-l eng`` alone scores the correct ``TENSAR`` at 61 % and
    # ``-l rus`` on the same crop scores its own ``ТЕМЗАВ`` at 17 %.
    # Comparing to primary there rejects the swap; comparing
    # single-lang vs single-lang picks eng by a clear 44-point margin.
    #
    # The tie-breaker when one direction is absent falls back to the
    # original comparison (single lang vs primary) to keep behaviour
    # for previously-"mixed" tokens unchanged.
    if rus_result is not None and eng_result is not None:
        rus_word, rus_conf = rus_result
        eng_word, eng_conf = eng_result
        if eng_conf >= rus_conf + _MIN_CONFIDENCE_LIFT and eng_word != original_word:
            logger.debug(
                "per-word disambiguation: %r (%.1f) → %r "
                "(eng %.1f vs rus %.1f)",
                original_word, original_confidence,
                eng_word, eng_conf, rus_conf,
            )
            return eng_word, eng_conf
        if rus_conf >= eng_conf + _MIN_CONFIDENCE_LIFT and rus_word != original_word:
            logger.debug(
                "per-word disambiguation: %r (%.1f) → %r "
                "(rus %.1f vs eng %.1f)",
                original_word, original_confidence,
                rus_word, rus_conf, eng_conf,
            )
            return rus_word, rus_conf
        # Neither side beats the other by enough margin — keep primary.
        return original_word, original_confidence

    # Only one language produced a candidate — fall back to the
    # original primary-vs-candidate comparison.
    candidate = rus_result if rus_result is not None else eng_result
    assert candidate is not None  # noqa: S101 — narrowed by the above
    best_word, best_conf = candidate
    if best_conf < original_confidence + _MIN_CONFIDENCE_LIFT:
        return original_word, original_confidence
    return best_word, best_conf


__all__ = ["disambiguate_word", "is_latin_brand_suspect"]
