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

    This heuristic picks up the class of suspicious tokens that
    SHOULD be re-OCR'd despite not being "mixed":

      * 3-10 chars (short enough to plausibly be a brand abbrev
        or acronym; longer tokens are prose).
      * All uppercase — Russian prose has very few all-caps
        tokens; invoices / catalogues are full of them (ИНН, КПП,
        ОГРН, brand names, product codes).
      * Alphabetic only (no digits / punctuation) — digit-mixed
        codes are caught by the numeric-context path in the
        postprocessor.

    Russian all-caps acronyms (``ИНН``, ``ОГРН`` etc) trigger this
    heuristic too, but that's fine: the disambiguator will run
    both ``-l rus`` and ``-l eng`` re-OCR and Russian wins the
    confidence race for a genuinely-Cyrillic token. Cost is one
    extra Tesseract call per candidate, which is ~5-15 tokens per
    page on typical Russian business docs.
    """
    if not (3 <= len(word) <= 10):
        return False
    if not word.isupper():
        return False
    return all(ch.isalpha() for ch in word)


def _run_single_lang_ocr(
    crop: Any, lang: str, tess_config: str,
) -> tuple[str, float] | None:
    """Run ``pytesseract.image_to_data`` on a crop with one language.

    Returns ``(joined_text, mean_confidence)`` on success or ``None``
    when the crop produced no recognised words — the latter is the
    signal that this language simply isn't the one in the image.
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

    words: list[str] = []
    confs: list[float] = []
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
        words.append(word)
        confs.append(c)
    if not words:
        return None
    return " ".join(words), sum(confs) / len(confs)


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

    candidates: list[tuple[str, float, str]] = []  # (word, conf, lang)
    if rus_result is not None:
        candidates.append((rus_result[0], rus_result[1], "rus"))
    if eng_result is not None:
        candidates.append((eng_result[0], eng_result[1], "eng"))

    if not candidates:
        return original_word, original_confidence

    best_word, best_conf, best_lang = max(candidates, key=lambda c: c[1])

    # Require a clear lift over the original. The alternative we
    # picked has to be meaningfully more confident than what the
    # line-level pass emitted; otherwise we might swap in a re-OCR
    # artefact (whitespace quirks, joined punctuation).
    if best_conf < original_confidence + _MIN_CONFIDENCE_LIFT:
        return original_word, original_confidence

    logger.debug(
        "per-word disambiguation: %r (%.1f) → %r (%.1f) via -l %s",
        original_word, original_confidence,
        best_word, best_conf, best_lang,
    )
    return best_word, best_conf


__all__ = ["disambiguate_word", "is_latin_brand_suspect"]
