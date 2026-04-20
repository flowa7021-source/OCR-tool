"""Per-word image-enhancement rescue: re-OCR low-confidence words with
aggressive local preprocessing applied to just that word's crop.

Companion to :mod:`src.core.per_word_script_disambiguator`. Where the
disambiguator re-OCRs mixed-script tokens to pick the right language,
this module re-OCRs LOW-confidence tokens by boosting the pixel data
before feeding it back to Tesseract — targeting the "faded ink, small
ticker-tape, stamp-overlay" failure mode where the line-level pass
produced plausible-but-uncertain output.

Two enhancement strategies, applied in sequence:

  * **CLAHE + unsharp mask** (``clahe_sharpen_rescue``) — aggressive
    local contrast (clip=8.0) plus a Gaussian-unsharp pass brings
    thin strokes back from the noise floor. Cheap: ~5 ms per crop.
    Targets faded-ink and low-contrast regions.
  * **2× bicubic upscaling** (``upscale_rescue``) — doubles the
    pixel dimensions with ``cv2.INTER_CUBIC`` and re-runs OCR.
    Tesseract's LSTM was trained on 20–40 px glyph height; tiny
    ticker-tape text at 10–15 px upscales into the sweet spot.
    Costs one extra Tesseract call on a 4× pixel crop, typically
    ~50 ms per candidate.

Both rescues only fire when the candidate word meets conservative
gates — confidence in the 30–70 band (below 30 is too noisy to
rescue; above 70 is already good enough), length ≥ 2 characters,
clean alphanumeric content. The re-OCR result replaces the original
only when it clears a clear confidence lift, so a failed rescue
never degrades output.

Fails open on any exception (missing pytesseract, OpenCV error,
tessdata missing) — rescue is additive, the original result stands
when we can't improve it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - cv2 is a real dep
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - numpy is a real dep
    np = None  # type: ignore[assignment]

try:
    import pytesseract  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - CI lint jobs without tesseract deps
    pytesseract = None  # type: ignore[assignment]


#: Pixel padding added around each word's bounding box before the
#: rescue crop. Same rationale as the disambiguator's padding —
#: covers ascenders/descenders without picking up adjacent ink.
_BBOX_PADDING_PX: int = 6

#: Confidence band in which the rescues fire. Below the low bound
#: the image is too damaged for a re-OCR to recover; above the high
#: bound the result is already good enough that churning it risks
#: downgrading. Empirically the "faded ink" regime sits in this
#: band: clean text scores > 80, pure noise < 30.
_RESCUE_MIN_ORIGINAL_CONF: float = 30.0
_RESCUE_MAX_ORIGINAL_CONF: float = 70.0

#: Minimum confidence lift required before we accept a rescue. The
#: re-OCR must CLEARLY beat the line-level pass; a tie or a
#: sub-10-point win is measurement noise in the faded-ink regime.
_MIN_CONFIDENCE_LIFT: float = 10.0


def _extract_crop(
    image: Any, bbox: tuple[int, int, int, int],
) -> Any | None:
    """Crop ``bbox`` out of ``image`` with :data:`_BBOX_PADDING_PX`.

    Returns ``None`` when the image / bbox is unusable so callers can
    fail open without adding special-case logic.
    """
    if image is None or cv2 is None or np is None:
        return None
    try:
        h, w = image.shape[:2]
    except (AttributeError, IndexError, ValueError):
        return None
    left, top, width, height = bbox
    if width <= 0 or height <= 0:
        return None
    pad = _BBOX_PADDING_PX
    x0 = max(0, left - pad)
    y0 = max(0, top - pad)
    x1 = min(w, left + width + pad)
    y1 = min(h, top + height + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def _run_word_ocr(
    crop: Any, lang: str, tess_config: str,
) -> tuple[str, float] | None:
    """Single-word Tesseract pass on ``crop``; return best word + conf.

    Mirrors ``_run_single_lang_ocr`` in the disambiguator — picks the
    HIGHEST-confidence word from the crop (not the mean) so adjacent-
    ink artefacts in the padding don't drag down the measurement of
    the token we actually care about.
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
        logger.debug("per-word rescue OCR failed (%s): %s", lang, exc)
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


def _is_rescuable(word: str, confidence: float) -> bool:
    """Gate keeper — which (word, conf) pairs are worth rescue-trying.

    Confidence must fall in the 30–70 band; below 30 the crop is
    noise no enhancement recovers, above 70 the original already
    reads correctly and a rescue risks churning it. Word must have
    ≥ 2 characters of alphanumeric content so we don't spend an OCR
    call on punctuation fragments.
    """
    if confidence < _RESCUE_MIN_ORIGINAL_CONF:
        return False
    if confidence > _RESCUE_MAX_ORIGINAL_CONF:
        return False
    stripped = word.strip()
    if len(stripped) < 2:
        return False
    return any(ch.isalnum() for ch in stripped)


def clahe_sharpen_rescue(
    image: Any,
    bbox: tuple[int, int, int, int],
    original_word: str,
    original_confidence: float,
    lang: str,
    tess_config: str = "",
) -> tuple[str, float]:
    """CLAHE + unsharp-mask re-OCR for a borderline-confidence word.

    Applied to the bbox crop on a grayscale (or grayscale-converted)
    ``image``:

      1. ``cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))`` —
         aggressive local contrast. The universal preset uses
         ``clipLimit=2.0`` as a "safe" global setting, but inside
         a tight word crop the tonal range is small enough that
         an 8.0 boost doesn't over-amplify noise.
      2. Gaussian unsharp mask (``cv2.GaussianBlur`` + weighted
         subtraction, σ=1.2) sharpens the thin strokes the CLAHE
         surface-boost revealed.

    Then re-OCR with ``--psm 8`` (single word) on the enhanced crop.
    If the new reading clears ``original_confidence + 10``, return
    it; otherwise keep the original untouched.

    Args:
        image: Full-page image (2-D grayscale or 3-D BGR/RGB). The
            rescue works best on the pre-binarisation grayscale
            snapshot the pipeline persists when
            ``per_word_script_disambiguation`` is on — re-applying
            CLAHE to a pre-OTSU binary image is useless.
        bbox: Word bounding box in ``image`` pixel coordinates.
        original_word: Token from the line-level OCR pass.
        original_confidence: Line-level confidence (0–100).
        lang: Tesseract language string (``"rus+eng"``, ``"rus"``,
            etc.) — matches the primary OCR pass.
        tess_config: Extra Tesseract config forwarded to the rescue
            call (e.g. ``"--oem 1"``). ``--psm 8`` is added
            internally.

    Returns:
        ``(word, confidence)`` — the swapped pair on success, or
        the original pair when the rescue failed or didn't clear
        the lift threshold.
    """
    if not _is_rescuable(original_word, original_confidence):
        return original_word, original_confidence
    if cv2 is None or np is None:
        return original_word, original_confidence
    crop = _extract_crop(image, bbox)
    if crop is None:
        return original_word, original_confidence

    # Work on grayscale — CLAHE's LAB-channel path is overkill for a
    # word crop and unsharp-mask operates cleanly on single-channel
    # input. _to_grayscale converts as needed.
    try:
        gray = (
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if crop.ndim == 3 else crop
        )
    except cv2.error as exc:
        logger.debug("CLAHE rescue: cvtColor failed: %s", exc)
        return original_word, original_confidence

    try:
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        boosted = clahe.apply(gray)
        blurred = cv2.GaussianBlur(boosted, (0, 0), sigmaX=1.2, sigmaY=1.2)
        sharpened = cv2.addWeighted(boosted, 1.5, blurred, -0.5, 0)
    except cv2.error as exc:
        logger.debug("CLAHE rescue: enhancement failed: %s", exc)
        return original_word, original_confidence

    config = (tess_config + " --psm 8").strip()
    result = _run_word_ocr(sharpened, lang, config)
    if result is None:
        return original_word, original_confidence
    new_word, new_conf = result
    if new_conf < original_confidence + _MIN_CONFIDENCE_LIFT:
        return original_word, original_confidence
    if new_word == original_word:
        # Same text, higher conf — keep original text but bump conf
        # so the downstream filter doesn't drop it. Half the rescue's
        # job is rescuing TEXT; the other half is rescuing a word from
        # the drop-threshold cliff.
        return original_word, new_conf
    logger.debug(
        "CLAHE rescue: %r (%.1f) → %r (%.1f)",
        original_word, original_confidence, new_word, new_conf,
    )
    return new_word, new_conf


def upscale_rescue(
    image: Any,
    bbox: tuple[int, int, int, int],
    original_word: str,
    original_confidence: float,
    lang: str,
    tess_config: str = "",
    *,
    scale: int = 2,
) -> tuple[str, float]:
    """2× bicubic upscale + re-OCR for tiny-glyph words.

    Tesseract's LSTM was trained on ~20–40 px glyph heights; small-
    ticker-tape text at 10–15 px scores 40–60 % confidence because
    the features the LSTM expects fall below the grid.
    ``cv2.INTER_CUBIC`` upscale to 2× puts the glyphs back in the
    trained regime without requiring a DNN super-resolution model
    bundle.

    Same return-on-success / return-original-on-failure contract as
    :func:`clahe_sharpen_rescue`. The two rescues compose cleanly —
    the pipeline runs CLAHE first (cheap, helps on faded ink),
    falls through to upscale only when CLAHE didn't improve conf.

    Args:
        scale: Upsample factor; 2 is the sweet spot (doubles glyph
            height from 12–15 px into the 24–30 px LSTM sweet spot).
            3× helps marginally more but doubles the OCR call cost.
    """
    if not _is_rescuable(original_word, original_confidence):
        return original_word, original_confidence
    if cv2 is None or np is None:
        return original_word, original_confidence
    crop = _extract_crop(image, bbox)
    if crop is None:
        return original_word, original_confidence

    try:
        upscaled = cv2.resize(
            crop,
            None,
            fx=float(scale),
            fy=float(scale),
            interpolation=cv2.INTER_CUBIC,
        )
    except cv2.error as exc:
        logger.debug("upscale rescue: resize failed: %s", exc)
        return original_word, original_confidence

    config = (tess_config + " --psm 8").strip()
    result = _run_word_ocr(upscaled, lang, config)
    if result is None:
        return original_word, original_confidence
    new_word, new_conf = result
    if new_conf < original_confidence + _MIN_CONFIDENCE_LIFT:
        return original_word, original_confidence
    if new_word == original_word:
        return original_word, new_conf
    logger.debug(
        "upscale rescue: %r (%.1f) → %r (%.1f)",
        original_word, original_confidence, new_word, new_conf,
    )
    return new_word, new_conf


__all__ = [
    "clahe_sharpen_rescue",
    "upscale_rescue",
]
