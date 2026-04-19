"""Pre-OCR removal of long horizontal and vertical lines.

Tesseract occasionally mis-reads table borders / signature rules /
form underlines as letter strokes, fusing nearby text into the
border glyph or emitting mysterious ``|`` runs next to legitimate
content. Removing those lines from the input image BEFORE handing
it to Tesseract gives the LSTM a clean letter-on-white canvas.

Algorithm: morphological open with a long horizontal structuring
element isolates horizontal runs; same with a vertical element
isolates vertical runs. The union of both runs is subtracted from
the original image (filled with background white). Diagonal strokes
— letter components like ``/``, ``A``, ``K``, Cyrillic ``К``, ``Ж``
— are NOT captured by either structuring element, so they survive.

Why not Hough-lines: Hough returns line equations we'd still have
to rasterise and subtract. Morphology does the same in one step
and scales to millions of pixels without choosing ``rho`` and
``theta`` thresholds we'd have to retune per DPI.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# Minimum line length (in pixels) for a run to count as a "border"
# rather than a glyph stroke. At 300 DPI a capital letter is
# ~40 px tall, so 50 px is a safe floor — letter serifs and strokes
# stay below that. At 500 DPI you may want to raise it; callers
# pass ``min_line_length=`` to override.
MIN_LINE_LENGTH_PX: int = 50


def _line_mask(gray: np.ndarray, *, horizontal: bool, length: int) -> np.ndarray:
    """Return a binary mask of all pixels belonging to long runs.

    Morphological open with a ``1×length`` (horizontal) or
    ``length×1`` (vertical) structuring element keeps only runs at
    least ``length`` pixels long in the given direction. Everything
    shorter — including diagonal glyph strokes and body text —
    vanishes from the mask.
    """
    import cv2

    # Threshold to binary (text is dark, background is light). Using
    # Otsu here would be sensitive to noise; a fixed threshold at 128
    # is robust for the clean preprocessed images we receive.
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    if horizontal:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def remove_border_lines(
    image: np.ndarray,
    *,
    min_line_length: int = MIN_LINE_LENGTH_PX,
) -> np.ndarray:
    """Erase long horizontal and vertical lines from ``image``.

    Args:
        image: Grayscale (H×W) or BGR (H×W×3) numpy array. Shape
            and dtype are preserved.
        min_line_length: Minimum run length in pixels to erase.
            Shorter runs (letter serifs, decorative dashes) are
            left alone. Scale with DPI — 50 px is fine for 300 DPI,
            bump to 75 px for 500 DPI.

    Returns:
        A copy of ``image`` with long horizontal + vertical runs
        replaced by background colour (255 for grayscale, white for
        BGR). No colour-space conversion, no noise introduction.
    """
    if image.ndim == 3:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Union of horizontal + vertical long-run masks.
    hor_mask = _line_mask(gray, horizontal=True, length=min_line_length)
    ver_mask = _line_mask(gray, horizontal=False, length=min_line_length)
    line_mask = hor_mask | ver_mask

    if not np.any(line_mask):
        # No borders detected — return input untouched. Exact array
        # equality matters for the "empty page passes through" test.
        return image

    cleaned = image.copy()
    if cleaned.ndim == 3:
        cleaned[line_mask > 0] = [255, 255, 255]
    else:
        cleaned[line_mask > 0] = 255

    covered = int(np.count_nonzero(line_mask))
    logger.debug(
        "border_remover: erased %d pixel(s) on long runs (min_length=%d)",
        covered, min_line_length,
    )
    return cleaned


__all__ = ["MIN_LINE_LENGTH_PX", "remove_border_lines"]
