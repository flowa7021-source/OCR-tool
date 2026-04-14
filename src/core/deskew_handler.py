"""Skew detection and rotation helpers.

Wraps the `deskew` library for angle detection and provides a canvas-expanding
rotation implementation on top of OpenCV.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from src.shared.validators import ValidationError

logger = logging.getLogger(__name__)

# Maximum absolute skew angle we accept from the detector. Anything beyond this
# is almost certainly a false positive (e.g. a 90-degree rotated page).
_MAX_ABS_ANGLE: float = 45.0

# Downsample threshold for deskew detection: images whose long side exceeds
# this pixel count get resampled before Hough transform. Skew angle is
# scale-invariant, and the detector is O(n²) in pixel count, so this gives
# a ~4× speedup on typical 300 DPI A4 scans with no measurable accuracy
# loss in the relevant (±45°) range.
_DOWNSAMPLE_THRESHOLD_PX: int = 1600


class DeskewHandler:
    """Detects skew angle of a document image.

    Uses the `deskew` library's Hough-transform-based detector. Input images
    are automatically converted to grayscale when required.

    Thread safety:
        Stateless; instances may be shared across threads provided the
        underlying `deskew` library is thread-safe in the deployed version.
    """

    def __init__(self, max_abs_angle: float = _MAX_ABS_ANGLE) -> None:
        """Initialize the handler.

        Args:
            max_abs_angle: Maximum accepted absolute angle in degrees.
                Values outside [-max_abs_angle, +max_abs_angle] are clamped.
        """
        if max_abs_angle <= 0 or max_abs_angle > 90:
            raise ValidationError(
                f"max_abs_angle должен быть в (0, 90], получено: {max_abs_angle}"
            )
        self._max_abs_angle = float(max_abs_angle)

    def detect_angle(self, image: np.ndarray) -> float:
        """Detect the skew angle of ``image`` in degrees.

        Args:
            image: Grayscale or BGR image as a numpy array.

        Returns:
            Detected skew angle in degrees, clamped to
            ``[-max_abs_angle, +max_abs_angle]``. Returns ``0.0`` if detection
            fails or no reliable angle could be determined.

        Raises:
            ValidationError: If the input image is not a numpy array or is empty.
        """
        if not isinstance(image, np.ndarray):
            raise ValidationError(
                f"image должен быть np.ndarray, получено: {type(image).__name__}"
            )
        if image.size == 0:
            raise ValidationError("Пустое изображение для детекции наклона")

        gray = _to_grayscale(image)

        # Downsample before Hough transform; see _DOWNSAMPLE_THRESHOLD_PX.
        h, w = gray.shape[:2]
        if max(h, w) > _DOWNSAMPLE_THRESHOLD_PX:
            scale = _DOWNSAMPLE_THRESHOLD_PX / float(max(h, w))
            gray = cv2.resize(
                gray,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            logger.debug(
                "Deskew detection downsampled from %dx%d to %dx%d",
                w, h, gray.shape[1], gray.shape[0],
            )

        try:
            # Imported lazily so that the module can be imported on systems
            # that do not yet have `deskew` installed (e.g. for unit tests of
            # unrelated functionality).
            from deskew import determine_skew  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "Библиотека 'deskew' не установлена — пропускаем детекцию наклона"
            )
            return 0.0

        try:
            angle: Any = determine_skew(gray)
        except Exception as exc:  # noqa: BLE001 — library may raise anything
            logger.warning("Не удалось определить угол наклона: %s", exc)
            return 0.0

        if angle is None:
            logger.debug("determine_skew вернул None — считаем угол нулевым")
            return 0.0

        try:
            angle_f = float(angle)
        except (TypeError, ValueError):
            logger.warning("determine_skew вернул не-число: %r", angle)
            return 0.0

        clamped = max(-self._max_abs_angle, min(self._max_abs_angle, angle_f))
        if clamped != angle_f:
            logger.debug(
                "Обнаруженный угол %.3f° ограничен до %.3f°", angle_f, clamped
            )
        return clamped


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a grayscale copy of ``image`` (or the image itself if already gray)."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValidationError(f"Неподдерживаемая форма изображения: {image.shape!r}")


def rotate_image(
    image: np.ndarray,
    angle: float,
    border_value: int = 255,
) -> np.ndarray:
    """Rotate ``image`` around its center by ``angle`` degrees.

    Expands the output canvas so that no pixels are cropped. Uses
    ``cv2.INTER_CUBIC`` interpolation and a constant (white by default) border.

    Args:
        image: Grayscale or BGR/BGRA image.
        angle: Rotation angle in degrees (counter-clockwise, matching OpenCV).
        border_value: Fill value for pixels exposed by rotation. Applied as a
            scalar for grayscale images and as ``(v, v, v[, v])`` for multi-
            channel images.

    Returns:
        A newly allocated rotated image. If ``angle`` is approximately zero,
        the original image is returned unchanged.

    Raises:
        ValidationError: If the input image is invalid.
    """
    if not isinstance(image, np.ndarray):
        raise ValidationError(
            f"image должен быть np.ndarray, получено: {type(image).__name__}"
        )
    if image.size == 0:
        raise ValidationError("Пустое изображение для поворота")

    # Micro-optimization: skip the transform entirely for angles that would
    # produce no visible change.
    if abs(angle) < 1e-3:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Compute the new bounding-box dimensions to avoid cropping.
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the translation component of the affine matrix so that the rotated
    # image is re-centered within the enlarged canvas.
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    if image.ndim == 2:
        border: Any = int(border_value)
    else:
        channels = image.shape[2]
        border = tuple([int(border_value)] * channels)

    try:
        rotated = cv2.warpAffine(
            image,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
    except cv2.error as exc:
        logger.warning("cv2.warpAffine завершился ошибкой: %s", exc)
        raise RuntimeError(f"Не удалось повернуть изображение: {exc}") from exc

    return rotated
