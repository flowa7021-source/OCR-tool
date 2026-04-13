"""Image preprocessing pipeline for OCR.

Orchestrates the per-page image-processing steps: dewarp, deskew, contrast,
background removal, denoising and binarization. Each operation is implemented
as a small, well-scoped private method so that individual steps can also be
exposed to the UI as live previews.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from src.core.deskew_handler import DeskewHandler, rotate_image
from src.core.dewarp_handler import DewarpHandler
from src.core.models import (
    BackgroundConfig,
    BinarizationConfig,
    ContrastConfig,
    DenoiseConfig,
    DenoiseStep,
    DeskewConfig,
    DewarpConfig,
    PreprocessConfig,
)
from src.shared.types import BinarizationMethod, DenoiseMethod
from src.shared.validators import ValidationError, validate_odd_int

logger = logging.getLogger(__name__)


# Step names accepted by :func:`preview_step`. Order matches the processing
# pipeline exactly.
_PREVIEW_STEPS: tuple[str, ...] = (
    "original",
    "dewarp",
    "deskew",
    "contrast",
    "background",
    "denoise",
    "binarization",
)


class ImagePreprocessor:
    """Full preprocessing pipeline applied to a single page image.

    Pipeline order (matches :meth:`process`):
        1. dewarp (cubic-sheet)
        2. deskew (rotation)
        3. contrast (CLAHE + manual)
        4. background removal
        5. denoise chain
        6. binarization

    Thread safety:
        The pipeline itself is stateless, but instances hold references to
        :class:`DeskewHandler` and :class:`DewarpHandler` which are also
        stateless. It is therefore safe to share a single
        :class:`ImagePreprocessor` across worker threads provided the
        underlying OpenCV / page-dewarp / deskew libraries are thread-safe in
        the deployed versions.
    """

    def __init__(
        self,
        deskew_handler: DeskewHandler | None = None,
        dewarp_handler: DewarpHandler | None = None,
    ) -> None:
        """Initialize the preprocessor.

        Args:
            deskew_handler: Custom :class:`DeskewHandler`. A default one is
                created when ``None``.
            dewarp_handler: Custom :class:`DewarpHandler`. A default one is
                created when ``None``.
        """
        self._deskew_handler = deskew_handler or DeskewHandler()
        self._dewarp_handler = dewarp_handler or DewarpHandler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self, image: np.ndarray, config: PreprocessConfig
    ) -> tuple[np.ndarray, float]:
        """Apply the full preprocessing pipeline to ``image``.

        Order:
            1. dewarp (if enabled) — runs before binarization, operates on
               grayscale or colour input.
            2. deskew — rotates the whole image.
            3. contrast (CLAHE + manual adjustment).
            4. background removal.
            5. denoise chain.
            6. binarization — always last.

        Args:
            image: Grayscale or BGR image as a numpy array.
            config: Full preprocessing configuration.

        Returns:
            Tuple ``(processed_image, detected_skew_angle)``. The angle is the
            one actually applied (zero when deskew is disabled, the manual
            angle when auto-detection is off, otherwise the detected value).

        Raises:
            ValidationError: If ``image`` or ``config`` is invalid.
        """
        self._validate_image(image)
        if not isinstance(config, PreprocessConfig):
            raise ValidationError(
                f"config должен быть PreprocessConfig, получено: {type(config).__name__}"
            )

        current = image
        angle = 0.0

        if config.dewarp.enabled:
            logger.debug("Preprocess: dewarp enabled")
            current = self._apply_dewarp(current, config.dewarp)

        if config.deskew.enabled:
            logger.debug("Preprocess: deskew enabled")
            current, angle = self._apply_deskew(current, config.deskew)

        if config.contrast.clahe_enabled or config.contrast.manual_enabled:
            logger.debug("Preprocess: contrast adjustment")
            current = self._apply_contrast(current, config.contrast)

        if config.background.enabled:
            logger.debug("Preprocess: background removal")
            current = self._apply_background_removal(current, config.background)

        if config.denoise.enabled and config.denoise.steps:
            logger.debug("Preprocess: denoise chain (%d steps)", len(config.denoise.steps))
            current = self._apply_denoise(current, config.denoise)

        if config.binarization.method != BinarizationMethod.NONE:
            logger.debug("Preprocess: binarization %s", config.binarization.method.value)
            current = self._apply_binarization(current, config.binarization)

        return current, angle

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def _apply_deskew(
        self, img: np.ndarray, cfg: DeskewConfig
    ) -> tuple[np.ndarray, float]:
        """Rotate ``img`` either by detected or manually-configured angle.

        Returns:
            Tuple ``(rotated_image, applied_angle)``.
        """
        self._validate_image(img)

        if cfg.auto_detect:
            angle = self._deskew_handler.detect_angle(img)
        else:
            angle = float(cfg.manual_angle)

        # Clamp against the configured maximum absolute angle.
        max_abs = abs(float(cfg.max_angle))
        if max_abs > 0:
            angle = max(-max_abs, min(max_abs, angle))

        if abs(angle) < 1e-3:
            logger.debug("Deskew: угол близок к нулю (%.4f), пропускаем поворот", angle)
            return img, angle

        rotated = rotate_image(img, angle, border_value=255)
        logger.debug("Deskew: повёрнуто на %.3f°", angle)
        return rotated, angle

    def _apply_dewarp(self, img: np.ndarray, cfg: DewarpConfig) -> np.ndarray:
        """Delegate to :class:`DewarpHandler`."""
        self._validate_image(img)
        return self._dewarp_handler.dewarp(img, cfg)

    def _apply_binarization(
        self, img: np.ndarray, cfg: BinarizationConfig
    ) -> np.ndarray:
        """Convert ``img`` to a binary (black/white) image.

        Supports OTSU, adaptive Gaussian, adaptive mean, and Sauvola (via
        :mod:`skimage`).
        """
        self._validate_image(img)
        gray = _to_grayscale(img)

        method = cfg.method
        try:
            if method == BinarizationMethod.OTSU:
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                return binary

            if method == BinarizationMethod.ADAPTIVE_GAUSSIAN:
                block = validate_odd_int(
                    cfg.adaptive_block_size, name="adaptive_block_size"
                )
                return cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block,
                    int(cfg.adaptive_c),
                )

            if method == BinarizationMethod.ADAPTIVE_MEAN:
                block = validate_odd_int(
                    cfg.adaptive_block_size, name="adaptive_block_size"
                )
                return cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    block,
                    int(cfg.adaptive_c),
                )

            if method == BinarizationMethod.SAUVOLA:
                return _sauvola_binarize(
                    gray,
                    window_size=validate_odd_int(
                        cfg.sauvola_window, name="sauvola_window"
                    ),
                    k=float(cfg.sauvola_k),
                )
        except cv2.error as exc:
            logger.warning("Бинаризация (%s) завершилась ошибкой: %s", method, exc)
            raise RuntimeError(f"Ошибка бинаризации: {exc}") from exc

        # NONE or unknown: return grayscale unchanged.
        return gray

    def _apply_denoise(self, img: np.ndarray, cfg: DenoiseConfig) -> np.ndarray:
        """Apply the ordered denoise chain."""
        self._validate_image(img)
        current = img
        for index, step in enumerate(cfg.steps):
            if not step.enabled:
                continue
            current = self._apply_denoise_step(current, step, index)
        return current

    def _apply_denoise_step(
        self, img: np.ndarray, step: DenoiseStep, index: int
    ) -> np.ndarray:
        """Apply a single denoising step."""
        method = step.method
        try:
            if method == DenoiseMethod.MEDIAN:
                ksize = validate_odd_int(step.ksize, name=f"denoise[{index}].ksize")
                return cv2.medianBlur(img, ksize)

            if method == DenoiseMethod.GAUSSIAN:
                ksize = validate_odd_int(step.ksize, name=f"denoise[{index}].ksize")
                sigma = max(0.0, float(step.sigma))
                return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

            if method == DenoiseMethod.MORPH_OPEN:
                ksize = validate_odd_int(
                    step.morph_ksize, name=f"denoise[{index}].morph_ksize"
                )
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            if method == DenoiseMethod.MORPH_CLOSE:
                ksize = validate_odd_int(
                    step.morph_ksize, name=f"denoise[{index}].morph_ksize"
                )
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            if method == DenoiseMethod.NLM:
                h = max(1, int(step.h))
                if img.ndim == 2:
                    return cv2.fastNlMeansDenoising(img, None, float(h), 7, 21)
                return cv2.fastNlMeansDenoisingColored(
                    img, None, float(h), float(h), 7, 21
                )
        except cv2.error as exc:
            logger.warning(
                "Денойз-шаг %d (%s) завершился ошибкой: %s — пропускаем",
                index,
                method,
                exc,
            )
            return img

        logger.warning("Неизвестный метод денойза: %r — пропускаем", method)
        return img

    def _apply_contrast(self, img: np.ndarray, cfg: ContrastConfig) -> np.ndarray:
        """Enhance contrast via CLAHE and/or a linear transform."""
        self._validate_image(img)
        current = img

        if cfg.clahe_enabled:
            clip = max(0.1, float(cfg.clahe_clip))
            tile = max(1, int(cfg.clahe_tile))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            try:
                if current.ndim == 2:
                    current = clahe.apply(current)
                else:
                    # Apply CLAHE on the luminance channel only to preserve colour.
                    lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB)
                    l_channel, a_channel, b_channel = cv2.split(lab)
                    l_channel = clahe.apply(l_channel)
                    lab = cv2.merge((l_channel, a_channel, b_channel))
                    current = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            except cv2.error as exc:
                logger.warning("CLAHE завершился ошибкой: %s", exc)

        if cfg.manual_enabled:
            alpha = float(cfg.alpha)
            beta = int(cfg.beta)
            try:
                current = cv2.convertScaleAbs(current, alpha=alpha, beta=beta)
            except cv2.error as exc:
                logger.warning("Manual contrast завершился ошибкой: %s", exc)

        return current

    def _apply_background_removal(
        self, img: np.ndarray, cfg: BackgroundConfig
    ) -> np.ndarray:
        """Remove illumination gradient by dividing the image by its blur.

        The blurred version approximates the background; dividing the original
        by it flattens shading while preserving foreground text.
        """
        self._validate_image(img)
        ksize = validate_odd_int(cfg.blur_kernel, name="background.blur_kernel")

        try:
            blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
            # Avoid division by zero.
            blurred_safe = np.where(blurred == 0, 1, blurred).astype(np.float32)
            divided = img.astype(np.float32) / blurred_safe
            normalized = cv2.normalize(divided, None, 0, 255, cv2.NORM_MINMAX)
            return normalized.astype(np.uint8)
        except cv2.error as exc:
            logger.warning("Background removal завершился ошибкой: %s", exc)
            return img

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image(image: Any) -> None:
        """Validate that ``image`` is a non-empty numpy array."""
        if not isinstance(image, np.ndarray):
            raise ValidationError(
                f"image должен быть np.ndarray, получено: {type(image).__name__}"
            )
        if image.size == 0:
            raise ValidationError("Пустое изображение")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def preview_step(
    image: np.ndarray, step_name: str, config: PreprocessConfig
) -> np.ndarray:
    """Return the intermediate result up to ``step_name`` (inclusive).

    The preview is produced by running the pipeline in order and stopping
    right after the requested step. This lets the UI show "what does the
    image look like after deskew?" while leaving later stages untouched.

    Args:
        image: Source image.
        step_name: One of ``"original"``, ``"dewarp"``, ``"deskew"``,
            ``"contrast"``, ``"background"``, ``"denoise"``, ``"binarization"``.
        config: Full preprocessing configuration.

    Returns:
        Image array representing the pipeline's state after ``step_name``.

    Raises:
        ValidationError: If ``step_name`` is unknown or inputs are invalid.
    """
    if step_name not in _PREVIEW_STEPS:
        raise ValidationError(
            f"Неизвестный шаг предпросмотра: {step_name!r}. "
            f"Допустимые: {list(_PREVIEW_STEPS)}"
        )

    preprocessor = ImagePreprocessor()
    preprocessor._validate_image(image)

    current = image
    if step_name == "original":
        return current

    if config.dewarp.enabled:
        current = preprocessor._apply_dewarp(current, config.dewarp)
    if step_name == "dewarp":
        return current

    if config.deskew.enabled:
        current, _ = preprocessor._apply_deskew(current, config.deskew)
    if step_name == "deskew":
        return current

    if config.contrast.clahe_enabled or config.contrast.manual_enabled:
        current = preprocessor._apply_contrast(current, config.contrast)
    if step_name == "contrast":
        return current

    if config.background.enabled:
        current = preprocessor._apply_background_removal(current, config.background)
    if step_name == "background":
        return current

    if config.denoise.enabled and config.denoise.steps:
        current = preprocessor._apply_denoise(current, config.denoise)
    if step_name == "denoise":
        return current

    if config.binarization.method != BinarizationMethod.NONE:
        current = preprocessor._apply_binarization(current, config.binarization)
    return current


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a grayscale view of ``image``."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValidationError(f"Неподдерживаемая форма изображения: {image.shape!r}")


def _sauvola_binarize(gray: np.ndarray, window_size: int, k: float) -> np.ndarray:
    """Apply Sauvola local thresholding via :mod:`skimage`.

    Falls back to Otsu when :mod:`skimage` is unavailable so that the
    pipeline keeps functioning on a minimally-provisioned environment.
    """
    try:
        from skimage.filters import threshold_sauvola  # type: ignore[import-not-found]
    except ImportError:
        logger.warning(
            "scikit-image не установлен — откат бинаризации на OTSU вместо Sauvola"
        )
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    try:
        thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    except Exception as exc:  # noqa: BLE001 — skimage can raise many things
        logger.warning("threshold_sauvola завершился ошибкой: %s", exc)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    binary = (gray > thresh).astype(np.uint8) * 255
    return binary
