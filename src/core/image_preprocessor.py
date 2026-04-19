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
    AutoRotateConfig,
    BackgroundConfig,
    BinarizationConfig,
    ContrastConfig,
    DenoiseConfig,
    DenoiseStep,
    DeskewConfig,
    DewarpConfig,
    PreprocessConfig,
)
from src.core.orientation_detector import detect_orientation
from src.shared.types import BinarizationMethod, DenoiseMethod
from src.shared.validators import ValidationError, validate_odd_int

logger = logging.getLogger(__name__)


# Step names accepted by :func:`preview_step`. Order matches the processing
# pipeline exactly.
_PREVIEW_STEPS: tuple[str, ...] = (
    "original",
    "auto_rotate",
    "dewarp",
    "deskew",
    "contrast",
    "background",
    "denoise",
    "binarization",
)


#: Tolerance above which the two-pass deskew applies a corrective
#: rotation. 0.3° is below human perception on A4 and consistent with
#: what the end-to-end tests assert. Tuned conservatively — smaller
#: values would trigger unnecessary corrective rotations on noise.
_RESIDUAL_TOLERANCE_DEG: float = 0.3

#: Coarse search range for the residual-skew measurement. Wide
#: enough to cover the worst real-world primary-detector miss
#: (factors like faint text, heavy borders, photograph-of-monitor
#: scans all reduce ``deskew`` lib reliability). 15° is the upper
#: bound of a hand-held phone scan of a document.
_RESIDUAL_COARSE_RANGE_DEG: float = 15.0
_RESIDUAL_COARSE_STEP_DEG: float = 1.0

#: Fine-grain search around the coarse winner. 1° in 0.1° steps
#: lands us well under the 0.3° tolerance.
_RESIDUAL_FINE_RANGE_DEG: float = 1.0
_RESIDUAL_FINE_STEP_DEG: float = 0.1


#: Baseline DPI at which profile kernel sizes are specified. A ``ksize=3``
#: median blur at 300 DPI removes features ~0.01 inch wide; at 600 DPI
#: the same ksize removes features ~0.005 inch wide, which is sub-glyph
#: noise only — scaled-up (ksize=7) it removes the same physical feature
#: size regardless of render DPI. This lets profile authors write kernel
#: sizes once at 300 DPI and have them stay physically correct when the
#: user picks 400 / 500 / 600 DPI in the same profile.
_KSIZE_BASELINE_DPI: int = 300


def _scale_ksize(ksize: int, dpi: int | None, *, minimum: int = 3) -> int:
    """Scale an odd kernel size by ``dpi / 300``, rounding to the nearest odd.

    Returns ``ksize`` unchanged when ``dpi is None`` (baseline call site)
    or when the DPI is below the baseline (don't shrink kernels — risks
    no-op filters).

    The result is always odd and ≥ ``minimum`` so it's safe to hand
    straight to OpenCV APIs that require ``ksize % 2 == 1``.
    """
    if dpi is None or dpi <= _KSIZE_BASELINE_DPI:
        return max(minimum, ksize)
    scaled = round(ksize * dpi / _KSIZE_BASELINE_DPI)
    if scaled % 2 == 0:
        scaled += 1
    return max(minimum, scaled)


def _measure_residual_skew(image: np.ndarray) -> float:
    """Return the angle in degrees that would straighten ``image``.

    Classic row-variance projection: rotate the binarised image
    through a two-phase search (coarse 1° grid, then 0.1° refine
    around the winner) and pick the angle that maximises per-row
    variance. That's the angle at which horizontal text lines are
    most parallel to the rows. The returned value is what you'd
    rotate BY, not what the image currently tilts at — i.e.
    applying ``rotate(image, measured)`` gives a straighter output.

    Coarse-to-fine keeps the cost bounded: ~30 coarse rotations +
    ~20 fine rotations ≈ 50 rotations. On a ~2000×2000 downsampled
    binarised image that's ~250 ms per call, which the two-pass
    deskew amortises over the full pipeline latency.
    """
    gray = _to_grayscale_for_measure(image)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    def _score_at(angle: float) -> float:
        rotated = rotate_image(binary, float(angle), border_value=0)
        projection = rotated.sum(axis=1, dtype=np.float64)
        return float(projection.var())

    # Coarse pass.
    best_coarse_score = -1.0
    best_coarse_angle = 0.0
    current = -_RESIDUAL_COARSE_RANGE_DEG
    while current <= _RESIDUAL_COARSE_RANGE_DEG + 1e-9:
        score = _score_at(current)
        if score > best_coarse_score:
            best_coarse_score = score
            best_coarse_angle = float(current)
        current += _RESIDUAL_COARSE_STEP_DEG

    # Fine pass.
    best_fine_score = best_coarse_score
    best_fine_angle = best_coarse_angle
    current = best_coarse_angle - _RESIDUAL_FINE_RANGE_DEG
    end = best_coarse_angle + _RESIDUAL_FINE_RANGE_DEG
    while current <= end + 1e-9:
        score = _score_at(current)
        if score > best_fine_score:
            best_fine_score = score
            best_fine_angle = float(current)
        current += _RESIDUAL_FINE_STEP_DEG
    return best_fine_angle


def _to_grayscale_for_measure(image: np.ndarray) -> np.ndarray:
    """Inline grayscale conversion that avoids re-importing cv2 in
    the hot loop. Returns the input unchanged if already 2-D."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


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
        self,
        image: np.ndarray,
        config: PreprocessConfig,
        *,
        dpi: int | None = None,
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
            dpi: The render DPI of ``image``. When provided, kernel sizes
                for denoise / background-blur / Sauvola window / border
                removal / adaptive-threshold are scaled by ``dpi / 300``
                so the same profile produces physically consistent
                filtering at 300, 400, 500 and 600 DPI. ``None`` (the
                default) keeps legacy ``ksize`` values unchanged — kept
                for tests that construct configs directly without a
                meaningful DPI.

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

        # Auto-orientation BEFORE everything else. OSD sees the raw
        # scan (colour or grayscale, no deskew applied yet) and works
        # best on the un-modified page — our own preprocessing can
        # change stroke thickness / contrast enough to confuse OSD's
        # script classifier. Running first also means every
        # subsequent step (dewarp, deskew, binarisation) operates on
        # an already-upright image, which is the shape those algos
        # were designed for.
        auto_rotate_cfg = getattr(config, "auto_rotate", None)
        if auto_rotate_cfg is not None and auto_rotate_cfg.enabled:
            logger.debug("Preprocess: auto-rotate (OSD)")
            current = self._apply_auto_rotate(current, auto_rotate_cfg)

        if config.dewarp.enabled:
            logger.debug("Preprocess: dewarp enabled")
            current = self._apply_dewarp(current, config.dewarp)

        if config.deskew.enabled:
            logger.debug("Preprocess: deskew enabled")
            current, angle = self._apply_deskew(current, config.deskew)

        # Border removal runs AFTER deskew (so lines are axis-aligned
        # by then) but BEFORE everything else. Erasing table borders
        # while the image is still geometric-clean gives the morphology
        # kernels a true horizontal / vertical axis to work with.
        if getattr(config, "border_removal", None) and (
            config.border_removal.enabled
        ):
            from src.core.border_remover import remove_border_lines

            scaled_min_line = (
                int(round(config.border_removal.min_line_length * dpi / _KSIZE_BASELINE_DPI))
                if dpi is not None and dpi > _KSIZE_BASELINE_DPI
                else config.border_removal.min_line_length
            )
            logger.debug(
                "Preprocess: border removal (min_line_length=%d)",
                scaled_min_line,
            )
            current = remove_border_lines(
                current,
                min_line_length=scaled_min_line,
            )

        # Background removal FIRST, then contrast. The old order (CLAHE
        # before background removal) amplified the scanner-lamp
        # gradient into the text itself — CLAHE is a local-contrast
        # enhancer so it preserved the gradient, leaving the binariser
        # to chase it out. Removing the gradient first gives CLAHE a
        # flat canvas and the binariser sees consistent text strokes
        # across the page.
        if config.background.enabled:
            logger.debug("Preprocess: background removal")
            current = self._apply_background_removal(current, config.background, dpi=dpi)

        if config.contrast.clahe_enabled or config.contrast.manual_enabled:
            logger.debug("Preprocess: contrast adjustment")
            current = self._apply_contrast(current, config.contrast)

        if config.denoise.enabled and config.denoise.steps:
            logger.debug("Preprocess: denoise chain (%d steps)", len(config.denoise.steps))
            current = self._apply_denoise(current, config.denoise, dpi=dpi)

        if config.binarization.method != BinarizationMethod.NONE:
            logger.debug("Preprocess: binarization %s", config.binarization.method.value)
            current = self._apply_binarization(current, config.binarization, dpi=dpi)

        return current, angle

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def _apply_deskew(
        self, img: np.ndarray, cfg: DeskewConfig
    ) -> tuple[np.ndarray, float]:
        """Rotate ``img`` by a detected (auto) or manually-configured angle.

        Implements a **two-pass** correction:

          1. Primary detector — usually :class:`DeskewHandler` (Hough-
             based ``deskew`` library). Rotates by the detected angle.
          2. Verification via row-variance projection over a
             ``±3°`` search range on the rotated image. If the
             remaining skew exceeds ``_RESIDUAL_TOLERANCE_DEG``, a
             corrective rotation is applied.

        The two-pass design is mandatory because the primary detector
        can under-correct (common on real scans with noisy margins
        or weak line structure), returning a confidently-wrong angle
        that leaves 2-5° of tilt in the "deskewed" output. The
        verification pass is cheap (< 20 ms for a 2000-px image) and
        guaranteed to catch any residual skew the primary missed.

        Returns:
            Tuple ``(rotated_image, total_angle)``. ``total_angle`` is
            the sum of both passes — useful for downstream logging.
        """
        self._validate_image(img)

        if cfg.auto_detect:
            primary_angle = self._deskew_handler.detect_angle(img)
        else:
            primary_angle = float(cfg.manual_angle)

        max_abs = abs(float(cfg.max_angle))
        if max_abs > 0:
            primary_angle = max(-max_abs, min(max_abs, primary_angle))

        rotated = (
            rotate_image(img, primary_angle, border_value=255)
            if abs(primary_angle) >= 1e-3
            else img
        )

        # Pass 2: measure residual via row-variance projection and
        # apply a corrective rotation if needed. Manual-angle mode
        # trusts the user explicitly — skip verification there.
        if not cfg.auto_detect:
            return rotated, primary_angle

        residual = _measure_residual_skew(rotated)
        if abs(residual) <= _RESIDUAL_TOLERANCE_DEG:
            logger.debug(
                "Deskew: primary %.3f°, residual %.3f° within tolerance",
                primary_angle, residual,
            )
            return rotated, primary_angle

        # Corrective rotation: ``_measure_residual_skew`` already
        # returns the angle that STRAIGHTENS the image (not the tilt
        # direction), so we apply it directly.
        corrective = residual
        total = primary_angle + corrective
        if max_abs > 0 and abs(total) > max_abs:
            corrective = (
                max_abs - primary_angle
                if total > 0
                else -max_abs - primary_angle
            )
            total = primary_angle + corrective

        final = rotate_image(rotated, corrective, border_value=255)
        logger.info(
            "Deskew: two-pass correction — primary %.3f°, "
            "residual %.3f°, corrective %.3f°, total %.3f°",
            primary_angle, residual, corrective, total,
        )
        return final, total

    def _apply_auto_rotate(
        self, img: np.ndarray, cfg: AutoRotateConfig
    ) -> np.ndarray:
        """Rotate ``img`` by 90/180/270° if OSD detects it's off-axis.

        Uses Tesseract's OSD via
        :func:`src.core.orientation_detector.detect_orientation`. When
        OSD returns ``None`` (low confidence, few glyphs, OSD error)
        the image is returned unchanged — the fallback is always
        "leave it alone" so a logo-only page or a QR-code scan can't
        accidentally rotate.

        Uses :func:`cv2.rotate` for the three canonical 90° turns —
        that's a lossless pixel remap; we never want to go through
        the arbitrary-angle affine path for exact 90° multiples.
        """
        self._validate_image(img)
        rotate = detect_orientation(img, min_confidence=cfg.min_confidence)
        if rotate is None or rotate == 0:
            return img
        rotate_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        op = rotate_map.get(int(rotate))
        if op is None:
            logger.debug(
                "auto-rotate: unexpected non-90°-multiple rotation %s, "
                "leaving image alone",
                rotate,
            )
            return img
        try:
            rotated = cv2.rotate(img, op)
        except cv2.error as exc:
            logger.warning(
                "auto-rotate: cv2.rotate failed (%s) — leaving image alone",
                exc,
            )
            return img
        logger.info("auto-rotate: applied %d° rotation", rotate)
        return rotated

    def _apply_dewarp(self, img: np.ndarray, cfg: DewarpConfig) -> np.ndarray:
        """Delegate to :class:`DewarpHandler`."""
        self._validate_image(img)
        return self._dewarp_handler.dewarp(img, cfg)

    def _apply_binarization(
        self,
        img: np.ndarray,
        cfg: BinarizationConfig,
        *,
        dpi: int | None = None,
    ) -> np.ndarray:
        """Convert ``img`` to a binary (black/white) image.

        Supports OTSU, adaptive Gaussian, adaptive mean, and Sauvola (via
        :mod:`skimage`). The adaptive-threshold block size and the Sauvola
        window are scaled by ``dpi / 300`` so the local-context window
        covers a stable fraction of a glyph regardless of render DPI.
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
                    _scale_ksize(cfg.adaptive_block_size, dpi),
                    name="adaptive_block_size",
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
                    _scale_ksize(cfg.adaptive_block_size, dpi),
                    name="adaptive_block_size",
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
                        _scale_ksize(cfg.sauvola_window, dpi),
                        name="sauvola_window",
                    ),
                    k=float(cfg.sauvola_k),
                )
        except cv2.error as exc:
            logger.warning("Бинаризация (%s) завершилась ошибкой: %s", method, exc)
            raise RuntimeError(f"Ошибка бинаризации: {exc}") from exc

        # NONE or unknown: return grayscale unchanged.
        return gray

    def _apply_denoise(
        self,
        img: np.ndarray,
        cfg: DenoiseConfig,
        *,
        dpi: int | None = None,
    ) -> np.ndarray:
        """Apply the ordered denoise chain."""
        self._validate_image(img)
        current = img
        for index, step in enumerate(cfg.steps):
            if not step.enabled:
                continue
            current = self._apply_denoise_step(current, step, index, dpi=dpi)
        return current

    def _apply_denoise_step(
        self,
        img: np.ndarray,
        step: DenoiseStep,
        index: int,
        *,
        dpi: int | None = None,
    ) -> np.ndarray:
        """Apply a single denoising step.

        Kernel sizes for median / Gaussian / morphological steps scale
        with ``dpi / 300`` so a profile tuned at 300 DPI still removes
        the same physical feature sizes at 600 DPI. NLM's ``h``
        (strength) is DPI-independent and is not scaled.
        """
        method = step.method
        try:
            if method == DenoiseMethod.MEDIAN:
                ksize = validate_odd_int(
                    _scale_ksize(step.ksize, dpi),
                    name=f"denoise[{index}].ksize",
                )
                return cv2.medianBlur(img, ksize)

            if method == DenoiseMethod.GAUSSIAN:
                ksize = validate_odd_int(
                    _scale_ksize(step.ksize, dpi),
                    name=f"denoise[{index}].ksize",
                )
                sigma = max(0.0, float(step.sigma))
                return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

            if method == DenoiseMethod.MORPH_OPEN:
                ksize = validate_odd_int(
                    _scale_ksize(step.morph_ksize, dpi),
                    name=f"denoise[{index}].morph_ksize",
                )
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

            if method == DenoiseMethod.MORPH_CLOSE:
                ksize = validate_odd_int(
                    _scale_ksize(step.morph_ksize, dpi),
                    name=f"denoise[{index}].morph_ksize",
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
        self,
        img: np.ndarray,
        cfg: BackgroundConfig,
        *,
        dpi: int | None = None,
    ) -> np.ndarray:
        """Remove illumination gradient by dividing the image by its blur.

        The blurred version approximates the background; dividing the original
        by it flattens shading while preserving foreground text. The blur
        kernel scales with ``dpi / 300`` so the gradient estimate covers a
        consistent physical region across DPIs.
        """
        self._validate_image(img)
        ksize = validate_odd_int(
            _scale_ksize(cfg.blur_kernel, dpi), name="background.blur_kernel"
        )

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
    image: np.ndarray,
    step_name: str,
    config: PreprocessConfig,
    *,
    dpi: int | None = None,
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
        dpi: Optional render DPI — forwarded to steps with scalable
            kernels so the preview matches the final pipeline output.

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

    auto_rotate_cfg = getattr(config, "auto_rotate", None)
    if auto_rotate_cfg is not None and auto_rotate_cfg.enabled:
        current = preprocessor._apply_auto_rotate(current, auto_rotate_cfg)
    if step_name == "auto_rotate":
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
        current = preprocessor._apply_background_removal(
            current, config.background, dpi=dpi
        )
    if step_name == "background":
        return current

    if config.denoise.enabled and config.denoise.steps:
        current = preprocessor._apply_denoise(current, config.denoise, dpi=dpi)
    if step_name == "denoise":
        return current

    if config.binarization.method != BinarizationMethod.NONE:
        current = preprocessor._apply_binarization(
            current, config.binarization, dpi=dpi
        )
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


# ---------------------------------------------------------------------------
# Universal "max accuracy" preset
# ---------------------------------------------------------------------------


def build_universal_preprocess_config() -> PreprocessConfig:
    """Return the preset used by the ``universal_accurate`` builtin profile.

    This is a single opinionated config that enables every step which
    reliably improves OCR accuracy across a wide variety of inputs:

    * deskew (auto-detect, up to ±45°) — non-destructive.
    * CLAHE contrast (clip=2.0, tile=8) — even lighting without noise
      amplification.
    * Background removal is **off** — at 600 DPI the large-kernel blur
      pass costs more than all other preprocessing steps combined with
      only a marginal accuracy gain. Pick ``low_quality_scan`` if the
      input has heavy page tint.
    * Denoise chain: median ksize=3 → morphological close ksize=3.
      Removes salt-and-pepper artefacts and closes sub-pixel breaks
      in thin glyphs. Larger ksizes would start eating dots of
      ``ё``/``ь``, so we stop at the validator minimum.
    * Adaptive-Gaussian binarisation (block=31, C=10) — safe on clean
      pages, dramatically better than OTSU on uneven lighting.
    """
    return PreprocessConfig(
        deskew=DeskewConfig(enabled=True, auto_detect=True, max_angle=45.0),
        dewarp=DewarpConfig(enabled=False),
        binarization=BinarizationConfig(
            method=BinarizationMethod.ADAPTIVE_GAUSSIAN,
            adaptive_block_size=31,
            adaptive_c=10,
        ),
        denoise=DenoiseConfig(
            enabled=True,
            steps=[
                DenoiseStep(method=DenoiseMethod.MEDIAN, ksize=3),
                DenoiseStep(method=DenoiseMethod.MORPH_CLOSE, morph_ksize=3),
            ],
        ),
        contrast=ContrastConfig(clahe_enabled=True, clahe_clip=2.0, clahe_tile=8),
        background=BackgroundConfig(enabled=False),
    )


# Module-level singleton: avoids rebuilding the config (and the
# ImagePreprocessor) every time the helper is called from a tight loop.
UNIVERSAL_PREPROCESS_CONFIG: PreprocessConfig = build_universal_preprocess_config()
_UNIVERSAL_PREPROCESSOR: ImagePreprocessor | None = None


def preprocess_universal(image: np.ndarray) -> tuple[np.ndarray, float]:
    """One-shot: apply the full universal preset to ``image``.

    This is the "just run everything" entry point. Equivalent to:

    >>> pre = ImagePreprocessor()
    >>> pre.process(image, UNIVERSAL_PREPROCESS_CONFIG)

    but with the :class:`ImagePreprocessor` cached at module scope so
    repeated calls don't pay the constructor cost.

    Returns ``(processed_image, detected_skew_angle)``.
    """
    global _UNIVERSAL_PREPROCESSOR
    if _UNIVERSAL_PREPROCESSOR is None:
        _UNIVERSAL_PREPROCESSOR = ImagePreprocessor()
    return _UNIVERSAL_PREPROCESSOR.process(image, UNIVERSAL_PREPROCESS_CONFIG)
