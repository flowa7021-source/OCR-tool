"""Super-resolution for low-DPI scans.

When a PDF's embedded image is under ~150 effective DPI, OCR accuracy
drops sharply: EasyOCR's CRNN was trained on 32-64 px tall glyphs, and
a 100-DPI scan of 10-point text gives it 14-16 px glyphs — too few
pixels for the recogniser to distinguish detail.

This module provides an engine-agnostic upscaling step you can apply
before OCR. Two modes:

- ``"bicubic_sharpen"`` (default, no dependencies): bicubic interpolation
  + unsharp mask. Fast (≤ 30 ms / A4 page), predictable, no model
  downloads. Gains ~5-10 pp OCR accuracy on 100-DPI scans.

- ``"dnn_edsr"`` / ``"dnn_fsrcnn"`` (optional): OpenCV's DNN super-
  resolution using downloaded ONNX models. Higher quality, ~300 ms /
  page on CPU. Gracefully falls back to bicubic_sharpen if the model
  file is not present.

Integrates with the existing DPI-advisory logic in pipeline.py —
see ``_estimate_page_source_dpi`` for source DPI detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Target effective DPI for OCR. Stock EasyOCR CRNN performs best around 300 DPI.
TARGET_DPI = 300

# Threshold below which upscaling is worth the cost. Above this, bicubic
# upscale doesn't measurably help and just burns CPU.
LOW_DPI_THRESHOLD = 200

# Where DNN model files are expected. Optional — module degrades
# gracefully if these don't exist.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "resources" / "sr_models"


@dataclass
class SRStats:
    """Observability for the upscale step."""
    applied: bool
    mode: str
    input_dpi: int
    output_dpi: int
    scale: float
    elapsed_ms: float


def _bicubic_sharpen(
    image: np.ndarray, scale: float, unsharp_amount: float = 0.5,
) -> np.ndarray:
    """Bicubic upscale + unsharp mask. Pure OpenCV, no model downloads."""
    h, w = image.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    upscaled = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    # Unsharp mask: image + amount × (image - blur(image))
    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharpened = cv2.addWeighted(upscaled, 1.0 + unsharp_amount,
                                blurred, -unsharp_amount, 0)
    return sharpened


def _dnn_upscale(
    image: np.ndarray, model_path: Path, model_name: str, scale: int,
) -> np.ndarray:
    """OpenCV DNN super-resolution. Used when the .pb model file exists.

    Supported models: ``edsr``, ``espcn``, ``fsrcnn``, ``lapsrn`` —
    all trainable at scales 2/3/4. Download .pb files to
    ``resources/sr_models/`` to enable.
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(model_name, scale)
    # DNN expects 3-channel input; replicate grayscale to BGR.
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        up = sr.upsample(bgr)
        return cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    return sr.upsample(image)


def _resolve_model(
    mode: str, scale: int, model_dir: Path = DEFAULT_MODEL_DIR,
) -> tuple[Path, str] | None:
    """Locate the ONNX/.pb file for the requested DNN mode.

    Returns (path, model_name) or None if the file isn't present.
    """
    if not mode.startswith("dnn_"):
        return None
    family = mode.split("_", 1)[1].upper()  # EDSR, FSRCNN, ESPCN, LAPSRN
    filename = f"{family}_x{scale}.pb"
    p = model_dir / filename
    if not p.exists():
        logger.warning(
            "SR model %s not found at %s — falling back to bicubic_sharpen",
            filename, p,
        )
        return None
    return p, family.lower()


def upscale_for_ocr(
    image: np.ndarray,
    source_dpi: int | None = None,
    target_dpi: int = TARGET_DPI,
    mode: str = "bicubic_sharpen",
    force: bool = False,
    model_dir: Path = DEFAULT_MODEL_DIR,
) -> tuple[np.ndarray, SRStats]:
    """Upscale ``image`` to bring effective DPI up to ``target_dpi``.

    Args:
        image: grayscale (H, W) or BGR (H, W, 3) uint8 array.
        source_dpi: estimated source DPI. If ``None``, upscaling is skipped
            unless ``force=True`` — we can't compute an appropriate scale
            factor without knowing the starting resolution.
        target_dpi: desired effective DPI. Default 300 matches EasyOCR's
            sweet spot.
        mode: ``"bicubic_sharpen"`` (no-deps) or ``"dnn_edsr"`` /
            ``"dnn_fsrcnn"`` / ``"dnn_espcn"`` / ``"dnn_lapsrn"``. DNN
            modes fall back to bicubic_sharpen if the model file is
            missing.
        force: apply upscale even when ``source_dpi`` is None or
            already ≥ ``LOW_DPI_THRESHOLD``.

    Returns:
        (upscaled_image, stats). If no upscale is applied, the input
        array is returned unchanged with ``stats.applied=False``.
    """
    import time
    t0 = time.perf_counter()
    if source_dpi is None and not force:
        return image, SRStats(False, mode, 0, 0, 1.0, 0.0)
    if source_dpi is not None and source_dpi >= LOW_DPI_THRESHOLD and not force:
        return image, SRStats(False, mode, source_dpi, source_dpi, 1.0,
                              (time.perf_counter() - t0) * 1000)

    # Decide scale factor. Round to whole number because DNN SR models
    # are trained only for integer scales.
    scale_f = float(target_dpi) / float(source_dpi or LOW_DPI_THRESHOLD / 2)
    scale_int = max(2, min(4, round(scale_f)))

    if mode == "bicubic_sharpen":
        out = _bicubic_sharpen(image, scale_f)
    elif mode.startswith("dnn_"):
        resolved = _resolve_model(mode, scale_int, model_dir=model_dir)
        if resolved is None:
            out = _bicubic_sharpen(image, scale_f)
            mode = "bicubic_sharpen (fallback)"
        else:
            model_path, model_name = resolved
            try:
                out = _dnn_upscale(image, model_path, model_name, scale_int)
            except cv2.error as e:  # pragma: no cover — OpenCV runtime issue
                logger.warning("DNN SR failed (%s), falling back to bicubic", e)
                out = _bicubic_sharpen(image, scale_f)
                mode = "bicubic_sharpen (dnn_error_fallback)"
    else:
        raise ValueError(f"Unknown super-resolution mode: {mode!r}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    effective_dpi = int((source_dpi or LOW_DPI_THRESHOLD // 2) * scale_f)
    return out, SRStats(
        applied=True,
        mode=mode,
        input_dpi=source_dpi or 0,
        output_dpi=effective_dpi,
        scale=scale_f,
        elapsed_ms=elapsed_ms,
    )


__all__ = ["upscale_for_ocr", "SRStats", "LOW_DPI_THRESHOLD", "TARGET_DPI"]
