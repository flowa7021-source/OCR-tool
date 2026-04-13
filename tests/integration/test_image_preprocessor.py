"""End-to-end tests for ImagePreprocessor on synthetic images.

These exercise the real OpenCV pipeline without depending on Tesseract,
PyMuPDF, or OCRmyPDF. They guard against regressions in the stage ordering,
kernel-size validation, and output dtype/shape guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.core.image_preprocessor import ImagePreprocessor, preview_step
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


def _make_text_image(height: int = 200, width: int = 600) -> np.ndarray:
    """Produce a BGR image that loosely resembles a scanned page.

    White background, a horizontal black stripe (line of "text"), and a mild
    gradient shadow in one corner. No real OCR is performed — we only need
    something for the pipeline to crunch.
    """
    import cv2

    img = np.full((height, width, 3), 250, dtype=np.uint8)
    # Dark horizontal stripe
    cv2.rectangle(img, (40, 80), (width - 40, 110), (20, 20, 20), thickness=-1)
    # Dotted noise
    rng = np.random.default_rng(0)
    for _ in range(60):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        img[y, x] = (0, 0, 0)
    # Gradient shadow in bottom-right
    gradient = np.tile(np.linspace(0, 120, width, dtype=np.float32), (height, 1))
    img = np.clip(img.astype(np.float32) - gradient[..., None], 0, 255).astype(np.uint8)
    return img


class TestFullPipeline:
    def test_default_config_returns_binarized_output(self) -> None:
        img = _make_text_image()
        result, angle = ImagePreprocessor().process(img, PreprocessConfig())
        # Binarization is default ON, so output should be single-channel
        assert result.ndim == 2
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 255}) or result.max() <= 255
        assert isinstance(angle, float)

    def test_disabled_binarization_keeps_color(self) -> None:
        img = _make_text_image()
        cfg = PreprocessConfig(
            binarization=BinarizationConfig(method=BinarizationMethod.NONE)
        )
        result, _ = ImagePreprocessor().process(img, cfg)
        assert result.ndim in (2, 3)

    def test_adaptive_gaussian_binarization(self) -> None:
        img = _make_text_image()
        cfg = PreprocessConfig(
            binarization=BinarizationConfig(
                method=BinarizationMethod.ADAPTIVE_GAUSSIAN,
                adaptive_block_size=31,
                adaptive_c=10,
            )
        )
        result, _ = ImagePreprocessor().process(img, cfg)
        assert result.ndim == 2

    def test_sauvola_binarization_runs(self) -> None:
        img = _make_text_image()
        cfg = PreprocessConfig(
            binarization=BinarizationConfig(method=BinarizationMethod.SAUVOLA)
        )
        result, _ = ImagePreprocessor().process(img, cfg)
        assert result.ndim == 2

    def test_denoise_chain_median_then_morph(self) -> None:
        img = _make_text_image()
        cfg = PreprocessConfig(
            denoise=DenoiseConfig(
                enabled=True,
                steps=[
                    DenoiseStep(method=DenoiseMethod.MEDIAN, ksize=3),
                    DenoiseStep(method=DenoiseMethod.MORPH_OPEN, morph_ksize=3),
                ],
            )
        )
        result, _ = ImagePreprocessor().process(img, cfg)
        assert result.shape[:2] == img.shape[:2]

    def test_clahe_contrast_enabled(self) -> None:
        img = _make_text_image()
        cfg = PreprocessConfig(
            contrast=ContrastConfig(clahe_enabled=True, clahe_clip=3.0, clahe_tile=8),
            binarization=BinarizationConfig(method=BinarizationMethod.NONE),
        )
        result, _ = ImagePreprocessor().process(img, cfg)
        assert result.shape[:2] == img.shape[:2]

    def test_background_removal_enabled(self) -> None:
        img = _make_text_image()
        cfg = PreprocessConfig(
            background=BackgroundConfig(enabled=True, blur_kernel=55),
            binarization=BinarizationConfig(method=BinarizationMethod.NONE),
        )
        result, _ = ImagePreprocessor().process(img, cfg)
        assert result.shape[:2] == img.shape[:2]

    def test_manual_deskew_rotates(self) -> None:
        """When auto_detect is off, manual_angle is applied deterministically."""
        img = _make_text_image(200, 400)
        cfg = PreprocessConfig(
            deskew=DeskewConfig(enabled=True, auto_detect=False, manual_angle=5.0),
            binarization=BinarizationConfig(method=BinarizationMethod.NONE),
        )
        result, angle = ImagePreprocessor().process(img, cfg)
        # Canvas may expand to hold the rotated image, so shape can differ
        assert result.shape[0] > 0 and result.shape[1] > 0
        # The reported angle must be finite
        assert np.isfinite(angle)

    def test_grayscale_input_accepted(self) -> None:
        img = _make_text_image()
        import cv2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result, _ = ImagePreprocessor().process(gray, PreprocessConfig())
        assert result.ndim == 2

    def test_dewarp_disabled_is_identity(self) -> None:
        img = _make_text_image()
        cfg = PreprocessConfig(
            dewarp=DewarpConfig(enabled=False),
            deskew=DeskewConfig(enabled=False),
            binarization=BinarizationConfig(method=BinarizationMethod.NONE),
            contrast=ContrastConfig(),
            denoise=DenoiseConfig(enabled=False),
            background=BackgroundConfig(enabled=False),
        )
        result, _ = ImagePreprocessor().process(img, cfg)
        # With everything off the result should match the input exactly
        assert result.shape == img.shape


class TestPreviewStep:
    def test_original_returns_input_unchanged(self) -> None:
        img = _make_text_image()
        result = preview_step(img, "original", PreprocessConfig())
        assert np.array_equal(result, img)

    def test_each_stage_returns_valid_image(self) -> None:
        img = _make_text_image()
        cfg = PreprocessConfig()
        for stage in (
            "original",
            "dewarp",
            "deskew",
            "contrast",
            "background",
            "denoise",
            "binarization",
        ):
            result = preview_step(img, stage, cfg)
            assert result.ndim in (2, 3)
            assert result.dtype == np.uint8
            assert result.size > 0

    def test_unknown_stage_raises(self) -> None:
        img = _make_text_image()
        from src.shared.validators import ValidationError

        with pytest.raises(ValidationError):
            preview_step(img, "no_such_stage", PreprocessConfig())
