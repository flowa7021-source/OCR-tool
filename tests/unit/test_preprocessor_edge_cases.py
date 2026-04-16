"""Edge-case / defensive-path tests for the preprocessing chain.

The integration tests in ``tests/integration/test_image_preprocessor``
cover the happy path on a realistic synthetic scan. These tests fill
the gaps that showed up in coverage: validation errors, degenerate
inputs, individual binarization/denoise/contrast branches, and
``preview_step`` selector dispatch.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.core.deskew_handler import DeskewHandler, rotate_image
from src.core.image_preprocessor import (
    ImagePreprocessor,
    _sauvola_binarize,
    _to_grayscale,
    preview_step,
)
from src.core.models import (
    BackgroundConfig,
    BinarizationConfig,
    ContrastConfig,
    DenoiseConfig,
    DenoiseStep,
    DeskewConfig,
    PreprocessConfig,
)
from src.shared.types import BinarizationMethod, DenoiseMethod
from src.shared.validators import ValidationError


def _solid_image(value: int = 200, shape: tuple[int, int, int] = (80, 120, 3)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def _minimal_cfg() -> PreprocessConfig:
    """PreprocessConfig with deskew off and no binarization.

    Makes individual-stage tests deterministic — otherwise the default
    profile's ``deskew.auto_detect=True`` runs the whole skew-detector
    on every call, and different scenarios can't be isolated cleanly.
    """
    cfg = PreprocessConfig()
    cfg.deskew = DeskewConfig(enabled=False)
    cfg.binarization = BinarizationConfig(method=BinarizationMethod.NONE)
    return cfg


# ---------------------------------------------------------------------------
# Validation: the stated contract is "raise ValidationError"
# ---------------------------------------------------------------------------


class TestValidation:
    def test_process_rejects_non_ndarray(self) -> None:
        pp = ImagePreprocessor()
        with pytest.raises(ValidationError):
            pp.process("not an array", PreprocessConfig())  # type: ignore[arg-type]

    def test_process_rejects_empty_image(self) -> None:
        pp = ImagePreprocessor()
        empty = np.array([], dtype=np.uint8)
        with pytest.raises(ValidationError):
            pp.process(empty, PreprocessConfig())

    def test_rotate_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValidationError):
            rotate_image("x", 30)  # type: ignore[arg-type]

    def test_rotate_rejects_empty_image(self) -> None:
        with pytest.raises(ValidationError):
            rotate_image(np.array([], dtype=np.uint8), 30)

    def test_detect_angle_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValidationError):
            DeskewHandler().detect_angle("x")  # type: ignore[arg-type]

    def test_detect_angle_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            DeskewHandler().detect_angle(np.array([], dtype=np.uint8))


# ---------------------------------------------------------------------------
# Individual stages — covering the branches inside the pipeline
# ---------------------------------------------------------------------------


class TestBinarization:
    @pytest.mark.parametrize(
        "method",
        [
            BinarizationMethod.NONE,
            BinarizationMethod.OTSU,
            BinarizationMethod.ADAPTIVE_GAUSSIAN,
            BinarizationMethod.ADAPTIVE_MEAN,
            BinarizationMethod.SAUVOLA,
        ],
    )
    def test_each_method_produces_valid_output(self, method) -> None:
        """All binarization methods return uint8 images."""
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.binarization = BinarizationConfig(method=method)

        out, angle = pp.process(_solid_image(), cfg)

        assert out.dtype == np.uint8
        assert out.size > 0
        assert angle == 0.0
        # NONE keeps the colour channels; the others flatten to grayscale.
        if method is BinarizationMethod.NONE:
            assert out.ndim == 3
        else:
            assert out.ndim == 2

    def test_sauvola_helper_direct(self) -> None:
        """Direct call to _sauvola_binarize with a contrasty gradient."""
        gray = np.tile(np.linspace(0, 255, 40, dtype=np.uint8), (40, 1))
        binarized = _sauvola_binarize(gray, window_size=25, k=0.2)
        assert binarized.dtype == np.uint8
        assert binarized.shape == gray.shape
        # Output should be strictly two-valued.
        unique = set(np.unique(binarized).tolist())
        assert unique <= {0, 255}

    def test_adaptive_rejects_even_block_size(self) -> None:
        """Adaptive thresholding needs an odd block size — we validate."""
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.binarization = BinarizationConfig(
            method=BinarizationMethod.ADAPTIVE_GAUSSIAN, adaptive_block_size=30
        )
        with pytest.raises(ValidationError):
            pp.process(_solid_image(), cfg)


class TestDenoise:
    @pytest.mark.parametrize(
        "method,kwargs",
        [
            (DenoiseMethod.MEDIAN, {"ksize": 5}),
            (DenoiseMethod.GAUSSIAN, {"sigma": 1.5}),
            (DenoiseMethod.MORPH_OPEN, {"ksize": 3}),
            (DenoiseMethod.MORPH_CLOSE, {"ksize": 3}),
            (DenoiseMethod.NLM, {"h": 7}),
        ],
    )
    def test_each_denoise_method(self, method, kwargs) -> None:
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.denoise = DenoiseConfig(
            enabled=True,
            steps=[DenoiseStep(method=method, **kwargs)],
        )
        out, _ = pp.process(_solid_image(), cfg)
        assert out.dtype == np.uint8
        assert out.shape[:2] == (80, 120)

    def test_disabled_denoise_returns_input_shape(self) -> None:
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.denoise = DenoiseConfig(enabled=False)
        out, _ = pp.process(_solid_image(), cfg)
        assert out.shape == (80, 120, 3)

    def test_median_rejects_even_ksize(self) -> None:
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.denoise = DenoiseConfig(
            enabled=True,
            steps=[DenoiseStep(method=DenoiseMethod.MEDIAN, ksize=4)],
        )
        with pytest.raises(ValidationError):
            pp.process(_solid_image(), cfg)


class TestContrast:
    def test_clahe_enabled_changes_image(self) -> None:
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.contrast = ContrastConfig(
            clahe_enabled=True, clahe_clip=3.0, clahe_tile=8
        )
        import cv2

        img = np.full((80, 120, 3), 128, dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (40, 40), (200, 200, 200), -1)
        out, _ = pp.process(img, cfg)
        assert out.dtype == np.uint8
        assert not np.array_equal(out, img)

    def test_manual_contrast_brightness_shift(self) -> None:
        """Manual α * x + β contrast/brightness shift is honoured."""
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.contrast = ContrastConfig(
            clahe_enabled=False, manual_enabled=True, alpha=1.5, beta=10
        )
        out, _ = pp.process(_solid_image(value=120), cfg)
        # 120 * 1.5 + 10 = 190, within uint8 range.
        assert abs(int(out.mean()) - 190) < 5


class TestBackgroundRemoval:
    def test_background_removal_runs(self) -> None:
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.background = BackgroundConfig(enabled=True, blur_kernel=15)
        # Image with a smooth gradient (simulating paper shading).
        grad = np.tile(
            np.linspace(50, 255, 120, dtype=np.uint8), (80, 1)
        )
        img = np.stack([grad, grad, grad], axis=-1)
        out, _ = pp.process(img, cfg)
        assert out.dtype == np.uint8

    def test_background_rejects_even_kernel(self) -> None:
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.background = BackgroundConfig(enabled=True, blur_kernel=10)
        with pytest.raises(ValidationError):
            pp.process(_solid_image(), cfg)


class TestPreviewStep:
    def test_original_returns_input(self) -> None:
        img = _solid_image()
        out = preview_step(img, "original", _minimal_cfg())
        assert out.dtype == np.uint8
        # "original" returns the input as-is.
        assert out.shape == img.shape

    def test_unknown_step_raises(self) -> None:
        with pytest.raises(ValidationError):
            preview_step(_solid_image(), "not-a-step", _minimal_cfg())

    @pytest.mark.parametrize(
        "step",
        ["dewarp", "deskew", "contrast", "background", "denoise", "binarization"],
    )
    def test_each_step_produces_array(self, step) -> None:
        """Every documented step returns an ndarray without crashing."""
        # Use _minimal_cfg so stages are off-by-default; preview_step
        # still runs without raising.
        out = preview_step(_solid_image(), step, _minimal_cfg())
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint8


class TestGrayscaleHelper:
    def test_rgb_to_gray(self) -> None:
        img = _solid_image()
        gray = _to_grayscale(img)
        assert gray.ndim == 2
        assert gray.shape == (80, 120)

    def test_already_gray_passthrough(self) -> None:
        gray = np.full((30, 30), 128, dtype=np.uint8)
        out = _to_grayscale(gray)
        assert out.ndim == 2
        assert np.array_equal(out, gray)


# ---------------------------------------------------------------------------
# Deskew: auto-detect + manual rotation branches
# ---------------------------------------------------------------------------


class TestDeskew:
    def test_detect_angle_on_flat_image_returns_bounded(self) -> None:
        """No text → skew-detector has nothing to lock onto → tiny angle."""
        handler = DeskewHandler()
        out = handler.detect_angle(_solid_image())
        # Result must be clamped within the configured maximum.
        assert abs(out) <= handler._max_abs_angle

    def test_detect_angle_clamps_to_max(self) -> None:
        handler = DeskewHandler(max_abs_angle=5.0)
        assert handler._max_abs_angle == 5.0

    def test_rotate_zero_angle_is_identity(self) -> None:
        img = _solid_image()
        out = rotate_image(img, 0.0)
        assert out is img  # fast path returns the original

    def test_rotate_nonzero_produces_new_array(self) -> None:
        img = _solid_image()
        out = rotate_image(img, 5.0)
        assert out is not img
        # Rotation produces a different but uint8 array.
        assert out.dtype == np.uint8
        # Size may grow slightly because rotate_image uses "expand" to
        # avoid corner clipping — bounding box grows with the angle.
        assert out.ndim == img.ndim

    def test_manual_angle_with_auto_detect_disabled(self) -> None:
        """Pipeline-level: deskew enabled + auto_detect=False → uses manual."""
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.deskew = DeskewConfig(
            enabled=True, auto_detect=False, manual_angle=3.0
        )
        _, angle = pp.process(_solid_image(), cfg)
        assert angle == pytest.approx(3.0)

    def test_disabled_deskew_yields_zero_angle(self) -> None:
        pp = ImagePreprocessor()
        cfg = _minimal_cfg()
        cfg.deskew = DeskewConfig(enabled=False)
        _, angle = pp.process(_solid_image(), cfg)
        assert angle == 0.0
