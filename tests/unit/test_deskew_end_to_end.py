"""Tests for end-to-end deskew correctness — TDD.

The user-facing complaint was: "после распознавания в итоговом файле
всё равно остаётся наклон страниц" — the output PDF still shows
tilt after the pipeline claims to have deskewed it. These tests
pin down the expected behaviour:

* A page rendered with a known skew angle, run through
  :class:`ImagePreprocessor`, must come out with a residual
  angle ≤ the tolerance below.
* Multiple angle magnitudes (1°, 3°, 5°, 7°, -4°) all land under
  tolerance. A single-angle test is a shallow guard — a deskew
  implementation that only handles one direction would still pass.
* Detection-and-correction cycle is idempotent: applying deskew
  to an already-straight image leaves it within tolerance.

Measurement technique: project the grayscale row variances onto
the skew angle — the classic horizontal-projection-profile
algorithm Tesseract itself uses internally for OSD. Returns the
angle that maximises variance across rows, i.e. the angle at
which horizontal text lines are most parallel to the rows.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.core.deskew_handler import DeskewHandler, rotate_image
from src.core.image_preprocessor import ImagePreprocessor
from src.core.models import (
    BackgroundConfig,
    BinarizationConfig,
    ContrastConfig,
    DenoiseConfig,
    DeskewConfig,
    DewarpConfig,
    PreprocessConfig,
)
from src.shared.types import BinarizationMethod

# Tolerance for residual skew. 0.3° is below human-visible threshold
# on an A4 page and tight enough that a broken deskew can't sneak
# through with "close enough" numbers.
RESIDUAL_ANGLE_TOLERANCE_DEG: float = 0.3


def _render_tilted_text_page(angle_deg: float, size: int = 1000) -> np.ndarray:
    """Render a white page with a few black text-like lines, rotated by
    ``angle_deg``. Returns the rotated grayscale image.

    Uses cv2.putText to stamp "TEXT LINE N" at evenly-spaced rows so
    the result has enough horizontal structure for row-variance
    projection to pick up the angle.
    """
    canvas = np.full((size, size), 255, dtype=np.uint8)
    for i in range(4, 16):
        y = int(size * i / 20)
        cv2.putText(
            canvas,
            f"TEXT LINE {i}",
            (int(size * 0.15), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            0,
            thickness=2,
        )
    return rotate_image(canvas, angle_deg, border_value=255)


def _measure_skew_angle(
    image: np.ndarray, *, search_range: float = 10.0, step: float = 0.05,
) -> float:
    """Return the angle in degrees at which horizontal row variance
    is maximised — i.e. the remaining skew of ``image``.

    We rotate the image through ``[-search_range, +search_range]``
    and score each candidate by the sum of row-wise variances on
    the binarised image. The argmax is the true remaining skew,
    because text lines project cleanest when they're parallel to
    rows.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    best_score = -1.0
    best_angle = 0.0
    angles = np.arange(-search_range, search_range + step, step)
    for a in angles:
        rotated = rotate_image(binary, float(a), border_value=0)
        projection = rotated.sum(axis=1, dtype=np.float64)
        score = projection.var()
        if score > best_score:
            best_score = float(score)
            best_angle = float(a)
    return best_angle


def _make_default_preprocess_config() -> PreprocessConfig:
    """PreprocessConfig with ONLY deskew enabled — we're measuring
    just the deskew step, not the full pipeline."""
    return PreprocessConfig(
        deskew=DeskewConfig(enabled=True, auto_detect=True, max_angle=30.0),
        dewarp=DewarpConfig(enabled=False),
        binarization=BinarizationConfig(method=BinarizationMethod.NONE),
        denoise=DenoiseConfig(enabled=False, steps=[]),
        contrast=ContrastConfig(clahe_enabled=False),
        background=BackgroundConfig(enabled=False),
    )


# ---------------------------------------------------------------------------
# Sanity: the measurement tool itself works
# ---------------------------------------------------------------------------


class TestMeasurementToolingSanity:
    def test_straight_page_measures_zero(self) -> None:
        page = _render_tilted_text_page(0.0)
        residual = _measure_skew_angle(page)
        assert abs(residual) < RESIDUAL_ANGLE_TOLERANCE_DEG

    def test_tilted_page_measures_expected_angle(self) -> None:
        for ground_truth in [2.0, -3.0, 5.0]:
            page = _render_tilted_text_page(ground_truth)
            measured = _measure_skew_angle(page)
            # rotate_image uses counter-clockwise positive; the
            # measurement inverts because the projection picks the
            # angle that STRAIGHTENS the page.
            assert abs(-measured - ground_truth) < 0.5, (
                f"Measurement tool off for {ground_truth}°: "
                f"measured {measured}°"
            )


# ---------------------------------------------------------------------------
# The real deskew contract: residual angle after ImagePreprocessor
# ---------------------------------------------------------------------------


class TestEndToEndDeskew:
    """Each input angle must come out ≤0.3° after the preprocessor.

    The test isolates deskew — no binarisation, no CLAHE, no background
    removal — so a failure points straight at the deskew stage.
    """

    @pytest.mark.parametrize("input_angle", [1.0, 2.0, -3.0, 5.0, -4.0, 7.0])
    def test_residual_angle_within_tolerance(
        self, input_angle: float,
    ) -> None:
        tilted = _render_tilted_text_page(input_angle)
        preprocessor = ImagePreprocessor(
            deskew_handler=DeskewHandler(max_abs_angle=30.0),
        )
        processed, applied_angle = preprocessor.process(
            tilted, _make_default_preprocess_config(),
        )

        residual = _measure_skew_angle(processed)
        assert abs(residual) <= RESIDUAL_ANGLE_TOLERANCE_DEG, (
            f"Input {input_angle}°: residual {residual:.3f}° "
            f"(tolerance {RESIDUAL_ANGLE_TOLERANCE_DEG}°). "
            f"applied_angle reported = {applied_angle:.3f}°"
        )

    def test_already_straight_page_stays_straight(self) -> None:
        page = _render_tilted_text_page(0.0)
        preprocessor = ImagePreprocessor()
        processed, _ = preprocessor.process(
            page, _make_default_preprocess_config(),
        )
        residual = _measure_skew_angle(processed)
        assert abs(residual) <= RESIDUAL_ANGLE_TOLERANCE_DEG


# ---------------------------------------------------------------------------
# Two-pass correction — guards against weak primary detection
# ---------------------------------------------------------------------------


class TestTwoPassCorrection:
    """Guards the user-reported failure: ``deskew`` library can
    under-detect on noisy/real scans. A second pass via row-
    projection measurement must catch that residual and apply a
    corrective rotation.

    Simulates the failure by mocking :class:`DeskewHandler` to
    return 0.0 regardless of input — i.e. "primary detector
    totally failed". The preprocessor's two-pass correction MUST
    still deliver a straight output.
    """

    def test_preprocessor_recovers_when_primary_returns_zero(self) -> None:
        from unittest.mock import MagicMock

        tilted = _render_tilted_text_page(4.0)
        failing_handler = MagicMock(spec=DeskewHandler)
        failing_handler.detect_angle.return_value = 0.0

        preprocessor = ImagePreprocessor(deskew_handler=failing_handler)
        processed, applied = preprocessor.process(
            tilted, _make_default_preprocess_config(),
        )

        residual = _measure_skew_angle(processed)
        assert abs(residual) <= RESIDUAL_ANGLE_TOLERANCE_DEG, (
            f"Two-pass correction failed when primary detector "
            f"returned 0: residual {residual:.3f}°, applied {applied:.3f}°"
        )

    def test_preprocessor_recovers_when_primary_undercorrects(self) -> None:
        from unittest.mock import MagicMock

        tilted = _render_tilted_text_page(5.0)
        # Primary returns half the real angle — simulating a weak
        # detection on real noisy content.
        undercorrecting = MagicMock(spec=DeskewHandler)
        undercorrecting.detect_angle.return_value = 2.5

        preprocessor = ImagePreprocessor(deskew_handler=undercorrecting)
        processed, applied = preprocessor.process(
            tilted, _make_default_preprocess_config(),
        )

        residual = _measure_skew_angle(processed)
        assert abs(residual) <= RESIDUAL_ANGLE_TOLERANCE_DEG, (
            f"Two-pass correction failed on under-detection: "
            f"residual {residual:.3f}°, applied {applied:.3f}°"
        )
