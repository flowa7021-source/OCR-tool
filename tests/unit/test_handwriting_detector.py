"""Tests for handwriting vs printed-text classifier."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.application.handwriting_detector import (
    HandwritingFeatures,
    classify_crop,
)


def _make_printed_text_crop(
    w: int = 200, h: int = 40, text: str = "HELLO",
) -> np.ndarray:
    """Synthesise a crop with cv2.putText — fairly uniform strokes,
    flat baseline — as a proxy for printed text.
    """
    img = np.full((h, w), 255, dtype=np.uint8)
    cv2.putText(img, text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, 0, thickness=2, lineType=cv2.LINE_AA)
    return img


def _make_scribble_crop(
    w: int = 200, h: int = 40, seed: int = 42,
) -> np.ndarray:
    """Simulate handwriting: varying stroke thickness + curved path.
    Not a real glyph but produces the right *feature statistics*.
    """
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 255, dtype=np.uint8)
    xs = np.linspace(10, w - 10, 60)
    ys = 20 + 10 * np.sin(xs / 10.0) + rng.uniform(-3, 3, len(xs))
    thicknesses = rng.integers(1, 5, len(xs))
    for (x0, y0, t0), (x1, y1, _t1) in zip(
        list(zip(xs, ys, thicknesses, strict=False)),
        list(zip(xs[1:], ys[1:], thicknesses[1:], strict=False)),
        strict=False,
    ):
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), 0,
                 thickness=int(t0))
    return img


class TestClassifyCrop:
    def test_rejects_non_grayscale(self):
        rgb = np.full((40, 200, 3), 255, dtype=np.uint8)
        with pytest.raises(ValueError, match="grayscale"):
            classify_crop(rgb)

    def test_tiny_crop_not_handwritten(self):
        tiny = np.full((5, 5), 255, dtype=np.uint8)
        is_hw, feats = classify_crop(tiny)
        assert is_hw is False
        assert isinstance(feats, HandwritingFeatures)
        assert feats.score == 0.0

    def test_blank_crop_not_handwritten(self):
        blank = np.full((40, 200), 255, dtype=np.uint8)
        is_hw, _ = classify_crop(blank)
        assert is_hw is False

    def test_printed_text_scores_low(self):
        img = _make_printed_text_crop()
        is_hw, feats = classify_crop(img)
        # Should have low score (clearly printed)
        assert feats.score < 0.7
        # Features sanity
        assert feats.stroke_variance >= 0
        assert feats.baseline_slope >= 0
        assert 0 <= feats.edge_density <= 1

    def test_scribble_scores_higher_than_printed(self):
        printed = _make_printed_text_crop()
        scribble = _make_scribble_crop()
        _, p_feats = classify_crop(printed)
        _, s_feats = classify_crop(scribble)
        # Scribble should score at least somewhat higher on the
        # handwriting score than printed text. Relative comparison
        # is what matters; absolute threshold is a heuristic.
        assert s_feats.score >= p_feats.score

    def test_feature_serialisation(self):
        img = _make_printed_text_crop()
        _, feats = classify_crop(img)
        d = feats.as_dict()
        assert set(d.keys()) == {
            "stroke_variance", "baseline_slope", "edge_density", "score",
        }
