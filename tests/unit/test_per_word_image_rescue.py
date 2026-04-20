"""Unit tests for :mod:`src.core.per_word_image_rescue`.

Covers the gate-keeper (``_is_rescuable``), both rescue strategies
(``clahe_sharpen_rescue``, ``upscale_rescue``) and their swap /
keep-original semantics. The actual Tesseract calls are mocked so
the tests run fast and deterministically on CI machines that don't
have the bundled model available.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.core.per_word_image_rescue import (
    _is_rescuable,
    clahe_sharpen_rescue,
    upscale_rescue,
)


def _fake_tsv(words: list[str], confs: list[float]) -> dict:
    return {"text": list(words), "conf": [str(c) for c in confs]}


class TestIsRescuable:
    def test_conf_in_band_rescuable(self) -> None:
        assert _is_rescuable("hello", 50.0)

    def test_conf_below_band_not_rescuable(self) -> None:
        # Below 30 — pure noise, re-OCR won't help.
        assert not _is_rescuable("hello", 20.0)

    def test_conf_above_band_not_rescuable(self) -> None:
        # Above 70 — already good, don't churn.
        assert not _is_rescuable("hello", 80.0)

    def test_boundary_30_included(self) -> None:
        assert _is_rescuable("hello", 30.0)

    def test_boundary_70_included(self) -> None:
        assert _is_rescuable("hello", 70.0)

    def test_short_word_not_rescuable(self) -> None:
        # 1-char tokens are usually punctuation fragments — skip.
        assert not _is_rescuable("a", 50.0)

    def test_punctuation_only_not_rescuable(self) -> None:
        assert not _is_rescuable("...", 50.0)
        assert not _is_rescuable(",,,", 50.0)

    def test_whitespace_stripped(self) -> None:
        # "  hi  " is 4 chars including spaces, 2 after strip.
        assert _is_rescuable("  hi  ", 50.0)


class TestClaheSharpenRescue:
    def test_swaps_when_conf_lifted(self) -> None:
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)

        with patch(
            "src.core.per_word_image_rescue.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.return_value = _fake_tsv(
                ["ИНН"], [85.0],
            )
            mock_py.Output.DICT = "dict"
            word, conf = clahe_sharpen_rescue(
                img, bbox,
                original_word="ИН",   # line-level misread
                original_confidence=55.0,
                lang="rus+eng",
            )
        assert word == "ИНН"
        assert conf == 85.0

    def test_keeps_original_when_lift_too_small(self) -> None:
        """Sub-10 point win is measurement noise — keep original."""
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)
        with patch(
            "src.core.per_word_image_rescue.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.return_value = _fake_tsv(
                ["ALT"], [58.0],
            )
            mock_py.Output.DICT = "dict"
            word, conf = clahe_sharpen_rescue(
                img, bbox,
                original_word="ORIG",
                original_confidence=55.0,
                lang="rus+eng",
            )
        assert word == "ORIG"
        assert conf == 55.0

    def test_bumps_conf_when_same_text_recovered(self) -> None:
        """Same text at higher conf — keep text but bump conf. The
        rescue serves TWO purposes: (1) rescue TEXT, (2) rescue
        CONFIDENCE from the drop-threshold cliff."""
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)
        with patch(
            "src.core.per_word_image_rescue.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.return_value = _fake_tsv(
                ["ИНН"], [80.0],
            )
            mock_py.Output.DICT = "dict"
            word, conf = clahe_sharpen_rescue(
                img, bbox,
                original_word="ИНН",
                original_confidence=50.0,
                lang="rus+eng",
            )
        assert word == "ИНН"
        assert conf == 80.0

    def test_skipped_below_conf_band(self) -> None:
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)
        # Confidence 20 < 30 low bound → gate keeper declines early.
        with patch(
            "src.core.per_word_image_rescue.pytesseract"
        ) as mock_py:
            word, conf = clahe_sharpen_rescue(
                img, bbox,
                original_word="noisy",
                original_confidence=20.0,
                lang="rus+eng",
            )
            mock_py.image_to_data.assert_not_called()
        assert word == "noisy"
        assert conf == 20.0

    def test_invalid_image_keeps_original(self) -> None:
        word, conf = clahe_sharpen_rescue(
            "not an array", (0, 0, 10, 10),
            original_word="orig", original_confidence=50.0,
            lang="rus+eng",
        )
        assert word == "orig"
        assert conf == 50.0

    def test_zero_bbox_keeps_original(self) -> None:
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        word, conf = clahe_sharpen_rescue(
            img, (10, 10, 0, 0),
            original_word="orig", original_confidence=50.0,
            lang="rus+eng",
        )
        assert word == "orig"
        assert conf == 50.0

    def test_tesseract_error_keeps_original(self) -> None:
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)
        with patch(
            "src.core.per_word_image_rescue.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = RuntimeError("ded")
            mock_py.Output.DICT = "dict"
            word, conf = clahe_sharpen_rescue(
                img, bbox,
                original_word="orig", original_confidence=55.0,
                lang="rus+eng",
            )
        assert word == "orig"
        assert conf == 55.0


class TestUpscaleRescue:
    def test_upscales_and_swaps_when_lifted(self) -> None:
        """2× upscale + re-OCR recovers small-glyph text."""
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)
        captured_shapes: list = []

        def _capture(crop, lang, config, output_type):
            captured_shapes.append(crop.shape[:2])
            return _fake_tsv(["7701234567"], [82.0])

        with patch(
            "src.core.per_word_image_rescue.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = _capture
            mock_py.Output.DICT = "dict"
            word, conf = upscale_rescue(
                img, bbox,
                original_word="77O12345G7",
                original_confidence=55.0,
                lang="rus+eng",
            )
        assert word == "7701234567"
        assert conf == 82.0
        # Crop was upscaled 2× — bbox is 60×30 plus 12 px padding
        # on each side ≈ 72×42, doubled to ≈ 144×84.
        assert captured_shapes[0][0] >= 70
        assert captured_shapes[0][1] >= 100

    def test_configurable_scale(self) -> None:
        """3× upscale produces larger crop than 2×."""
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)
        shapes: list = []

        def _capture(crop, lang, config, output_type):
            shapes.append(crop.shape[:2])
            return _fake_tsv(["word"], [90.0])

        with patch(
            "src.core.per_word_image_rescue.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = _capture
            mock_py.Output.DICT = "dict"
            upscale_rescue(
                img, bbox,
                original_word="orig", original_confidence=55.0,
                lang="rus+eng", scale=3,
            )
        # 3× is at least 50% bigger than 2× would produce.
        assert shapes[0][0] > 100

    def test_skipped_when_original_too_high(self) -> None:
        """75 % is above the 70 % ceiling — no rescue attempt."""
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)
        with patch(
            "src.core.per_word_image_rescue.pytesseract"
        ) as mock_py:
            word, conf = upscale_rescue(
                img, bbox,
                original_word="good", original_confidence=75.0,
                lang="rus+eng",
            )
            mock_py.image_to_data.assert_not_called()
        assert word == "good"
        assert conf == 75.0
