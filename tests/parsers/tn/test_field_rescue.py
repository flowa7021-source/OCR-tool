"""Tests for per-field OCR retry via EasyOCR.

Pipeline retry works at the PAGE level. When a specific field is
empty / garbage the rescue re-runs OCR on a targeted bbox via
EasyOCR. We monkeypatch ``_easyocr_readtext`` so the suite runs
without the real ``easyocr.Reader``.
"""

from __future__ import annotations

import numpy as np


class TestRescueFieldCrop:
    """``_crop_bbox(raster, bbox)`` — safe crop."""

    def _call(self, raster, bbox):
        from src.tn_parser.field_rescue import _crop_bbox
        return _crop_bbox(raster, bbox)

    def test_basic_crop(self):
        raster = np.ones((100, 200, 3), dtype=np.uint8) * 128
        cropped = self._call(raster, (10, 20, 50, 30))
        assert cropped.shape == (30, 50, 3)

    def test_bbox_clipped_to_raster_bounds(self):
        raster = np.ones((50, 50), dtype=np.uint8) * 100
        cropped = self._call(raster, (40, 40, 100, 100))
        assert cropped.shape == (10, 10)

    def test_invalid_bbox_returns_none(self):
        raster = np.ones((50, 50), dtype=np.uint8)
        assert self._call(raster, (100, 100, 50, 50)) is None
        assert self._call(raster, (-10, -10, 5, 5)) is None


class TestRescueField:
    """``rescue_field(raster, bbox, ...)`` — calls EasyOCR and returns
    ``(text, conf 0..1)``."""

    def test_rescue_returns_text_and_conf(self, monkeypatch):
        from src.tn_parser import field_rescue

        def fake_readtext(img, *, reader=None, allowlist=None):
            return [
                ([(0, 0), (50, 0), (50, 20), (0, 20)], "Беляев", 0.85),
                ([(60, 0), (110, 0), (110, 20), (60, 20)], "А.Н.", 0.90),
            ]

        monkeypatch.setattr(
            field_rescue, "_easyocr_readtext", fake_readtext,
        )

        raster = np.ones((200, 300, 3), dtype=np.uint8) * 128
        text, conf = field_rescue.rescue_field(
            raster, bbox=(10, 10, 200, 40),
        )
        assert "Беляев" in text
        assert "А.Н." in text
        assert abs(conf - 0.875) < 0.01

    def test_rescue_handles_easyocr_error(self, monkeypatch):
        from src.tn_parser import field_rescue

        def broken(*args, **kwargs):
            raise RuntimeError("easyocr crashed")

        monkeypatch.setattr(
            field_rescue, "_easyocr_readtext", broken,
        )

        raster = np.ones((100, 100), dtype=np.uint8)
        text, conf = field_rescue.rescue_field(
            raster, bbox=(0, 0, 50, 50),
        )
        assert text == ""
        assert conf == 0.0

    def test_rescue_with_invalid_bbox_returns_empty(self, monkeypatch):
        from src.tn_parser import field_rescue

        call_count = {"n": 0}

        def sentinel(*args, **kwargs):
            call_count["n"] += 1
            return []

        monkeypatch.setattr(
            field_rescue, "_easyocr_readtext", sentinel,
        )

        raster = np.ones((50, 50), dtype=np.uint8)
        text, conf = field_rescue.rescue_field(
            raster, bbox=(100, 100, 50, 50),  # invalid
        )
        assert text == ""
        assert conf == 0.0
        # EasyOCR не вызывался — crop вернул None.
        assert call_count["n"] == 0


class TestRescueFilters:
    """Low-conf tokens are filtered out of the final text."""

    def test_low_conf_tokens_skipped(self, monkeypatch):
        from src.tn_parser import field_rescue

        def stub(img, *, reader=None, allowlist=None):
            return [
                ([(0, 0), (10, 0), (10, 10), (0, 10)], "real", 0.90),
                ([(20, 0), (30, 0), (30, 10), (20, 10)], "", 0.0),
                ([(40, 0), (60, 0), (60, 10), (40, 10)], "LOWCONF", 0.20),
                ([(70, 0), (90, 0), (90, 10), (70, 10)], "another", 0.85),
            ]

        monkeypatch.setattr(
            field_rescue, "_easyocr_readtext", stub,
        )

        raster = np.ones((100, 100), dtype=np.uint8)
        text, conf = field_rescue.rescue_field(
            raster, bbox=(0, 0, 50, 50), min_token_conf=0.4,
        )
        assert "real" in text
        assert "another" in text
        assert "LOWCONF" not in text
        assert abs(conf - (0.90 + 0.85) / 2) < 0.01
