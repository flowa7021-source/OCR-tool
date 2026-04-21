"""Тесты per-field OCR retry (idea #6 top-10).

Pipeline retry работает на СТРАНИЦЕ целиком. Но часто проблема
локализована: раздел «6. Водитель» OCR'улся в мусор, а остальные
поля OK. Тогда нужна targeted rescue: re-run tesseract только на
bbox раздела 6 с specialized настройками — PSM=7 (single line),
whitelist=Cyrillic+digits, user-words=ФИО-лексикон.

Контракт:

  * ``rescue_field(raster, bbox, psm=7, whitelist=None,
    user_words=None) → (text, conf)``.
  * raster — ndarray preprocessed страницы.
  * bbox — (x, y, w, h) в пикселях raster'а.
  * При ошибке (tesseract недоступен / bbox выходит за границы)
    возвращает ('', 0.0) без exception'а.

Mocking: в unit-тестах заменяем pytesseract-call на stub чтобы
не требовать tesseract на test-машине.
"""

from __future__ import annotations

import numpy as np


class TestRescueFieldCrop:
    """``_crop_bbox(raster, bbox)`` — безопасный crop."""

    def _call(self, raster, bbox):
        from src.tn_parser.field_rescue import _crop_bbox
        return _crop_bbox(raster, bbox)

    def test_basic_crop(self):
        raster = np.ones((100, 200, 3), dtype=np.uint8) * 128
        cropped = self._call(raster, (10, 20, 50, 30))
        assert cropped.shape == (30, 50, 3)

    def test_bbox_clipped_to_raster_bounds(self):
        """bbox выходит за края — обрезается до raster size."""
        raster = np.ones((50, 50), dtype=np.uint8) * 100
        cropped = self._call(raster, (40, 40, 100, 100))
        # Фактически (40..50, 40..50) = 10×10.
        assert cropped.shape == (10, 10)

    def test_invalid_bbox_returns_none(self):
        """bbox вне raster bounds совсем → None."""
        raster = np.ones((50, 50), dtype=np.uint8)
        assert self._call(raster, (100, 100, 50, 50)) is None
        assert self._call(raster, (-10, -10, 5, 5)) is None


class TestRescueField:
    """``rescue_field(raster, bbox, psm, whitelist, user_words)``
    — high-level entry. Вызывает pytesseract и возвращает (text,
    conf 0..1)."""

    def test_rescue_returns_text_and_conf(self, monkeypatch):
        """Stub pytesseract: возвращаем гарантированный text."""
        from src.tn_parser import field_rescue

        def fake_image_to_data(img, **kwargs):
            return {
                "text": ["Беляев", "А.Н."],
                "conf": [85.0, 90.0],
                "level": [5, 5],
            }

        # Применяем monkeypatch к модулю rescue:
        monkeypatch.setattr(
            field_rescue, "_pytesseract_image_to_data", fake_image_to_data,
        )

        raster = np.ones((200, 300, 3), dtype=np.uint8) * 128
        text, conf = field_rescue.rescue_field(
            raster, bbox=(10, 10, 200, 40), psm=7,
        )
        assert "Беляев" in text
        assert "А.Н." in text
        # (85 + 90) / 2 / 100 = 0.875
        assert abs(conf - 0.875) < 0.01

    def test_rescue_handles_pytesseract_error(self, monkeypatch):
        """pytesseract падает — rescue возвращает ('', 0.0),
        НЕ пропагирует exception."""
        from src.tn_parser import field_rescue

        def broken(*args, **kwargs):
            raise RuntimeError("tesseract crashed")

        monkeypatch.setattr(
            field_rescue, "_pytesseract_image_to_data", broken,
        )

        raster = np.ones((100, 100), dtype=np.uint8)
        text, conf = field_rescue.rescue_field(
            raster, bbox=(0, 0, 50, 50),
        )
        assert text == ""
        assert conf == 0.0

    def test_rescue_with_invalid_bbox_returns_empty(self, monkeypatch):
        """Bbox вне raster → empty, no call to tesseract."""
        from src.tn_parser import field_rescue

        call_count = {"n": 0}

        def sentinel(*args, **kwargs):
            call_count["n"] += 1
            return {"text": [], "conf": [], "level": []}

        monkeypatch.setattr(
            field_rescue, "_pytesseract_image_to_data", sentinel,
        )

        raster = np.ones((50, 50), dtype=np.uint8)
        text, conf = field_rescue.rescue_field(
            raster, bbox=(100, 100, 50, 50),  # invalid
        )
        assert text == ""
        assert conf == 0.0
        # tesseract вообще не вызывался — crop вернул None.
        assert call_count["n"] == 0


class TestRescueFilters:
    """Low-conf tokens в output'е фильтруются (tesseract иногда
    даёт conf=-1 для placeholder-токенов; они должны не попадать
    в итоговый text)."""

    def test_low_conf_tokens_skipped(self, monkeypatch):
        from src.tn_parser import field_rescue

        def stub(img, **kwargs):
            return {
                "text": ["real", "", "LOWCONF", "another"],
                "conf": [90, -1, 20, 85],
                "level": [5, 5, 5, 5],
            }

        monkeypatch.setattr(
            field_rescue, "_pytesseract_image_to_data", stub,
        )

        raster = np.ones((100, 100), dtype=np.uint8)
        text, conf = field_rescue.rescue_field(
            raster, bbox=(0, 0, 50, 50), min_token_conf=40,
        )
        # "" с conf=-1 skipped; "LOWCONF" с conf=20 < 40 skipped.
        assert "real" in text
        assert "another" in text
        assert "LOWCONF" not in text
        # Conf — среднее по двум прошедшим fil'ра токенам.
        assert abs(conf - (90 + 85) / 200) < 0.01
