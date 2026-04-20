"""Unit tests for :mod:`src.core.per_word_script_disambiguator`."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.core.per_word_script_disambiguator import (
    disambiguate_word,
    is_latin_brand_suspect,
)


class TestIsLatinBrandSuspect:
    """Heuristic for "should we speculative-Latin re-OCR this token?"

    The look-alike classifier in text_postprocessor returns ``"cyr"``
    for tokens like ``Тапваг`` (Tesseract's hallucinated rendering of
    ``TENSAR`` on a Russian-dominant page) — not "mixed" — so the
    disambiguator would skip them without this extra trigger.
    """

    def test_all_caps_cyrillic_is_suspect(self) -> None:
        assert is_latin_brand_suspect("ТАПВАГ")
        assert is_latin_brand_suspect("СКАНИЯ")
        assert is_latin_brand_suspect("ИНН")

    def test_all_caps_latin_is_suspect(self) -> None:
        assert is_latin_brand_suspect("TENSAR")
        assert is_latin_brand_suspect("SCANIA")
        assert is_latin_brand_suspect("VOLVO")

    def test_mixed_case_not_suspect(self) -> None:
        """Prose tokens in mixed or lowercase are Russian body text —
        re-OCR would just waste cycles."""
        assert not is_latin_brand_suspect("Тапваг")
        assert not is_latin_brand_suspect("Scania")
        assert not is_latin_brand_suspect("contract")

    def test_too_short_not_suspect(self) -> None:
        # 2-char tokens are too short to safely apply re-OCR — they're
        # often fragments from table cells / stamp-overlay lines.
        assert not is_latin_brand_suspect("XY")
        assert not is_latin_brand_suspect("АБ")

    def test_too_long_not_suspect(self) -> None:
        # Long all-caps tokens are typically yelled prose / stamp text,
        # not brand acronyms. Skip re-OCR to avoid false positives.
        assert not is_latin_brand_suspect("ABCDEFGHIJK")  # 11 chars
        assert not is_latin_brand_suspect("АБВГДЕЖЗИКЛ")

    def test_digits_not_suspect(self) -> None:
        # Digit-mixed codes are caught by the numeric-context path in
        # the postprocessor; the disambiguator skips them here to
        # avoid double-processing.
        assert not is_latin_brand_suspect("USD123")
        assert not is_latin_brand_suspect("INV12")
        assert not is_latin_brand_suspect("1234567")

    def test_punctuation_not_suspect(self) -> None:
        assert not is_latin_brand_suspect("INV-5")
        assert not is_latin_brand_suspect("P.O.")

    def test_empty_not_suspect(self) -> None:
        assert not is_latin_brand_suspect("")

    def test_three_char_boundary(self) -> None:
        # 3 is the lower bound — inclusive.
        assert is_latin_brand_suspect("ДСК")
        assert is_latin_brand_suspect("MAN")

    def test_ten_char_boundary(self) -> None:
        # 10 is the upper bound — inclusive.
        assert is_latin_brand_suspect("TRANSINZKO")  # 10 chars
        assert is_latin_brand_suspect("ТРАНСИНЖКО")


def _fake_tsv(words: list[str], confs: list[float]) -> dict:
    """Build a pytesseract.image_to_data DICT-style payload."""
    return {"text": list(words), "conf": [str(c) for c in confs]}


class TestDisambiguateWord:
    def test_latin_wins_when_confidence_clearly_higher(self) -> None:
        """Mixed token re-OCR'd: eng gets 85 %, rus gets 60 % → pick eng."""
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)

        def _fake_image_to_data(crop, lang, config, output_type):
            if lang == "eng":
                return _fake_tsv(["INV"], [85.0])
            return _fake_tsv(["ИНV"], [60.0])

        with patch(
            "src.core.per_word_script_disambiguator.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = _fake_image_to_data
            mock_py.Output.DICT = "dict"
            word, conf = disambiguate_word(
                image, bbox,
                original_word="ИНV",
                original_confidence=55.0,
            )

        assert word == "INV"
        assert conf == 85.0

    def test_keeps_original_when_lift_too_small(self) -> None:
        """A 3-point win doesn't clear the 5-point lift floor."""
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)

        def _fake_image_to_data(crop, lang, config, output_type):
            if lang == "eng":
                return _fake_tsv(["INV"], [78.0])
            return _fake_tsv(["ИНV"], [76.0])

        with patch(
            "src.core.per_word_script_disambiguator.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = _fake_image_to_data
            mock_py.Output.DICT = "dict"
            word, conf = disambiguate_word(
                image, bbox,
                original_word="orig",
                original_confidence=75.0,
            )

        # Best alt was 78 vs original 75 — only 3 point lift, < 5 floor.
        assert word == "orig"
        assert conf == 75.0

    def test_both_langs_fail_returns_original(self) -> None:
        """No words recognised by either language — keep original."""
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)

        with patch(
            "src.core.per_word_script_disambiguator.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.return_value = _fake_tsv([], [])
            mock_py.Output.DICT = "dict"
            word, conf = disambiguate_word(
                image, bbox,
                original_word="ИНV",
                original_confidence=55.0,
            )

        assert word == "ИНV"
        assert conf == 55.0

    def test_invalid_image_returns_original(self) -> None:
        word, conf = disambiguate_word(
            "not an array", (0, 0, 10, 10),
            original_word="orig", original_confidence=50.0,
        )
        assert word == "orig"
        assert conf == 50.0

    def test_zero_size_bbox_returns_original(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        word, conf = disambiguate_word(
            image, (10, 10, 0, 0),
            original_word="orig", original_confidence=50.0,
        )
        assert word == "orig"
        assert conf == 50.0

    def test_pytesseract_exception_returns_original(self) -> None:
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)

        with patch(
            "src.core.per_word_script_disambiguator.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = RuntimeError("tesseract died")
            mock_py.Output.DICT = "dict"
            word, conf = disambiguate_word(
                image, bbox,
                original_word="ИНV",
                original_confidence=55.0,
            )

        assert word == "ИНV"
        assert conf == 55.0

    def test_bbox_padding_clamped_to_image_bounds(self) -> None:
        """A bbox touching the edge still produces a valid crop."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        bbox = (0, 0, 50, 50)

        def _fake_image_to_data(crop, lang, config, output_type):
            # Crop must be exactly the image (bbox fills it + padding
            # clamps to bounds).
            assert crop.shape == image.shape
            if lang == "eng":
                return _fake_tsv(["A"], [90.0])
            return _fake_tsv([], [])  # rus produces no words

        with patch(
            "src.core.per_word_script_disambiguator.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = _fake_image_to_data
            mock_py.Output.DICT = "dict"
            word, conf = disambiguate_word(
                image, bbox,
                original_word="orig",
                original_confidence=50.0,
            )

        # eng reported 90 — original 50, lift = 40 > 5 → accept.
        assert word == "A"
        assert conf == 90.0

    def test_rus_wins_on_russian_text(self) -> None:
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        bbox = (10, 20, 60, 30)

        def _fake_image_to_data(crop, lang, config, output_type):
            if lang == "rus":
                return _fake_tsv(["Привет"], [92.0])
            return _fake_tsv(["Пpивeт"], [70.0])  # Latin letters mixed in

        with patch(
            "src.core.per_word_script_disambiguator.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = _fake_image_to_data
            mock_py.Output.DICT = "dict"
            word, conf = disambiguate_word(
                image, bbox,
                original_word="Пpивeт",
                original_confidence=70.0,
            )

        assert word == "Привет"
        assert conf == 92.0
