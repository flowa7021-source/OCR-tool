"""Tests for the domain language model and OCR post-correction."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.shared.domain_lm import DomainLM


def _make_lm(words: list[str]) -> DomainLM:
    return DomainLM(Counter(words))


class TestBuildFromFiles:
    def test_from_default_corpus_loads_real_data(self):
        lm = DomainLM.from_default_corpus()
        # Should have hundreds of domain words.
        assert len(lm) > 100
        # Sanity: known words from our corpus are present.
        assert "гексаформ" in lm or "моспроект-3" in lm or "упд" in lm

    def test_from_custom_files(self, tmp_path: Path):
        txt = tmp_path / "doc.txt"
        txt.write_text("Товар накладная перевозчик перевозчик", encoding="utf-8")
        js = tmp_path / "doc.json"
        js.write_text(json.dumps({
            "parties": [{"name": "ГЕКСАФОРМ", "inn": "7813266190"}],
            "items": ["Товар А"],
        }, ensure_ascii=False), encoding="utf-8")
        lm = DomainLM.from_files([txt], [js])
        assert "накладная" in lm
        assert "гексаформ" in lm
        # Duplicate counts properly — "перевозчик" twice in txt.
        assert lm.vocab["перевозчик"] == 2


class TestCorrectSkipsHighConfidence:
    def test_high_conf_ocr_kept_as_is(self):
        lm = _make_lm(["гексаформ"])
        r = lm.correct("гексафирмm", ocr_conf=0.95)
        assert not r.was_corrected
        assert r.word == "гексафирмm"

    def test_vocab_hit_boosts_confidence(self):
        lm = _make_lm(["гексаформ"] * 100 + ["ооо"] * 50)
        r = lm.correct("гексаформ", ocr_conf=0.5)
        assert not r.was_corrected
        assert r.word == "гексаформ"
        assert r.confidence > 0.5


class TestFuzzyCorrection:
    def test_single_char_error(self):
        lm = _make_lm(["гексаформ"] * 100)
        # OCR dropped one char: "гексафрм" → should correct to "гексаформ"
        r = lm.correct("гексафрм", ocr_conf=0.3)
        assert r.was_corrected
        assert r.word == "гексаформ"
        assert r.edit_distance == 1

    def test_transposition(self):
        lm = _make_lm(["накладная"] * 10)
        r = lm.correct("накланая", ocr_conf=0.4)
        assert r.word == "накладная"

    def test_out_of_budget_no_correction(self):
        lm = _make_lm(["гексаформ"] * 10)
        # 5 char diff — nothing close enough
        r = lm.correct("абвгдежз", ocr_conf=0.3)
        assert not r.was_corrected

    def test_frequency_wins_ties(self):
        # Two candidates both at edit distance 1 from input; the more
        # frequent one should win.
        lm = _make_lm(["один"] * 100 + ["олин"] * 5)
        r = lm.correct("един", ocr_conf=0.3)
        assert r.word == "один"

    def test_short_word_budget_is_smaller(self):
        lm = _make_lm(["код"] * 10 + ["кот"] * 10 + ["код1"] * 5)
        # "коз" is 1 edit from both "код" (0 vs д) and "кот" (1 vs т).
        # With short-word budget=1, we accept. Still has to return
        # something sensible.
        r = lm.correct("коз", ocr_conf=0.3)
        assert r.was_corrected
        assert r.word in {"код", "кот"}


class TestNumericTokensSkipped:
    def test_inn_not_fuzzy_matched(self):
        lm = _make_lm(["гексаформ"] * 100)
        # A 10-digit ИНН must not get fuzzy-corrected to a word.
        r = lm.correct("7813266190", ocr_conf=0.5)
        assert not r.was_corrected
        assert r.word == "7813266190"

    def test_date_left_alone(self):
        lm = _make_lm(["гексаформ"] * 100)
        r = lm.correct("02.09.2022", ocr_conf=0.4)
        assert not r.was_corrected
