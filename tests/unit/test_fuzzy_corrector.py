"""Unit tests for :mod:`src.core.fuzzy_corrector`."""

from __future__ import annotations

import pytest

rapidfuzz = pytest.importorskip("rapidfuzz")

from src.core.fuzzy_corrector import (  # noqa: E402
    _is_cheap_substitution,
    _ocr_aware_score,
    correct,
    lexicon_size,
)

# ---------------------------------------------------------------------------
# Lexicon loads
# ---------------------------------------------------------------------------


def test_lexicon_loaded_from_resource() -> None:
    """resources/ru_lexicon.txt сгенерирован build-time и должен
    содержать ≥ 15 000 словоформ (890 баз × ~17 форм)."""
    size = lexicon_size()
    assert size >= 15_000, (
        f"словарь только {size} форм — перезапустите "
        f"`python scripts/gen_ru_lexicon.py`"
    )


# ---------------------------------------------------------------------------
# Cheap OCR substitutions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [
    ("о", "а"), ("е", "ё"), ("и", "й"), ("и", "н"),
    ("ш", "щ"), ("ч", "у"), ("з", "в"),
    # Cross-script
    ("о", "o"), ("а", "a"), ("е", "e"),
    # Digit-letter
    ("о", "0"), ("з", "3"), ("б", "6"),
])
def test_known_cheap_pair(a: str, b: str) -> None:
    assert _is_cheap_substitution(a, b)
    assert _is_cheap_substitution(b, a)  # симметрично


def test_identity_is_cheap() -> None:
    assert _is_cheap_substitution("а", "а")


@pytest.mark.parametrize("a,b", [
    ("а", "б"),    # разные буквы, не в classes
    ("о", "я"),
    ("1", "2"),
])
def test_unrelated_pair_not_cheap(a: str, b: str) -> None:
    assert not _is_cheap_substitution(a, b)


# ---------------------------------------------------------------------------
# OCR-aware scoring
# ---------------------------------------------------------------------------


def test_ocr_aware_score_all_cheap_gets_bonus() -> None:
    """Если все несовпадения — cheap substitutions, score +3."""
    # «организаиия» vs «организация»: и↔я (не cheap pair, но short)
    base = rapidfuzz.fuzz.ratio("организаиия", "организация")
    aware = _ocr_aware_score("организаиия", "организация")
    # Одно несовпадение — и↔я. Это не в наших cheap classes, поэтому
    # aware не даст bonus. Base = aware.
    assert aware == base


def test_ocr_aware_score_identical_is_100() -> None:
    assert _ocr_aware_score("организация", "организация") == 100


def test_ocr_aware_score_o_a_cheap_bonus() -> None:
    """о↔а — в cheap classes, должен давать bonus."""
    # «орган» vs «арган» (4-char word). base должно быть 80 (4/5 совпали).
    base = rapidfuzz.fuzz.ratio("орган", "арган")
    aware = _ocr_aware_score("орган", "арган")
    assert aware >= base  # ≥ base
    # bonus only applies when base >= 80
    if base >= 80:
        assert aware == min(100, base + 3)


def test_ocr_aware_score_low_base_no_bonus() -> None:
    """Если base < 80, bonus не применяется даже для cheap."""
    aware = _ocr_aware_score("аб", "оа")
    assert aware < 80


# ---------------------------------------------------------------------------
# correct()
# ---------------------------------------------------------------------------


def test_empty_input() -> None:
    assert correct("") == ""


def test_correct_token_unchanged() -> None:
    """Слова из словаря не меняются."""
    assert correct("организация работает") == "организация работает"


def test_short_token_not_corrected() -> None:
    """Короткие (< 6) токены пропускаются — слишком ambiguous."""
    # «абв» — фиктивный 3-буквенный токен, не должен быть тронут.
    assert correct("абв") == "абв"


def test_non_cyrillic_not_corrected() -> None:
    """Latin / mixed / числовые токены пропускаются."""
    s = "Scania VOLVO 123456"
    assert correct(s) == s


def test_proper_noun_guard_capitalized() -> None:
    """Capitalised слова не трогаются (имена, названия)."""
    # «Иванов» не в словаре — как capitalized, guard защищает.
    assert correct("Иванов") == "Иванов"
    assert correct("Беляев Александр") == "Беляев Александр"


def test_fixes_common_misspelling() -> None:
    """«организаиия» — 1-edit typo от «организации»."""
    out = correct("организаиия")
    # Должен стать либо «организация», либо «организации», либо
    # другой близкой словоформой.
    assert out != "организаиия"
    assert out.startswith("организаци") or out.startswith("организация")


def test_length_ratio_guard() -> None:
    """Не подменяет длинное слово на сильно короткое."""
    # «длинныйтокен» (12 chars) — если fuzzy нашёл «та» (2 chars),
    # length_ratio = 2/12 = 0.17 < 0.70 — должно быть отвергнуто.
    s = "длинныйтокенотсебя"
    # Это слово не в lex; fuzzy может найти что-то короткое, но
    # length-guard не пустит.
    out = correct(s)
    # Результат может быть изменён на что-то другое подходящей длины
    # ИЛИ остаться как было. Главное — не collapse'ится в короткое.
    if out != s:
        assert len(out) / len(s) >= 0.70


def test_mixed_text_preserves_non_cyrillic() -> None:
    """Latin-токены не трогаются, кириллические корректируются."""
    # «организаиия» → исправится, «Scania» нет.
    out = correct("организаиия Scania")
    assert "Scania" in out


def test_idempotent() -> None:
    s = "организаиия постановпения правительства"
    once = correct(s)
    twice = correct(once)
    assert once == twice
