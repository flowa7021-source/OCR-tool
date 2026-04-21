"""Тесты pymorphy3-based validator для fuzzy-correction candidates.

Цель: защитить fuzzy_corrector от false-positive замен. Когда
fuzzy-match предлагает candidate, проверяем через pymorphy3:
candidate должен быть ``is_known`` (реальное русское слово в
morphological dictionary).

  * Если оригинал неизвестен + candidate известен → заменяем (typo fix)
  * Если оригинал известен + candidate известен → оставляем original
    (обе формы legitimate, fuzzy может выдать ложный suggest)
  * Если pymorphy3 недоступен → fallback на старое поведение

TDD-план:

  1. ``is_known_russian_word(word)`` → bool
  2. ``should_accept_correction(original, candidate)`` →
     bool (logic как выше)
  3. Integration: fuzzy_corrector.correct с pymorphy-gate.
"""

from __future__ import annotations

import pytest

try:
    import pymorphy3  # noqa: F401

    HAS_PYMORPHY = True
except ImportError:
    HAS_PYMORPHY = False


@pytest.mark.skipif(
    not HAS_PYMORPHY, reason="pymorphy3 не установлен"
)
class TestIsKnownRussianWord:
    """``is_known_russian_word(word)`` → True если pymorphy3
    распознаёт слово как реальное русское."""

    def _call(self, word):
        from src.core.morphology_validator import is_known_russian_word
        return is_known_russian_word(word)

    def test_common_word_known(self):
        assert self._call("договор") is True
        assert self._call("транспорт") is True

    def test_ocr_typo_unknown(self):
        """«Экземиляр» — OCR typo для «Экземпляр», pymorphy3 не знает."""
        assert self._call("экземиляр") is False
        assert self._call("являетси") is False  # typo для «является»

    def test_empty_word_returns_false(self):
        assert self._call("") is False


@pytest.mark.skipif(
    not HAS_PYMORPHY, reason="pymorphy3 не установлен"
)
class TestShouldAcceptCorrection:
    """``should_accept_correction(original, candidate)`` применяет
    3 правила:

      1. Оба unknown (typo → typo) → False (candidate не помог).
      2. Оба known → False (мы не знаем что настоящая форма).
      3. Original unknown + candidate known → True (typo fix).
    """

    def _call(self, original, candidate):
        from src.core.morphology_validator import should_accept_correction
        return should_accept_correction(original, candidate)

    def test_typo_to_known_word_accepted(self):
        """экземиляр (unknown) → экземпляр (known) → accept."""
        assert self._call("экземиляр", "экземпляр") is True

    def test_known_to_known_rejected(self):
        """организация → организации: обе legitimate, consrvative no."""
        assert self._call("организация", "организации") is False

    def test_typo_to_typo_rejected(self):
        """Оба unknown — candidate не legitimate."""
        assert self._call("кабалжабалага", "кабилжабелегу") is False

    def test_empty_candidate_rejected(self):
        assert self._call("foo", "") is False
        assert self._call("foo", None) is False
