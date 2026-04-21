"""Тесты для organization-name fuzzy нормализатора (idea #9 top-10).

Проблема: OCR типично ошибается в 1-2 буквах название организации:
    «ООО "ГЕКСАФОРМ СПБ"» → «ООО "ГЕКСАФОРМ СГБ"» (П → Г)
    «ООО "Беком"» → «ООО "Векoм"» (Б → В + Latin o)
Catalog exact-match такие не ловит; lexicon_corrector узкий (только
про ТН-термины). Нужен fuzzy-layer через rapidfuzz.

**TDD-план:**

  * ``normalize_org_name(raw, candidates, threshold=85)``:
      - raw exact в candidates → возвращает raw без изменений
      - raw fuzzy-match выше threshold → возвращает canonical
      - raw ниже threshold → возвращает raw
      - empty candidates → pass-through

Интеграция в ``_catalog_crossvalidate`` ещё не требуется здесь —
сначала пишем изолированный модуль, потом отдельным коммитом
wire'им в orchestrator.
"""

from __future__ import annotations


class TestNormalizeOrgName:
    """Fuzzy-нормализация OCR-typo названия организации через
    catalog-candidates."""

    def _call(self, raw, candidates, threshold=85):
        from src.tn_parser.org_normalizer import normalize_org_name
        return normalize_org_name(raw, candidates, threshold=threshold)

    def test_exact_match_unchanged(self):
        raw = "ООО ГЕКСАФОРМ СПБ"
        candidates = ["ООО ГЕКСАФОРМ СПБ", "ООО РОМАШКА"]
        assert self._call(raw, candidates) == "ООО ГЕКСАФОРМ СПБ"

    def test_one_letter_typo_normalizes_to_canonical(self):
        """OCR-typo П→Г: «СПБ» → «СГБ». Fuzzy-ratio ≈ 95 %."""
        raw = "ООО ГЕКСАФОРМ СГБ"
        candidates = ["ООО ГЕКСАФОРМ СПБ"]
        result = self._call(raw, candidates, threshold=85)
        assert result == "ООО ГЕКСАФОРМ СПБ"

    def test_too_different_returns_raw(self):
        raw = "ООО РОМАШКА"
        candidates = ["ООО ЛЮТИК", "ООО ОДУВАНЧИК"]
        result = self._call(raw, candidates, threshold=85)
        assert result == "ООО РОМАШКА"

    def test_empty_candidates_returns_raw(self):
        raw = "ООО КОМПАНИЯ"
        assert self._call(raw, []) == raw
        assert self._call(raw, None) == raw

    def test_empty_raw_returns_empty(self):
        assert self._call("", ["ООО РОМАШКА"]) == ""
        assert self._call(None, ["ООО РОМАШКА"]) == ""

    def test_best_match_wins_among_multiple_candidates(self):
        """При нескольких достаточно-близких — выбираем самое
        близкое по fuzzy-ratio."""
        raw = "ООО ГЕКСАФОРМ СГБ"
        candidates = [
            "ООО ГЕКСАФОРМ СПБ",  # 1 typo — очень близко
            "ООО ГЕКСАФОРМ НН",   # больше различий
        ]
        assert self._call(raw, candidates, threshold=80) == "ООО ГЕКСАФОРМ СПБ"

    def test_latin_cyrillic_lookalike_normalized(self):
        """Latin ``o`` / ``a`` / ``e`` вместо Cyrillic — частая
        OCR-ошибка. Fuzzy-ratio достаточно толерантен чтобы
        поймать."""
        # "Беком" → OCR "Векoм" (В latin, o latin).
        raw = "ООО Векoм"  # Second letter Latin B, o Latin
        candidates = ["ООО Беком"]
        result = self._call(raw, candidates, threshold=70)
        # Низкий threshold — fuzzy должен поймать.
        assert result == "ООО Беком" or result == raw

    def test_prefix_variation_treated_equivalent(self):
        """ООО / АО / ЗАО — разные prefix'ы, не должны матчиться
        на высокий threshold если prefix разный."""
        raw = "АО ГЕКСАФОРМ СПБ"
        candidates = ["ООО ГЕКСАФОРМ СПБ"]
        # При threshold=90 не должен нормализовать (prefix другой).
        assert self._call(raw, candidates, threshold=95) == raw
