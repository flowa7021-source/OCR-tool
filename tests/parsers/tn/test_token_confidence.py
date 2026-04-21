"""Тесты token-level confidence propagation (idea #5 top-10).

Проблема: парсер выставляет дискретную confidence (0.0, 0.5, 0.7,
0.9, 1.0). Если ИНН извлечён из low-conf токенов (OCR не уверен
в цифрах), но checksum случайно сошёлся — confidence остаётся 1.0.
В Excel ячейка зелёная, пользователь доверяет, а фактически OCR
в этих цифрах был не уверен.

**Подход**: к каждому value мы можем привязать OCR-confidence из
:class:`TokenConfMap` — (start_idx, end_idx) → OCR conf 0.0-1.0.
Когда extract_inn / find_inn находит substring, возвращаем также
OCR-confidence этих токенов (среднее).

**TDD-план:**

  * ``TokenConfMap.for_substring(text, value) → float`` — средняя
    OCR-conf токенов в substring, или None если map пустой.
  * ``find_inn(text, token_map=None) → (inn, ocr_conf)`` —
    расширенная signature; без map возвращает (inn, 1.0).
  * Integration в core: shipper_inn_conf не ставится в 1.0
    безусловно, а берёт min(checksum_conf, ocr_conf * 1.2).
"""

from __future__ import annotations


class TestTokenConfMapForSubstring:
    """``TokenConfMap.for_substring(text, value)`` — средняя OCR-
    conf токенов substring'а, нормализованная 0..1."""

    def _map_from(self, token_conf_list):
        """Helper: собрать TokenConfMap из списка (start, end, conf)."""
        from src.tn_parser.token_confidence import TokenConfMap
        return TokenConfMap.from_ranges(token_conf_list)

    def test_single_range_covering_substring(self):
        """text='7701234567', токен 0-10 с conf=80 → 0.8."""
        text = "7701234567"
        tcmap = self._map_from([(0, 10, 80.0)])
        assert abs(tcmap.for_substring(text, "7701234567") - 0.8) < 1e-6

    def test_multi_range_averages(self):
        """text='a 7701234567 b', substring '7701234567' внутри —
        охватывается двумя токенами с разной conf."""
        text = "abc 7701234567 def"
        # Один токен "abc" conf=50, второй "7701234567" conf=90,
        # третий "def" conf=60.
        tcmap = self._map_from([
            (0, 3, 50.0),
            (4, 14, 90.0),
            (15, 18, 60.0),
        ])
        # Substring "7701234567" полностью в токене (4, 14) с conf=90.
        assert abs(tcmap.for_substring(text, "7701234567") - 0.9) < 1e-6

    def test_missing_substring_returns_zero(self):
        text = "abc"
        tcmap = self._map_from([(0, 3, 80.0)])
        assert tcmap.for_substring(text, "xyz") == 0.0

    def test_empty_map_returns_none(self):
        from src.tn_parser.token_confidence import TokenConfMap
        empty = TokenConfMap.from_ranges([])
        assert empty.for_substring("abc", "abc") is None


class TestFindInnWithTokenConf:
    """``find_inn(text, token_map=None)`` extension — возвращает
    optional ocr_conf."""

    def test_without_map_returns_inn_only(self):
        """Backward compat: без token_map возвращается только inn
        (существующая signature)."""
        from src.tn_parser.validators import find_inn
        inn = find_inn("ИНН 7701234567")
        # Старая signature — возврат str или None.
        # 7701234567 has checksum — check if valid:
        # Не проверяем валидность тут, только сигнатуру.
        assert isinstance(inn, (str, type(None)))

    def test_with_map_returns_tuple(self):
        from src.tn_parser.token_confidence import TokenConfMap
        from src.tn_parser.validators import find_inn_with_conf

        text = "ИНН 7707820890"
        # 7707820890 — валидный ИНН (Мозэнерго).
        tcmap = TokenConfMap.from_ranges([(4, 14, 75.0)])
        inn, ocr_conf = find_inn_with_conf(text, tcmap)
        assert inn == "7707820890"
        assert abs(ocr_conf - 0.75) < 1e-6

    def test_no_inn_returns_none_and_zero(self):
        from src.tn_parser.token_confidence import TokenConfMap
        from src.tn_parser.validators import find_inn_with_conf

        inn, conf = find_inn_with_conf("no numbers here", TokenConfMap.from_ranges([]))
        assert inn is None
        assert conf == 0.0
