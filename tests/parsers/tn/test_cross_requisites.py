"""Тесты cross-field requisites consistency (idea #2 top-10).

Проблема: ``shipper_inn`` / ``shipper_kpp`` / ``shipper_ogrn``
извлекаются из одной строки независимо. Если OCR сбил разбивку
блоков (shipper-block OCR'улся в conjunction с consignee-block),
парсер может склеить ИНН одной org + КПП другой → тихий
consolidated record с wrong mapping.

Подход:

  1. ``check_inn_ogrn_compatibility(inn, ogrn)`` — структурная
     проверка: длина ОГРН (13 → ЮЛ, 15 → ИП) coherent с длиной
     ИНН (10 → ЮЛ, 12 → ИП/физлицо).

  2. ``check_catalog_consistency(inn, kpp, ogrn, catalog)`` —
     если catalog знает про ИНН, проверяем что наши KPP/OGRN
     совпадают с каталожными. Mismatch → confidence drop.

  3. ``cross_check_requisites(inn, kpp, ogrn, catalog=None)`` —
     главный entry point; возвращает (ok, note, confidence_delta).

Интеграция: в :func:`core._build_row` после extraction 6 полей
(shipper_* + consignee_*) вызываем cross_check и:

  * При mismatch — confidence shipper_inn/_kpp/_ogrn выставляется
    не выше 0.5 (ставить 1.0 на заведомо противоречивых данных
    нельзя — пользователь увидит «зелёную» ячейку на неверных
    реквизитах).
  * ``note`` поле получает marker «inn_kpp_mismatch» или
    «catalog_conflict».
"""

from __future__ import annotations


class TestInnOgrnStructuralCompatibility:
    """Структурные инварианты (без каталога):

    * ОГРН 13 digit (ЮЛ) + ИНН 10 digit (ЮЛ) — OK
    * ОГРН 15 digit (ИП/физлицо) + ИНН 12 digit (ИП) — OK
    * ОГРН 13 + ИНН 12 — mismatch: ЮЛ не может иметь 12-знач ИНН
    * ОГРН 15 + ИНН 10 — mismatch: ИП не может иметь 10-знач ИНН
    """

    def _call(self, inn: str, ogrn: str) -> bool:
        from src.tn_parser.cross_requisites import check_inn_ogrn_compatibility
        return check_inn_ogrn_compatibility(inn, ogrn)

    def test_legal_entity_pair_ok(self):
        assert self._call("7701234567", "5137746157490") is True  # ЮЛ+ЮЛ

    def test_individual_entrepreneur_pair_ok(self):
        assert self._call("123456789012", "123456789012345") is True  # ИП+ИП

    def test_legal_ogrn_with_individual_inn_mismatch(self):
        assert self._call("123456789012", "5137746157490") is False

    def test_individual_ogrn_with_legal_inn_mismatch(self):
        assert self._call("7701234567", "123456789012345") is False

    def test_either_empty_returns_true(self):
        """Пустые поля — нечего проверять, возвращаем True."""
        assert self._call("", "5137746157490") is True
        assert self._call("7701234567", "") is True
        assert self._call("", "") is True

    def test_invalid_length_returns_true_pass_through(self):
        """Не наше дело валидировать длины — этим занимаются
        ``is_valid_inn`` / ``is_valid_ogrn``. Здесь просто пропускаем."""
        assert self._call("abc", "xyz") is True


class TestCatalogConsistency:
    """Если catalog знает про ИНН — проверяем matched КПП/ОГРН."""

    def _call(self, inn, kpp, ogrn, catalog):
        from src.tn_parser.cross_requisites import check_catalog_consistency
        return check_catalog_consistency(inn, kpp, ogrn, catalog)

    def test_no_catalog_returns_ok(self):
        """Без каталога — ничего не проверяем, все OK."""
        result = self._call("7701234567", "770101001", "", None)
        assert result.ok is True
        assert result.note == ""

    def test_catalog_match_returns_ok(self):
        """Catalog: {inn: {kpp, ogrn}}. Совпадает → OK."""
        catalog = {"7701234567": {"kpp": "770101001", "ogrn": "5137746157490"}}
        result = self._call("7701234567", "770101001", "5137746157490", catalog)
        assert result.ok is True

    def test_catalog_kpp_mismatch_flagged(self):
        """KPP в документе отличается от каталожного — mismatch."""
        catalog = {"7701234567": {"kpp": "770101001"}}
        result = self._call("7701234567", "999999999", "", catalog)
        assert result.ok is False
        assert "kpp" in result.note.lower() or "кпп" in result.note.lower()

    def test_inn_not_in_catalog_passes_through(self):
        """Новый ИНН, которого нет в каталоге → pass через, не
        mismatch (просто нечего сверять)."""
        catalog = {"7701234567": {"kpp": "770101001"}}
        result = self._call("7709876543", "770901001", "", catalog)
        assert result.ok is True


class TestCrossCheckRequisites:
    """Главный entry point. Аgreggates structural + catalog checks."""

    def _call(self, inn, kpp, ogrn, catalog=None):
        from src.tn_parser.cross_requisites import cross_check_requisites
        return cross_check_requisites(inn, kpp, ogrn, catalog)

    def test_empty_returns_passthrough(self):
        result = self._call("", "", "")
        assert result.ok is True
        assert result.confidence_delta == 0.0

    def test_coherent_legal_entity_passes(self):
        catalog = {"7701234567": {"kpp": "770101001"}}
        result = self._call("7701234567", "770101001", "5137746157490", catalog)
        assert result.ok is True
        assert result.confidence_delta >= 0.0

    def test_structural_mismatch_lowers_confidence(self):
        """ОГРН 13 + ИНН 12 — невозможная комбинация. Conf drops."""
        result = self._call("123456789012", "", "5137746157490")
        assert result.ok is False
        assert result.confidence_delta < 0.0

    def test_catalog_mismatch_lowers_confidence(self):
        catalog = {"7701234567": {"kpp": "770101001"}}
        result = self._call("7701234567", "999999999", "", catalog)
        assert result.ok is False
        assert result.confidence_delta < 0.0


# Интеграция в core._build_row — отдельный коммит после того как
# catalog получит mapping inn→(kpp, ogrn). Сейчас DocCatalog хранит
# плоские сеты идентификаторов; full mapping требует extension
# либо отдельного ORG_REGISTRY. Заводим под #3 batch-context.
