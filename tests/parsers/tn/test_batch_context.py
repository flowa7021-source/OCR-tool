"""Тесты batch document-profile learning (idea #3 top-10).

Сценарий: бухгалтер заливает пакет 10 ТН от одного грузоотправителя.
Первые 3 OCR'ятся хорошо (ИНН/КПП извлечены); оставшиеся 7 страдают
— размытая печать, рукописные правки, низкий контраст.

Текущий парсер обрабатывает каждый документ независимо:
  * doc1: shipper="ООО ГЕКСАФОРМ СПБ", inn="7813266190" (conf 1.0)
  * doc2: shipper="ООО ГЕКСАФОРМ" (short-fail), inn=None (OCR потерял)
  → parser даёт doc2.shipper_inn=None, confidence=0.0 ⇒ в Excel
    «жёлтое/красное» поле хотя legitimately этот ИНН в том же батче
    уже известен.

Решение — **BatchContext**: объект, накапливающий high-confidence
shippers / consignees по мере обработки документов в батче.
Второй+ документ проверяет через context: если raw-name / частичные
реквизиты near-match одного из seen_orgs → заполняем недостающие
ИНН/КПП/ОГРН из уже-verified записи.

TDD-план:

1. ``BatchContext.add_row(row)`` — запоминает shipper/consignee из
   high-conf row (overall ≥ 0.7).
2. ``BatchContext.apply_learning(row)`` — если у row пустой
   shipper_inn И name fuzzy-match'ит один из seen_shippers,
   заполняет inn/kpp/ogrn из seen. Note: ``learned_from_batch``.
3. Integration в orchestrator: ``extract_from_pages(pages, config,
   source_path, batch_ctx=None)`` — caller передаёт общий
   BatchContext, orchestrator после extract'а вызывает
   ``batch_ctx.add_row(row)`` + ``apply_learning(row)``.
"""

from __future__ import annotations


class TestBatchContextAddRow:
    """BatchContext запоминает только high-confidence shipper /
    consignee для последующего learning'а."""

    def _ctx(self):
        from src.tn_parser.batch_context import BatchContext
        return BatchContext()

    def test_high_conf_row_added_to_seen_shippers(self):
        ctx = self._ctx()
        high_conf_row = {
            "shipper": "ООО ГЕКСАФОРМ СПБ",
            "shipper_inn": "7813266190",
            "shipper_kpp": "",
            "shipper_ogrn": "",
            "confidence": {"shipper": 1.0, "shipper_inn": 1.0},
            "overall_confidence": 0.9,
        }
        ctx.add_row(high_conf_row)
        assert len(ctx.seen_shippers) == 1
        assert ctx.seen_shippers[0]["inn"] == "7813266190"

    def test_low_conf_row_not_added(self):
        ctx = self._ctx()
        low_conf_row = {
            "shipper": "ООО ???",
            "shipper_inn": "",
            "overall_confidence": 0.3,
        }
        ctx.add_row(low_conf_row)
        assert len(ctx.seen_shippers) == 0

    def test_both_sides_tracked_separately(self):
        ctx = self._ctx()
        row = {
            "shipper": "ООО SHIP",
            "shipper_inn": "7701234567",
            "consignee": "ООО CONS",
            "consignee_inn": "7709876543",
            "overall_confidence": 0.9,
        }
        ctx.add_row(row)
        assert len(ctx.seen_shippers) == 1
        assert len(ctx.seen_consignees) == 1
        assert ctx.seen_shippers[0]["inn"] == "7701234567"
        assert ctx.seen_consignees[0]["inn"] == "7709876543"


class TestApplyLearning:
    """Второй+ документ в батче заполняет пустые реквизиты из
    уже seen'ов."""

    def _ctx_with_seen(self):
        from src.tn_parser.batch_context import BatchContext
        ctx = BatchContext()
        ctx.add_row({
            "shipper": "ООО ГЕКСАФОРМ СПБ",
            "shipper_inn": "7813266190",
            "shipper_kpp": "781301001",
            "shipper_ogrn": "1137847123456",
            "overall_confidence": 0.95,
        })
        return ctx

    def test_missing_inn_filled_from_near_match(self):
        """Text в doc2 содержит похожее имя, но ИНН OCR потерял.
        Expect: shipper_inn заполнен из seen."""
        ctx = self._ctx_with_seen()
        new_row = {
            "shipper": "ООО ГЕКСАФОРМ СГБ",  # typo СПБ→СГБ, no INN
            "shipper_inn": "",
            "shipper_kpp": "",
            "shipper_ogrn": "",
            "overall_confidence": 0.5,
        }
        ctx.apply_learning(new_row)
        assert new_row["shipper_inn"] == "7813266190"
        assert new_row["shipper_kpp"] == "781301001"
        assert new_row["shipper_ogrn"] == "1137847123456"
        assert "learned_from_batch" in (new_row.get("note") or "")

    def test_already_filled_not_overwritten(self):
        """Если inn уже есть — НЕ перезаписываем, даже если в seen
        другое значение. Primary-extraction priority."""
        ctx = self._ctx_with_seen()
        new_row = {
            "shipper": "ООО ГЕКСАФОРМ СПБ",
            "shipper_inn": "9999999999",  # чужой ИНН, но уже stable
            "overall_confidence": 0.7,
        }
        ctx.apply_learning(new_row)
        # Не трогаем:
        assert new_row["shipper_inn"] == "9999999999"

    def test_name_mismatch_skips_learning(self):
        """Имя явно другое → не применяем."""
        ctx = self._ctx_with_seen()
        new_row = {
            "shipper": "ООО РОМАШКА",
            "shipper_inn": "",
            "overall_confidence": 0.7,
        }
        ctx.apply_learning(new_row)
        assert new_row["shipper_inn"] == ""

    def test_no_seen_rows_noop(self):
        """Первый doc в батче — нет seen, apply_learning не-op."""
        from src.tn_parser.batch_context import BatchContext
        ctx = BatchContext()
        new_row = {
            "shipper": "ООО ГЕКСАФОРМ",
            "shipper_inn": "",
            "overall_confidence": 0.5,
        }
        ctx.apply_learning(new_row)
        assert new_row.get("shipper_inn") == ""
