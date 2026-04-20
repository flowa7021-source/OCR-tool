# -*- coding: utf-8 -*-
from pathlib import Path

from tn_parser.normalize import normalize_for_sections
from tn_parser.sections import split_sections


FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return normalize_for_sections(
        (FIXTURES / name).read_text(encoding="utf-8")
    )


def test_standard_all_roles():
    sec = split_sections(_load("tn_standard.txt"))
    assert "head" in sec and "Транспортная накладная" in sec["head"]
    for role in ("shipper", "consignee", "cargo", "carrier", "vehicle", "reception"):
        assert role in sec, f"Роль {role} не найдена"
    assert "Ромашка" in sec["shipper"]
    assert "Василёк" in sec["consignee"]
    assert "Мука" in sec["cargo"]
    assert "Быстрые Перевозки" in sec["carrier"]
    assert "А123ВС777" in sec["vehicle"]


def test_real_sample_numbering_variant():
    """Старая редакция ТН: перевозчик = раздел 6, ТС = 7, приём = 8."""
    sec = split_sections(_load("tn_real_7145B.txt"))
    assert "Бекам" in sec["shipper"]
    assert "Моспроект" in sec["consignee"]
    assert "Блок облицовочный" in sec["cargo"] or "Наименование" in sec["cargo"]
    assert "Самовывоз" in sec["carrier"]
    assert "RENAULT" in sec["vehicle"] or "Р 814" in sec["vehicle"]
    assert "Подолино" in sec["reception"]
    # Следующие разделы не должны утекать в приём груза.
    assert "Переадресовка" not in sec["reception"]


def test_messy_recovers_by_name():
    sec = split_sections(_load("tn_messy.txt"))
    assert "Сидоров" in sec["shipper"]
    assert "ТрансЛайн" in sec["carrier"]
    assert "vehicle" in sec


def test_empty_text():
    assert split_sections("") == {}


def test_head_without_markers():
    text = "Просто текст без номеров разделов"
    sec = split_sections(text)
    assert sec == {"head": text}


# --- fuzzy-классификация заголовков (rapidfuzz) ---------------------------


class TestFuzzyTitles:
    """OCR искажает названия разделов произвольным образом. Секционер
    через rapidfuzz должен относить заголовки с edit-distance ≤ ~20%
    к правильной роли — без ручного перечисления каждого варианта."""

    def _classify(self, title: str):
        from tn_parser.sections import _classify_title
        return _classify_title(title)

    def test_shipper_ocr_variants(self):
        for s in ("Грузоотправитель", "Грузоптправитель",
                  "Гпузоотправитель", "Грузоотпрапитель"):
            assert self._classify(s) == "shipper", s

    def test_consignee_ocr_variants(self):
        for s in ("Грузополучатель", "ГРузопалуцатель",
                  "Гпузополучатель", "Грузопопучатель"):
            assert self._classify(s) == "consignee", s

    def test_shipper_synonyms_from_invoice(self):
        # В счёте-фактуре / УПД роль «грузоотправитель» называется
        # «Продавец» или «Поставщик». OCR-варианты допустимы.
        for s in ("Продавец", "Поставщик", "Продавец:", "Прадавец",
                  "Поставщнк"):
            assert self._classify(s) == "shipper", s

    def test_consignee_synonyms_from_invoice(self):
        # «Покупатель» → consignee.
        for s in ("Покупатель", "Покупатель:", "Пакупатель", "Покупател"):
            assert self._classify(s) == "consignee", s

    def test_carrier_ocr_variants(self):
        for s in ("Перевозчик", "Перевозчип", "Перевазчик", "Пёревозчик"):
            assert self._classify(s) == "carrier", s

    def test_vehicle_ocr_variants(self):
        for s in ("Транспортное средство", "Транспортнпое средство",
                  "Транспортное средстро"):
            assert self._classify(s) == "vehicle", s

    def test_reception_ocr_variants(self):
        for s in ("Приём груза", "Прием груза", "Прйём груза", "Приом груза"):
            assert self._classify(s) == "reception", s

    def test_cargo_only_for_short_word(self):
        assert self._classify("Груз") == "cargo"
        assert self._classify("Груз:") == "cargo"
        assert self._classify("Груз — Блок") == "cargo"

    def test_no_false_positives_for_gruz_compounds(self):
        # «Грузоподъёмность», «Грузооборот» НЕ должны классифицироваться
        # как cargo (это пометки внутри карточки ТС, не заголовки графы).
        for s in ("Грузоподъёмность", "Грузооборот", "Грузопоток"):
            assert self._classify(s) is None, s

    def test_no_false_positives_for_random_text(self):
        for s in ("Какая-то чушь", "12345", "Adresat",
                  "Стоимость перевозки"):  # последнее — __ignored__
            res = self._classify(s)
            assert res in (None, "__ignored__"), (s, res)
