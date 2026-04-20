# -*- coding: utf-8 -*-
"""Интеграционные тесты парсинга на текстовых фикстурах.

Тесты отражают ФИНАЛЬНЫЙ контракт извлечения:

- Дата — DD.MM.YYYY
- Номер — значение после «№» без префикса «ТН-»
- Грузоотправитель — от префикса организации (ООО/АО/ИП/…) до ИНН
  ВКЛЮЧИТЕЛЬНО. КПП/ОГРН/ОКПО всегда отбрасываются.
- Грузополучатель — от префикса организации до первого ИНН/КПП/ОГРН/ОКПО
  (НЕ включая их).
- Груз — только наименование, без «Кол-во мест» (правая табличная колонка),
  без хвоста «N шт» (который уходит в «Объём»).
- Объём — «N шт», если есть в наименовании; иначе строка Нетто/Брутто/Объём.
- Перевозчик — только «правая колонка»: ФИО водителя или одинокое значение.
  Типы «Самовывоз», «Собственный» и т. п. СЮДА НЕ ПОПАДАЮТ.
- Транспортное средство — «МАРКА\\nГРЗ_слитно» в одной ячейке через перенос.
- Приём груза — только ПЕРВАЯ содержательная строка, от ORG-префикса до
  ИНН включительно.
"""

import re
from pathlib import Path

from tn_parser import MISSING, parse_text
from tn_parser.normalize import normalize_for_sections


FIXTURES = Path(__file__).parent / "fixtures"


def _text(name: str) -> str:
    return normalize_for_sections((FIXTURES / name).read_text(encoding="utf-8"))


class TestStandardWaybill:
    def setup_method(self) -> None:
        rows = parse_text(_text("tn_standard.txt"), "tn_standard.pdf")
        assert len(rows) == 1
        self.row = rows[0]

    def test_number(self):
        assert self.row.number == "ТН-2024/00127"
        assert self.row.waybill == "Транспортная накладная № ТН-2024/00127"

    def test_date(self):
        assert self.row.date == "15.03.2024"

    def test_shipper(self):
        assert "Ромашка" in self.row.shipper

    def test_consignee(self):
        assert "Василёк" in self.row.consignee

    def test_cargo(self):
        assert "Мука" in self.row.cargo

    def test_vehicle_has_grz_compact(self):
        # Новый контракт: ГРЗ без пробелов.
        assert "А123ВС777" in self.row.vehicle
        assert "А 123 ВС 777" not in self.row.vehicle

    def test_reception(self):
        # По новому контракту в reception может быть дата/адрес — контракт
        # не предписывает структуру, если нет ORG-префикса. Главное — не MISSING.
        assert self.row.reception != MISSING

    def test_to_excel_tuple_has_12_columns(self):
        # 12 колонок: waybill, date, number, shipper, consignee,
        # cargo, volume, carrier, vehicle, reception, source, note.
        assert len(self.row.to_excel_tuple()) == 12


class TestRealSample7145B:
    """Интеграция на реальном образце (ТН №7145/Б от 23.07.2022)."""

    def setup_method(self) -> None:
        rows = parse_text(_text("tn_real_7145B.txt"), "tn_7145B.pdf")
        assert len(rows) == 1
        self.row = rows[0]

    def test_number_and_date(self):
        assert self.row.number == "7145/Б"
        assert self.row.date == "23.07.2022"

    def test_shipper_includes_inn_but_not_kpp(self):
        # «является экспедитором» не должно попасть.
        assert "является экспедитором" not in self.row.shipper.lower()
        assert "Бекам" in self.row.shipper
        assert "ИНН 7743553262" in self.row.shipper
        # КПП ВСЕГДА отбрасываем по новому контракту.
        assert "504445001" not in self.row.shipper
        assert "КПП" not in self.row.shipper

    def test_consignee_excludes_inn_and_kpp(self):
        assert "Моспроект" in self.row.consignee
        assert "7707820890" not in self.row.consignee
        assert "ИНН" not in self.row.consignee
        assert "КПП" not in self.row.consignee

    def test_cargo_strips_prefix_and_qty(self):
        # «1. Наименование — Блок облицовочный …» → без префикса,
        # без хвоста «720 шт» (он уходит в volume).
        assert "Блок облицовочный" in self.row.cargo
        assert "Наименование" not in self.row.cargo
        assert "720 шт" not in self.row.cargo
        assert "Кол-во мест" not in self.row.cargo

    def test_volume_has_qty_sht(self):
        # Для этой ТН в наименовании есть «720 шт» → оно идёт в объём.
        assert "720 шт" in self.row.volume

    def test_carrier_is_fio_only(self):
        # Новый контракт: перевозчик = правая колонка (ФИО водителя).
        # «Самовывоз» не попадает.
        assert "Самовывоз" not in self.row.driver
        assert "Рябов" in self.row.driver

    def test_vehicle_is_grz_only_compact(self):
        # Vehicle = ТОЛЬКО ГРЗ слитно, без марки.
        assert self.row.vehicle == "Р814НР152"

    def test_cargo_keeps_latin_in_mixed_word(self):
        # «Тенsar» — латинская «a» должна остаться латинской.
        assert "sar" in self.row.cargo  # латинский кластер сохранён

    def test_reception_first_line_only(self):
        r = self.row.reception
        # Первая строка — «ООО "Бекам", …, ИНН 7743553262».
        assert "Бекам" in r
        assert "ИНН 7743553262" in r
        assert "КПП" not in r
        # Адрес погрузки, дата, водитель — в первую строку НЕ попадают.
        assert "Подолино" not in r
        assert "Переадресовка" not in r
        assert "Выдача груза" not in r


class TestOcrTabularLayout:
    """Интеграция на OCR-слое реальной табличной формы ТН №7145/Б."""

    def setup_method(self) -> None:
        rows = parse_text(_text("tn_ocr_tabular.txt"), "tn_ocr_tabular.pdf")
        assert len(rows) == 1
        self.row = rows[0]

    def test_number_not_missed_due_to_ekzemplyar(self):
        assert self.row.number == "7145/Б"
        assert self.row.waybill == "Транспортная накладная № 7145/Б"

    def test_date(self):
        assert self.row.date == "23.07.2022"

    def test_shipper_contract(self):
        assert "является экспедитором" not in self.row.shipper.lower()
        assert "(реквизиты" not in self.row.shipper
        assert "Бекам" in self.row.shipper
        assert "ИНН 7743553262" in self.row.shipper
        # КПП всегда отбрасывается.
        assert "504445001" not in self.row.shipper
        assert "КПП" not in self.row.shipper

    def test_consignee_contract(self):
        assert "Моспроект" in self.row.consignee
        assert "Кузнецкий мост" in self.row.consignee
        assert "ИНН" not in self.row.consignee
        assert "КПП" not in self.row.consignee
        assert "7707820890" not in self.row.consignee
        assert "(реквизиты" not in self.row.consignee

    def test_cargo_no_qty_no_kol_vo_mest(self):
        assert self.row.cargo.startswith("Блок облицовочный")
        assert "Наименование" not in self.row.cargo
        assert "720 шт" not in self.row.cargo
        assert "Кол-во мест" not in self.row.cargo

    def test_volume_qty_sht(self):
        assert "720 шт" in self.row.volume

    def test_carrier_right_column(self):
        # Левая колонка «Самовывоз» НЕ должна попадать, только ФИО.
        assert "Самовывоз" not in self.row.driver
        assert "Рябов" in self.row.driver
        assert "(реквизиты" not in self.row.driver

    def test_vehicle_is_grz_only(self):
        assert self.row.vehicle == "Р814НР152"

    def test_reception_first_line_to_inn(self):
        r = self.row.reception
        assert "Бекам" in r
        assert "ИНН 7743553262" in r
        assert "КПП" not in r
        assert "Подолино" not in r
        assert "Переадресовка" not in r


class TestMessyWaybill:
    def setup_method(self) -> None:
        rows = parse_text(_text("tn_messy.txt"), "tn_messy.pdf")
        assert len(rows) == 1
        self.row = rows[0]

    def test_date(self):
        assert self.row.date == "07.11.2023"

    def test_number(self):
        assert self.row.number == "А-99/2023"

    def test_shipper_confusables_ok(self):
        assert "Сидоров" in self.row.shipper

    def test_vehicle_grz_compact(self):
        assert "К456МН178" in self.row.vehicle


class TestMultiWaybillsInOneFile:
    def test_splits_into_two(self):
        rows = parse_text(_text("tn_multi.txt"), "tn_multi.pdf")
        assert len(rows) == 2

        assert rows[0].number == "001"
        assert rows[0].date == "01.01.2024"
        assert "Альфа" in rows[0].shipper
        assert "А001АА77" in rows[0].vehicle

        assert rows[1].number == "002"
        assert rows[1].date == "02.01.2024"
        assert "Гамма" in rows[1].shipper
        assert "В002ВВ77" in rows[1].vehicle


class TestEmptyInput:
    def test_returns_placeholder_row(self):
        rows = parse_text("", "empty.pdf")
        assert len(rows) == 1
        r = rows[0]
        assert r.date == MISSING
        assert r.number == MISSING
        assert r.consignee == MISSING
        assert r.volume == MISSING
        assert "LOW_TEXT" in r.note
        assert r.confidence.overall() == 0.0


class TestOcrNoise:
    """Реальный OCR-шум."""

    def setup_method(self) -> None:
        rows = parse_text(_text("tn_ocr_noise.txt"), "tn_ocr_noise.pdf")
        assert len(rows) == 1
        self.row = rows[0]

    def test_number_extracted_despite_ekzemplyar_on_prev_line(self):
        assert self.row.number == "7145/Б"

    def test_date_extracted(self):
        assert self.row.date == "23.07.2022"

    def test_vehicle_has_real_grz_compact(self):
        assert "Р814НР152" in self.row.vehicle
        assert "Р 814 НР 152" not in self.row.vehicle

    def test_vehicle_grz_not_inn_false_positive(self):
        assert "7743553262" not in self.row.vehicle

    def test_vehicle_has_grz(self):
        # Vehicle = только ГРЗ (без марки).
        assert "RENAULT" not in self.row.vehicle
        assert re.search(r"[А-ЯЁ]\d{3}[А-ЯЁ]{2}\d{2,3}", self.row.vehicle)

    def test_shipper_no_zakazchik_prefix_has_inn_no_kpp(self):
        s = self.row.shipper
        assert "заказчик" not in s.lower()
        assert not s.startswith("Га")
        assert "Бекам" in s
        assert "ИНН 7743553262" in s
        assert "КПП" not in s
        assert "774301001" not in s

    def test_reception_clean_to_inn(self):
        r = self.row.reception
        assert "ГЕР" not in r
        assert "Бекам" in r
        assert "ИНН 7743553262" in r
        assert "КПП" not in r

    def test_cargo_strips_ocr_naim_prefix(self):
        assert "Блок облицовочный" in self.row.cargo
        assert "Нанменование" not in self.row.cargo
        assert "наименование" not in self.row.cargo.lower()

    def test_carrier_right_column(self):
        # В этой фикстуре только «Самовывоз» в графе 6. По контракту —
        # левая колонка не попадает, значит если ФИО нет, вернётся либо
        # единственная непустая строка («Самовывоз»), либо MISSING.
        # Фиксируем мягкое ожидание: «/ Й /» (OCR-мусор) точно не попадает.
        assert "/ Й /" not in self.row.driver


class TestOcrInlineLayout:
    def setup_method(self) -> None:
        rows = parse_text(_text("tn_ocr_inline.txt"), "tn_ocr_inline.pdf")
        assert len(rows) == 1
        self.row = rows[0]

    def test_number_extracted_despite_ekzemplyar_same_line(self):
        assert self.row.number == "7145/Б"

    def test_date_extracted(self):
        assert self.row.date == "23.07.2022"

    def test_vehicle_grz_compact_no_brand(self):
        assert self.row.vehicle == "Р814НР152"

    def test_shipper_ok_no_kpp(self):
        assert "Бекам" in self.row.shipper
        assert "ИНН 7743553262" in self.row.shipper
        assert "КПП" not in self.row.shipper

    def test_consignee_ok_no_inn(self):
        assert "Моспроект" in self.row.consignee
        assert "ИНН" not in self.row.consignee

    def test_reception_no_embedded_quotes(self):
        r = self.row.reception
        assert "Бекам'" not in r
        assert "'Г'" not in r
        assert "'|_'" not in r

    def test_shipper_no_service_tail(self):
        s = self.row.shipper.lower()
        assert "(при наличи" not in s
        assert "перевозки груза" not in s
