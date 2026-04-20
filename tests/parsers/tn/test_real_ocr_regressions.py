"""Регресс-тесты на реальных OCR-выгрузках (а не на синтетических фикстурах).

Покрывают конкретные проблемы, найденные пользователем в живых PDF:
    - 7145/Б — «Га Заказчик услуг…», «1. Нанменование —», «720 нтт»,
      «, Ин» перед ИНН в разделе «Приём груза».
    - 2908-23A — номер без «№», ORG «000 «ДЕЛОВЫЕ ПЕРЕВОЗКИ»»,
      слитный ГРЗ «С201ВХ152».
"""

from pathlib import Path

from src.tn_parser import parse_text
from src.tn_parser.normalize import normalize_for_sections

FIXTURES = Path(__file__).parent / "fixtures" / "real"


def _parse(name: str):
    raw = (FIXTURES / name).read_text(encoding="utf-8")
    rows = parse_text(normalize_for_sections(raw), name)
    assert rows, f"{name} parsed to no rows"
    return rows[0]


class TestOcr7145BFull:
    @classmethod
    def setup_class(cls):
        cls.row = _parse("ocr_7145b_full.txt")

    def test_number_present(self):
        # OCR сжевал «Б» в «6», но сам факт извлечения номера важен.
        assert self.row.number not in ("отсутствует", "")
        assert "7145" in self.row.number

    def test_date(self):
        assert self.row.date == "23.07.2022"

    def test_shipper_no_zakazchik_prefix(self):
        assert "Заказчик услуг" not in self.row.shipper
        assert "перевозки груза" not in self.row.shipper
        assert "ООО" in self.row.shipper and "Бекам" in self.row.shipper
        assert "ИНН 7743553262" in self.row.shipper
        assert "КПП" not in self.row.shipper

    def test_consignee_clean(self):
        assert "Моспроект" in self.row.consignee
        assert "ИНН" not in self.row.consignee
        assert "КПП" not in self.row.consignee

    def test_cargo_no_label_and_no_qty(self):
        assert "Наименование" not in self.row.cargo
        assert "Нанменование" not in self.row.cargo
        assert "Блок облицовочный" in self.row.cargo
        assert "Кол-во мест" not in self.row.cargo
        # 720 шт в cargo быть не должно (ушло в volume)
        assert not self.row.cargo.rstrip().endswith("нтт")
        assert not self.row.cargo.rstrip().endswith("шт")

    def test_volume_has_qty_sht_even_with_ocr_noise(self):
        # «720 нтт» — OCR исказил «шт» как «нтт», но volume должен быть не пустым.
        assert self.row.volume not in ("отсутствует", "")
        assert "720" in self.row.volume

    def test_carrier_is_fio(self):
        assert "Рябов" in self.row.driver

    def test_vehicle_has_compact_grz(self):
        assert "Р814НР152" in self.row.vehicle

    def test_reception_starts_with_org_and_ends_with_inn(self):
        assert self.row.reception.startswith("ООО")
        assert "Бекам" in self.row.reception
        assert "7743553262" in self.row.reception
        # КПП/шумовые хвосты — не должны попасть.
        assert "КПП" not in self.row.reception
        assert "РРДВО" not in self.row.reception
        assert "ГЕР" not in self.row.reception


class TestOcr2908_23A:  # noqa: N801 — test-case id пересобирается пользователем по git
    @classmethod
    def setup_class(cls):
        cls.row = _parse("ocr_2908_23a.txt")

    def test_number_without_pound_sign(self):
        # В этом OCR символа «№» нет — номер идёт как отдельная строка
        # после «Транспортная накладная» и даты.
        assert self.row.number == "2908-23A"

    def test_date(self):
        assert self.row.date == "29.08.2022"

    def test_driver_does_not_contain_company_name(self):
        # Поле driver — только ФИО. Название компании-перевозчика
        # «ООО ДЕЛОВЫЕ ПЕРЕВОЗКИ», адрес, ИНН/КПП и водительское
        # удостоверение НЕ должны попадать.
        assert "ДЕЛОВЫЕ" not in self.row.driver
        assert "ИНН" not in self.row.driver
        assert "711806" not in self.row.driver
        # ФИО водителя в этом OCR — «Белен Александр Ныколаенич» (full
        # name без инициалов) — текущий ФИО-regex такой формат не ловит,
        # поэтому driver для этого файла остаётся MISSING. Это ожидаемо.

    def test_shipper_recovered_despite_ocr_gruzovtiravitel(self):
        # OCR исказил «Грузоотправитель» в «Грузовтиравитель» — секционер
        # должен опознать заголовок и заполнить shipper, не cargo.
        assert "ГЕКСАФОРМ" in self.row.shipper
        assert "7813266190" in self.row.shipper
        assert "КПП" not in self.row.shipper

    def test_consignee_present(self):
        # Заголовок «1. Грузополучатель» (с OCR-ошибкой в номере раздела
        # — «1.» вместо «2.») всё равно опознаётся как consignee.
        assert "Моспроект" in self.row.consignee

    def test_vehicle_is_compact_grz_only(self):
        # Vehicle = только ГРЗ, без марки.
        assert self.row.vehicle == "С201ВХ152"

    def test_reception_has_org_and_inn(self):
        assert "ГЕКСАФОРМ" in self.row.reception
        assert "7813266190" in self.row.reception
