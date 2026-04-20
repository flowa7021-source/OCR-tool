"""Архитектурные регрессии: парсер захватывает лишний текст до/после нужного
значения или «тянет» в поле содержимое соседней графы.

Три архетипа жалобы пользователя:

    1. suffix-leak    — после значения тянутся подписи/печати/комментарии
    2. graph confusion — OCR потерял точку в номере графы («6 Перевозчик»
                         вместо «6. Перевозчик»)
    3. inline-leak    — fallback-regex захватывает соседнюю колонку на
                         той же строке

Тесты обновлены под ФИНАЛЬНЫЙ контракт извлечения (см. test_parsing.py).
"""

from pathlib import Path

from src.tn_parser import parse_text
from src.tn_parser.normalize import normalize_for_sections

FIXTURES = Path(__file__).parent / "fixtures"


def _text(name: str) -> str:
    return normalize_for_sections((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Архетип 1: suffix-leak — подпись/печать/контакт после реквизитов
# ---------------------------------------------------------------------------


class TestSuffixLeak:
    def setup_method(self) -> None:
        rows = parse_text(_text("tn_suffix_leak.txt"), "tn_suffix_leak.pdf")
        assert len(rows) == 1
        self.row = rows[0]

    def test_shipper_stops_before_signature(self):
        s = self.row.shipper
        assert "Альфа-Металл" in s
        assert "ИНН 7801234567" in s
        # КПП всегда отбрасывается.
        assert "КПП" not in s
        assert "780101001" not in s
        # Всё после ИНН (подпись/МП) тоже не попадает.
        assert "Подпись" not in s
        assert "Петров" not in s
        assert "МП" not in s.split()

    def test_consignee_stops_before_inn_or_contact(self):
        c = self.row.consignee
        assert "Бета-Пром" in c
        # ИНН/КПП/контакт — всё за границей contract-а.
        assert "ИНН" not in c
        assert "КПП" not in c
        assert "7701098765" not in c
        assert "Контактное лицо" not in c
        assert "Сидорова" not in c

    def test_cargo_stops_at_technical_attrs(self):
        cg = self.row.cargo
        assert "Прокат стальной" in cg
        assert "Класс опасности" not in cg
        assert "Упаковка" not in cg
        assert "Кол-во мест" not in cg

    def test_carrier_right_column_only(self):
        # По новому контракту перевозчик = только правая колонка.
        car = self.row.driver
        # VOLVO — ТС из соседней графы, утечь не должно.
        assert "VOLVO" not in car
        assert "О 777" not in car


# ---------------------------------------------------------------------------
# Архетип 2: graph confusion — OCR потерял точку в номере графы
# ---------------------------------------------------------------------------


class TestGraphConfusion:
    def setup_method(self) -> None:
        rows = parse_text(_text("tn_graph_confusion.txt"), "tn_graph_confusion.pdf")
        assert len(rows) == 1
        self.row = rows[0]

    def test_carrier_extracted_despite_missing_dot(self):
        # «6 Перевозчик» без точки. ФИО/фамилия должны извлечься.
        car = self.row.driver
        assert "Захаров" in car

    def test_cargo_does_not_leak_into_carrier(self):
        cg = self.row.cargo
        assert "Картон" in cg
        assert "Захаров" not in cg
        assert "Перевозчик" not in cg

    def test_reception_extracted_despite_missing_dot(self):
        # «8 Приём груза» без точки — reception всё равно найдёт контент.
        r = self.row.reception
        assert "Баумана" in r
        assert "Переадресовка" not in r

    def test_vehicle_grz_only(self):
        v = self.row.vehicle
        assert "VOLVO" not in v
        assert "В404КМ716" in v
        assert "Приём" not in v
        assert "Баумана" not in v


# ---------------------------------------------------------------------------
# Архетип 3: inline-leak — две графы на одной строке, fallback-regex тянет
# соседнюю колонку
# ---------------------------------------------------------------------------


class TestInlineLeak:
    def setup_method(self) -> None:
        rows = parse_text(_text("tn_inline_leak.txt"), "tn_inline_leak.pdf")
        assert len(rows) == 1
        self.row = rows[0]

    def test_shipper_does_not_absorb_consignee_on_same_line(self):
        s = self.row.shipper
        assert "Омега" in s
        assert "ИНН 7712345678" in s
        assert "КПП" not in s
        # Грузополучатель не должен утечь в грузоотправителя.
        assert "Сигма" not in s
        assert "7798765432" not in s
        assert "Грузополучатель" not in s

    def test_consignee_extracted_separately(self):
        c = self.row.consignee
        assert "Сигма" in c
        # По контракту: ИНН/КПП получателя не попадают.
        assert "ИНН" not in c
        assert "7798765432" not in c
        assert "Омега" not in c
