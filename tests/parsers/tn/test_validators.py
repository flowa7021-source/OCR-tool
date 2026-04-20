from src.tn_parser.validators import (
    find_grz,
    find_inn,
    is_valid_date,
    is_valid_grz,
    is_valid_inn,
)


class TestDate:
    def test_valid(self):
        # Для ТН валидным считаем только «живой» диапазон: последние
        # 10 лет + ближайший год. Архивы и далёкое будущее — OCR-шум.
        import datetime as _dt
        cur = _dt.date.today().year
        assert is_valid_date(f"15.03.{cur}")
        assert is_valid_date(f"15.03.{cur - 9}")
        assert is_valid_date("29.02.2024")  # leap year внутри диапазона

    def test_invalid(self):
        import datetime as _dt
        cur = _dt.date.today().year
        assert not is_valid_date("")
        assert not is_valid_date("32.01.2024")
        assert not is_valid_date("15.13.2024")
        assert not is_valid_date("29.02.2023")  # not leap
        assert not is_valid_date("15.03.1800")
        assert not is_valid_date("15/03/2024")
        assert not is_valid_date("abc")
        # Вне «живого» диапазона ТН:
        assert not is_valid_date("01.01.2000")
        assert not is_valid_date(f"01.01.{cur - 20}")
        assert not is_valid_date(f"01.01.{cur + 5}")


class TestGRZ:
    def test_valid_main(self):
        assert is_valid_grz("А123ВС777")
        assert is_valid_grz("А123ВС 77")
        assert is_valid_grz("О001ОО199")

    def test_valid_trailer(self):
        assert is_valid_grz("ВЕ2468 50")
        assert is_valid_grz("АА123450")

    def test_invalid(self):
        assert not is_valid_grz("")
        assert not is_valid_grz("Z123ВС777")  # не из алфавита ГОСТа
        assert not is_valid_grz("А12ВС77")

    def test_find_grz_in_text(self):
        text = "Транспортное средство: Volvo FH, гос.номер А123ВС777, ИНН 1234"
        assert find_grz(text) == "А123ВС777"

    def test_find_grz_none(self):
        assert find_grz("нет номера") is None


class TestINN:
    def test_valid_10(self):
        # Известный ИНН (Яндекс) — валидная контрольная сумма.
        assert is_valid_inn("7736207543")

    def test_valid_12(self):
        # Сгенерированный валидный 12-значный.
        assert is_valid_inn("500100732259")

    def test_invalid(self):
        assert not is_valid_inn("")
        assert not is_valid_inn("1234567890")  # случайные цифры
        assert not is_valid_inn("7736207544")  # сломанная контрольная
        assert not is_valid_inn("abcdefghij")
        assert not is_valid_inn("12345")

    def test_find_inn(self):
        text = "ИНН 7736207543, КПП 770101001"
        assert find_inn(text) == "7736207543"

    def test_find_inn_skips_bad(self):
        text = "Код 1234567890 и ИНН 7736207543"
        assert find_inn(text) == "7736207543"
