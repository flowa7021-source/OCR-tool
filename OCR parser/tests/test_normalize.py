# -*- coding: utf-8 -*-
from tn_parser.normalize import (
    clean_value,
    collapse,
    is_garbage,
    is_noise_line,
    is_ocr_garbage_token,
    normalize_for_sections,
    strip_garbage_tokens,
)


def test_soft_hyphen_removed():
    assert normalize_for_sections("при\u00adём") == "приём"


def test_hyphen_break_joined():
    # «при-\nём» → «приём»
    assert normalize_for_sections("при-\nём") == "приём"
    assert normalize_for_sections("транс-\n портное") == "транспортное"


def test_multiple_spaces_collapsed():
    assert normalize_for_sections("a    b\tc") == "a b c"


def test_newlines_preserved():
    text = "строка 1\n\n\nстрока 2"
    assert normalize_for_sections(text) == "строка 1\nстрока 2"


def test_confusables_fixed_in_cyrillic_words():
    # «ИHH» (с латинскими H) → «ИНН»
    assert normalize_for_sections("ИHH 7736207543") == "ИНН 7736207543"
    # «Мapкa» (латинские a, p) → «Марка»
    assert normalize_for_sections("Мapкa: Volvo") == "Марка: Volvo"


def test_latin_only_words_preserved():
    # Чисто латинские слова не трогаем.
    assert "Volvo" in normalize_for_sections("Volvo FH")


def test_mixed_script_product_name_preserved():
    # В «Тенsar» латинский кластер «sar» окружён только латиницей — не трогаем.
    result = normalize_for_sections("Тенsar")
    # Латинская «a» (U+0061) сохранена, а не заменена на кириллическую «а».
    assert "a" in result  # ASCII 'a' (U+0061)
    assert "\u0430" not in result  # cyrillic 'а'


def test_collapse_joins_lines():
    assert collapse("a\nb\nc") == "a b c"


def test_is_garbage():
    assert is_garbage("")
    assert is_garbage("...---???")
    assert not is_garbage("Ромашка")
    assert not is_garbage("12345")


def test_clean_value():
    assert clean_value("", "MISS", "GRB") == "MISS"
    assert clean_value("  Ромашка :,  ", "MISS", "GRB") == "Ромашка"
    assert clean_value("---???", "MISS", "GRB") == "GRB"


# --- OCR garbage token filters --------------------------------------------


class TestIsOcrGarbageToken:
    def test_ukrainian_letters_are_garbage(self):
        # Украинские і, ї, є, ґ не бывают в русских ТН.
        assert is_ocr_garbage_token("іі-і")
        assert is_ocr_garbage_token("Ц:і")
        assert is_ocr_garbage_token("іЁт")
        assert is_ocr_garbage_token("ав4'і")

    def test_embedded_quotes_are_garbage(self):
        # Апостроф/кавычка внутри слова = OCR-артефакт.
        assert is_ocr_garbage_token("Бекам'тбд")
        assert is_ocr_garbage_token("'Г'чп")
        assert is_ocr_garbage_token("ав4'")

    def test_exotic_diacritic_is_garbage(self):
        assert is_ocr_garbage_token("Mоscowé")
        assert is_ocr_garbage_token("ïюнь")

    def test_mixed_script_many_runs_is_garbage(self):
        # 3+ переключений кирилл↔латин в одном слове.
        assert is_ocr_garbage_token("аBсD")   # 4 runs
        assert is_ocr_garbage_token("МaМbМ")  # 5 runs

    def test_random_punctuation_is_garbage(self):
        assert is_ocr_garbage_token("'|_'")
        assert is_ocr_garbage_token("'*!\"'\"")

    def test_legitimate_tokens_are_not_garbage(self):
        # Чистая кириллица и латиница.
        assert not is_ocr_garbage_token("Москва")
        assert not is_ocr_garbage_token("RENAULT")
        assert not is_ocr_garbage_token("ООО")
        # Артикулы/коды.
        assert not is_ocr_garbage_token("B30F300")
        assert not is_ocr_garbage_token("7743553262")
        # Легитимная смесь: «Тенsar» (2 runs: Тен + sar).
        assert not is_ocr_garbage_token("Тенsar")
        # Разделители.
        assert not is_ocr_garbage_token("—")
        assert not is_ocr_garbage_token("-")
        assert not is_ocr_garbage_token("...")
        assert not is_ocr_garbage_token("———")

    def test_empty_or_none(self):
        assert is_ocr_garbage_token("")


class TestIsNoiseLine:
    def test_fully_noisy_line(self):
        assert is_noise_line("'|_' г-.: іі-і\"\" '*!\"'\" 'Г'чп'-'-\\і' Ц:і'\\ 1'] іЁт'\\ РРДВО")

    def test_clean_address_line_is_not_noise(self):
        assert not is_noise_line(
            'ООО "Бекам", 125212, г. Москва, ул. Адмирала Макарова, ИНН 7743553262'
        )

    def test_mixed_with_few_garbage_tokens_is_not_noise(self):
        # Один мусорный токен среди нормальных — не считается шумом.
        assert not is_noise_line("Москва і Санкт-Петербург и Казань")


class TestStripGarbageTokens:
    def test_removes_ukrainian_and_quoted_tokens(self):
        dirty = "ООО \"Бекам\" іі-і Бекам'тбд Москва"
        cleaned = strip_garbage_tokens(dirty)
        assert "іі-і" not in cleaned
        assert "Бекам'тбд" not in cleaned
        assert "Москва" in cleaned
        assert "ООО" in cleaned

    def test_keeps_em_dash_separator(self):
        assert "—" in strip_garbage_tokens("Дата — 23.07.2022")

    def test_empty_returns_empty(self):
        assert strip_garbage_tokens("") == ""
