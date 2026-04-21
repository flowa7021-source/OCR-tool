"""Тесты для geometric layout-anchor'а (idea #1).

Контекст — Bug 1 класс «adjacent-cell leak»:

    Форма ТН (Приложение № 4 к Постановлению № 2200) имеет
    двух-колончатую разметку:

        +----------------------+----------------------+
        | 1. Грузоотправитель  | 1а Заказчик услуг    |
        |   ООО "X", ИНН ...   |   (если есть)        |
        +----------------------+----------------------+
        | 2. Грузополучатель   | 2а Реквизиты         |
        | ...                  |   (лицо принимающее) |
        +----------------------+----------------------+

    Regex-based split (``sections.split_sections``) смотрит только
    на текст и ищет якоря «1. Грузоотправитель» / «1а Заказчик».
    Когда OCR сбивает порядок строк (тексты обеих колонок идут
    через переносы строк), regex ставит первым найденным якорем
    СОСЕДНЮЮ ячейку и извлекает shipper из неё.

    Реальный артефакт на UPD_36: shipper='являетси экспелитопом а,
    Заказ (заявка), Дата | м |, 1a Заказчик услут по' (= содержимое
    ячейки 1а) вместо ячейки 1.

**TDD-план:**

    1. Unit: ``find_tokens_by_column(tsv_data, column="left")``
       возвращает токены где x-center < page_width / 2.
    2. Unit: ``find_section_region(tsv, anchor_text, prefer_column)``
       — поиск bbox начала секции, приоритет колонки.
    3. Integration: ``sections.split_sections`` подхватывает
       опциональный ``tsv_data`` аргумент и использует layout-
       аware вариант первым, regex — fallback'ом.

Почему через ``image_to_data`` TSV а не hOCR:

    pytesseract.image_to_data уже вызывается в pipeline для
    вычисления per-word confidence (PageResult.mean_confidence).
    Переиспользуем TSV — не вводим новый OCR-call. hOCR-XML также
    валиден в принципе, но он не прокидывается в PageResult и
    потребовал бы больших изменений в models.py.
"""

from __future__ import annotations


# Мок TSV в формате pytesseract.image_to_data(output_type=DICT):
# dict с массивами равной длины N для каждой TSV-колонки.
def _make_tsv(words: list[dict]) -> dict[str, list]:
    """Упаковать список dict'ов в pytesseract-TSV формат.

    Дефолтные значения имитируют реальный image_to_data вывод:
    level=5 (word-level), conf=90 (уверенное слово) — чтобы не
    дублировать эти поля в каждом mock-токене.
    """
    defaults = {
        "level": 5, "page_num": 1, "block_num": 1, "par_num": 1,
        "line_num": 1, "word_num": 1, "conf": 90,
    }
    keys = ("level", "page_num", "block_num", "par_num", "line_num",
            "word_num", "left", "top", "width", "height", "conf", "text")
    tsv: dict[str, list] = {k: [] for k in keys}
    for w in words:
        for k in keys:
            default = defaults.get(k, 0 if k != "text" else "")
            tsv[k].append(w.get(k, default))
    return tsv


class TestFindTokensByColumn:
    """``find_tokens_by_column(tsv, column, page_width)`` возвращает
    токены чьи x-центры попадают в выбранную колонку."""

    def _call(self, tsv, column, page_width=1000):
        from src.tn_parser.layout_anchor import find_tokens_by_column
        return find_tokens_by_column(tsv, column=column, page_width=page_width)

    def test_left_column_returns_tokens_with_center_below_half(self):
        tsv = _make_tsv([
            {"left": 50,  "top": 100, "width": 100, "height": 20, "text": "LEFT"},
            {"left": 700, "top": 100, "width": 100, "height": 20, "text": "RIGHT"},
        ])
        tokens = self._call(tsv, "left", page_width=1000)
        texts = [t["text"] for t in tokens]
        assert "LEFT" in texts
        assert "RIGHT" not in texts

    def test_right_column_returns_tokens_with_center_above_half(self):
        tsv = _make_tsv([
            {"left": 50,  "top": 100, "width": 100, "height": 20, "text": "LEFT"},
            {"left": 700, "top": 100, "width": 100, "height": 20, "text": "RIGHT"},
        ])
        tokens = self._call(tsv, "right", page_width=1000)
        texts = [t["text"] for t in tokens]
        assert "RIGHT" in texts
        assert "LEFT" not in texts

    def test_token_spanning_column_boundary_goes_to_majority_side(self):
        """Token шириной 200 px с центром 510 (page_width=1000) —
        правая колонка (510 > 500)."""
        tsv = _make_tsv([
            {"left": 410, "top": 100, "width": 200, "height": 20, "text": "SPAN"},
        ])
        assert "SPAN" not in [t["text"] for t in self._call(tsv, "left", 1000)]
        assert "SPAN" in [t["text"] for t in self._call(tsv, "right", 1000)]

    def test_empty_tsv_returns_empty_list(self):
        tsv = _make_tsv([])
        assert self._call(tsv, "left") == []
        assert self._call(tsv, "right") == []

    def test_skips_empty_text_tokens(self):
        """Tesseract часто выдаёт whitespace-only или conf=-1 токены —
        эти НЕ должны возвращаться в результате."""
        tsv = _make_tsv([
            {"left": 50, "top": 100, "width": 100, "height": 20,
             "text": "", "conf": -1},
            {"left": 50, "top": 130, "width": 100, "height": 20,
             "text": "REAL", "conf": 90},
        ])
        tokens = self._call(tsv, "left", page_width=1000)
        assert [t["text"] for t in tokens] == ["REAL"]


class TestFindSectionRegion:
    """``find_section_region(tsv, anchor_pattern, prefer_column)``
    возвращает (top_y, bottom_y) вертикальный range секции, или
    None если якорь не найден в указанной колонке."""

    def _call(self, tsv, anchor, column, page_width=1000):
        from src.tn_parser.layout_anchor import find_section_region
        return find_section_region(
            tsv, anchor_pattern=anchor,
            prefer_column=column, page_width=page_width,
        )

    def test_anchor_in_left_column_returns_bbox(self):
        """«Грузоотправитель» в левой колонке (x=100) — вернуть bbox."""
        tsv = _make_tsv([
            {"left": 100, "top": 200, "width": 200, "height": 30,
             "text": "Грузоотправитель", "conf": 90},
            {"left": 700, "top": 200, "width": 200, "height": 30,
             "text": "Заказчик", "conf": 90},
        ])
        result = self._call(tsv, r"грузоотправ", column="left", page_width=1000)
        assert result is not None
        top, bottom = result
        # Якорь на y=200, полоса обычно включает эту строку.
        assert top <= 200 <= bottom

    def test_anchor_prefers_left_when_both_columns_have_it(self):
        """OCR может видеть «Грузоотправитель» в обеих колонках
        (например, если правая содержит «3. Грузоотправитель
        перевозчика»). Приоритет — prefer_column."""
        tsv = _make_tsv([
            # В правой — сейчас попадёт shipper-мусор раньше левого.
            {"left": 700, "top": 100, "width": 200, "height": 30,
             "text": "Грузоотправитель", "conf": 90},
            {"left": 100, "top": 400, "width": 200, "height": 30,
             "text": "Грузоотправитель", "conf": 90},
        ])
        left_result = self._call(tsv, r"грузоотправ", "left", 1000)
        right_result = self._call(tsv, r"грузоотправ", "right", 1000)
        # Левый якорь — на y=400; правый — на y=100. find_section_region
        # с prefer_column="left" возвращает левый.
        assert left_result is not None and left_result[0] >= 300
        assert right_result is not None and right_result[0] <= 200

    def test_anchor_not_in_preferred_column_returns_none(self):
        """Якорь есть только справа, а просили слева → None."""
        tsv = _make_tsv([
            {"left": 700, "top": 100, "width": 200, "height": 30,
             "text": "Заказчик", "conf": 90},
        ])
        result = self._call(tsv, r"заказчик", "left", 1000)
        assert result is None

    def test_missing_anchor_returns_none(self):
        tsv = _make_tsv([
            {"left": 100, "top": 100, "width": 200, "height": 30,
             "text": "РАНДОМ", "conf": 90},
        ])
        result = self._call(tsv, r"грузоотправ", "left", 1000)
        assert result is None


class TestExtractColumnText:
    """``extract_column_text_between(tsv, y_top, y_bottom, column)`` —
    склейка токенов выбранной колонки в вертикальном диапазоне."""

    def _call(self, tsv, top, bottom, column, page_width=1000):
        from src.tn_parser.layout_anchor import extract_column_text_between
        return extract_column_text_between(
            tsv, y_top=top, y_bottom=bottom,
            column=column, page_width=page_width,
        )

    def test_extracts_only_left_column_tokens_in_range(self):
        tsv = _make_tsv([
            {"left": 100, "top": 200, "width": 200, "height": 20,
             "text": "shipper", "conf": 90},
            {"left": 100, "top": 220, "width": 200, "height": 20,
             "text": "inn", "conf": 90},
            # Вне диапазона по y.
            {"left": 100, "top": 400, "width": 200, "height": 20,
             "text": "outside", "conf": 90},
            # Правая колонка — должна быть исключена даже в диапазоне.
            {"left": 700, "top": 200, "width": 200, "height": 20,
             "text": "customer", "conf": 90},
        ])
        text = self._call(tsv, top=180, bottom=260, column="left")
        assert "shipper" in text
        assert "inn" in text
        assert "customer" not in text
        assert "outside" not in text

    def test_preserves_left_to_right_top_to_bottom_order(self):
        """Сохраняет естественный читательский порядок —
        сверху-вниз, внутри строки слева-направо."""
        tsv = _make_tsv([
            {"left": 300, "top": 200, "width": 80, "height": 20,
             "text": "SECOND", "conf": 90},
            {"left": 100, "top": 200, "width": 80, "height": 20,
             "text": "FIRST", "conf": 90},
            {"left": 100, "top": 230, "width": 80, "height": 20,
             "text": "THIRD", "conf": 90},
        ])
        text = self._call(tsv, top=180, bottom=260, column="left", page_width=1000)
        # Ожидаем "FIRST SECOND\nTHIRD" или подобное — чёткий
        # y-sort перед x-sort.
        fi = text.index("FIRST")
        se = text.index("SECOND")
        th = text.index("THIRD")
        assert fi < se < th


class TestSectionsIntegration:
    """``sections.split_sections(text, tsv=...)`` использует layout
    когда TSV доступен, иначе regex-путь без изменений."""

    def test_split_sections_without_tsv_is_backward_compatible(self):
        """Старый вызов без tsv — не регрессируется, всё работает
        как раньше через regex."""
        from src.tn_parser.sections import split_sections

        text = (
            "1. Грузоотправитель\n"
            "ООО «ТЕСТ», ИНН 7701234567\n"
            "2. Грузополучатель\n"
            "ООО «ПОЛУЧ», ИНН 7709876543\n"
        )
        sections = split_sections(text)
        assert "shipper" in sections or "Грузоотправитель" in str(sections)

    def test_split_sections_with_tsv_accepts_optional_arg(self):
        """Минимальный smoke: передача tsv=None не ломает старый API."""
        from src.tn_parser.sections import split_sections

        text = "1. Грузоотправитель\nООО «ТЕСТ»\n"
        sections = split_sections(text, tsv=None)
        # Не падает, возвращает dict.
        assert isinstance(sections, dict)

    def test_geometry_disambiguates_adjacent_cell_leak(self):
        """Настоящая интеграция: OCR-текст где регекс ошибочно
        склеивает shipper с ячейкой ``1а Заказчик``. Передача TSV
        + page_width позволяет убрать leak.

        Input: ``text`` — плоский OCR-текст, куда попали обе
        ячейки ТН-формы. TSV содержит bbox: «Грузоотправитель» в
        left-column (x=100), «Заказчик» в right-column (x=700).

        Ожидание: ``sections['shipper']`` НЕ содержит слово
        "Заказчик" если layout disambig работает.
        """
        from src.tn_parser.sections import split_sections

        # Плоский text — как будто OCR вывел обе колонки подряд.
        text = (
            "1. Грузоотправитель\n"
            "ООО «РОМАШКА», ИНН 7701234567\n"
            "1а Заказчик услуг по перевозке\n"
            "ООО «ЛЮТИК», ИНН 7709876543\n"
            "2. Грузополучатель\n"
            "ООО «ПОЛУЧ»\n"
        )
        tsv = _make_tsv([
            {"left": 100, "top": 100, "width": 400, "height": 25,
             "text": "Грузоотправитель"},
            {"left": 100, "top": 130, "width": 400, "height": 25,
             "text": "ООО"},
            {"left": 180, "top": 130, "width": 200, "height": 25,
             "text": "РОМАШКА"},
            # Заказчик — в правой колонке (x=700).
            {"left": 700, "top": 100, "width": 200, "height": 25,
             "text": "Заказчик"},
            {"left": 700, "top": 130, "width": 400, "height": 25,
             "text": "ООО"},
            {"left": 780, "top": 130, "width": 200, "height": 25,
             "text": "ЛЮТИК"},
            {"left": 100, "top": 250, "width": 400, "height": 25,
             "text": "Грузополучатель"},
        ])
        sections = split_sections(text, tsv=tsv, page_width=1200)

        # TDD goal: shipper должен содержать РОМАШКА но НЕ ЛЮТИК.
        shipper = sections.get("shipper", "").lower()
        assert "ромашка" in shipper, (
            f"shipper должен содержать 'ромашка', got: {shipper!r}"
        )
        assert "лютик" not in shipper, (
            f"shipper НЕ должен включать содержимое '1а Заказчик' ячейки, "
            f"got: {shipper!r}"
        )
