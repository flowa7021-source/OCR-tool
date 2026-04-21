"""Тесты для form-aware cargo table parser (idea #7 top-10).

Контекст — Bug 4 класс:

    Раздел 4 «Груз» в ТН-форме (Приложение № 4 к ПП РФ № 2200)
    имеет ТАБЛИЧНУЮ структуру с колонками::

        № п/п | Наименование | Кол-во мест | Масса нетто | Класс
              | груза        |             | / брутто     | опасн
        ------+--------------+-------------+-------------+------
          1   | Георешетка   |    129      | 7.3095 т    |  —
              | TENSAR       |             | 7.3095 т    |
              ...

    OCR флэттенит таблицу в плоский text (столбцы склеиваются
    через разделители табуляции/|/пробелы). Existing regex-based
    extract_cargo применяет эвристики для очистки, но сами
    «наименование» vs «класс опасности» vs «упаковка» неразделимы
    без понимания табличной структуры.

    На реальном UPD_36 cargo после Bug 4 fix'а даёт:
        «тм. TENSAR/cxors»
    Ожидаемо же — «Георешетка TENSAR» (только колонка «Наименование»,
    без leading-маркировки «тм.» и без trailing-upakovka).

**TDD-план:**

1. ``detect_table_structure(body)`` — эвристика «это таблица?».
   Признак: ≥ 3 строк с одинаковым кол-вом разделителей
   (``|``/``\\t``/multi-space), либо наличие «№ п/п» header'а.

2. ``parse_cargo_table(body)`` — разбивает строки на колонки,
   идентифицирует колонку Наименование по header'у или по
   позиции (первая содержательная колонка после «№»).

3. ``extract_name_from_table(body)`` — возвращает конкатенацию
   значений столбца Наименование (одного / нескольких строк).

Интеграция в ``fields.extract_cargo``:

    1. Сначала пробуем extract_cargo как раньше (regex-way).
    2. Параллельно (если detected табличная структура) пробуем
       parse_cargo_table.
    3. Сравниваем кандидаты по качеству: table-way лучше если
       regex-way содержит label-слова соседних колонок (класс
       опасности, упаковка, бирка, ярлык, способ).
"""

from __future__ import annotations


class TestDetectTableStructure:
    """``detect_table_structure(body)`` возвращает True если в body
    обнаружена табличная структура (≥ 2 строк с col-separator'ами).
    """

    def _call(self, body: str) -> bool:
        from src.tn_parser.cargo_table import detect_table_structure
        return detect_table_structure(body)

    def test_pipe_separated_rows_detected_as_table(self):
        body = (
            "№ | Наименование | Кол-во | Масса\n"
            "1 | Георешетка TENSAR | 66 рул | 2805 кг\n"
            "2 | Георешетка TriAx | 63 рул | 4504 кг\n"
        )
        assert self._call(body) is True

    def test_flat_text_not_detected_as_table(self):
        body = (
            "Георешетка TENSAR, 129 мест, масса нетто 7.3095 т, "
            "объем 57.948 м³\n"
        )
        assert self._call(body) is False

    def test_single_row_with_pipes_not_enough(self):
        body = "1 | Георешетка | 66 | 2805"
        assert self._call(body) is False

    def test_tab_separated_rows_also_detected(self):
        body = (
            "№\tНаименование\tКол-во\tМасса\n"
            "1\tГеорешетка TENSAR\t66 рул\t2805 кг\n"
            "2\tГеорешетка TriAx\t63 рул\t4504 кг\n"
        )
        assert self._call(body) is True

    def test_empty_body_returns_false(self):
        assert self._call("") is False


class TestParseCargoTable:
    """``parse_cargo_table(body)`` → list of dict rows."""

    def _call(self, body: str):
        from src.tn_parser.cargo_table import parse_cargo_table
        return parse_cargo_table(body)

    def test_simple_pipe_table_produces_rows(self):
        body = (
            "№ | Наименование | Кол-во | Масса\n"
            "1 | Георешетка TENSAR | 66 рул | 2805 кг\n"
            "2 | Георешетка TriAx | 63 рул | 4504 кг\n"
        )
        rows = self._call(body)
        assert len(rows) == 2
        assert rows[0]["Наименование"] == "Георешетка TENSAR"
        assert rows[1]["Наименование"] == "Георешетка TriAx"

    def test_header_normalised_case_insensitive(self):
        """Header может быть uppercase / mixed — нормализуем."""
        body = (
            "№ П/П | НАИМЕНОВАНИЕ | КОЛ-ВО | МАССА\n"
            "1 | Груз A | 10 | 100\n"
        )
        rows = self._call(body)
        # Ключи канонические (нижний регистр) ИЛИ точные header'ы —
        # главное чтобы «Наименование»-колонка нашлась. Проверим
        # через case-insensitive lookup:
        row = rows[0]
        name_value = next(
            (v for k, v in row.items() if "наимен" in k.lower()),
            None,
        )
        assert name_value == "Груз A"


class TestExtractNameFromTable:
    """``extract_name_from_table(body)`` — удобная обёртка:
    возвращает join'ed строку по колонке Наименование."""

    def _call(self, body: str) -> str:
        from src.tn_parser.cargo_table import extract_name_from_table
        return extract_name_from_table(body)

    def test_multi_row_table_joins_names(self):
        body = (
            "№ | Наименование | Кол-во\n"
            "1 | Георешетка TENSAR | 66\n"
            "2 | Георешетка TriAx | 63\n"
        )
        result = self._call(body)
        assert "TENSAR" in result
        assert "TriAx" in result

    def test_no_table_returns_empty_string(self):
        body = "просто текст без структуры"
        assert self._call(body) == ""

    def test_table_without_name_column_returns_empty(self):
        """Таблица есть, но колонки «Наименование» в header'е нет."""
        body = (
            "№ | Кол-во | Масса\n"
            "1 | 66 | 2805\n"
            "2 | 63 | 4504\n"
        )
        assert self._call(body) == ""


class TestExtractCargoIntegration:
    """``fields.extract_cargo`` должен предпочесть table-way если
    она даёт более чистое имя, чем regex-way с label-leak'ами."""

    def test_regex_way_still_works_on_flat_text(self):
        """Если input не табличный — extract_cargo возвращает
        regex-результат как раньше (backward compat)."""
        from src.tn_parser.fields import extract_cargo

        body = "Наименование: Георешетка TENSAR\nКол-во: 129 мест"
        name, conf = extract_cargo(body, body)
        assert "TENSAR" in name
        assert conf > 0

    def test_table_way_wins_when_regex_yields_label_leak(self):
        """Input — таблица; regex-way без изменений забрал бы
        label-слова соседних колонок. Table-way должен дать чистое
        имя и override regex-результата.

        Настоящая мотивация: на реальном UPD_36 мы получаем
        «тм. TENSAR/cxors» — flat-text без явной табличной
        структуры. Но для случаев где таблица чёткая, table-path
        должен давать лучший результат.
        """
        from src.tn_parser.fields import extract_cargo

        body = (
            "№ п/п | Наименование | Кол-во мест | Масса нетто | Класс опасности | Упаковка\n"
            "1 | Георешетка TENSAR | 129 | 7.3095 т | неопасный | рулон, стянут пп лентой\n"
        )
        name, conf = extract_cargo(body, body)
        # Name — только колонка Наименование, без «неопасный»,
        # «рулон», «пп лентой».
        assert "TENSAR" in name
        assert "неопасный" not in name.lower()
        assert "рулон" not in name.lower()
        assert "пп лентой" not in name.lower()
        assert conf > 0
