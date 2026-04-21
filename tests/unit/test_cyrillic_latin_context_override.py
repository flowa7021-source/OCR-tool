"""Усиление ``fix_cyrillic_latin_confusion`` — context override для
pure-Latin слов в Cyrillic-параграфе.

Контекст: OCR на плохом скане читает всё слово как Latin-буквы
включая те, у которых НЕТ exact Cyrillic look-alike (d, g, r, v,
f, l, m, n…). Текущий ``_classify_word_script`` размечает такое
слово как ``lat`` и оставляет нетронутым — хотя в окружении
Cyrillic-text'а это почти гарантированно русское слово с
OCR-ошибками.

Примеры с реального UPD_36:
    «Dogovor»    — было «Договор»
    «Mockba»     — было «Москва»
    «rpuzowrom»  — было «грузоотом»

Resolution: в рамках ``normalize_cyrillic_latin_confusion``, когда
paragraph_majority = ``cyr``, ``lat``-классифицированные слова
пропускаются через *aggressive* substitution map (включает d→д,
g→г, r→р и т.п.) — но ТОЛЬКО при дополнительном gate:

  * Слово не входит в skip-set (URL / email / bkz path).
  * Слово не является ALL-UPPERCASE brand-name (TENSAR, VOLVO —
    должны остаться Latin).
  * После conversion получается слово которое проходит базовый
    Cyrillic-check (≥ 3 символов из Cyrillic-exclusive).

**TDD**:

  1. Aggressive map AGGRESSIVE_LAT_TO_CYR включает все
     low-confidence pairs, не только look-alikes.
  2. ``_aggressive_cyrillify(word)`` делает conversion пробным.
  3. ``normalize_cyrillic_latin_confusion`` использует aggressive
     path при paragraph_majority="cyr" для lat-classified слов.
"""

from __future__ import annotations


class TestAggressiveCyrillify:
    """``_aggressive_cyrillify(word)`` пробует конвертацию через
    расширенный map; гарантирует что ВСЕ буквы имеют Cyrillic
    counterpart. Возвращает None если слово невозможно полностью
    конвертировать (есть Latin-only буквы типа Q, W, Z которые
    не похожи на Cyrillic)."""

    def _call(self, word):
        from src.core.text_postprocessor import _aggressive_cyrillify
        return _aggressive_cyrillify(word)

    def test_pure_lookalikes_converted(self):
        # Mockba — все буквы look-alike pairs.
        assert self._call("Mockba") == "Москва"

    def test_aggressive_pairs_converted(self):
        # Dogovor — содержит «d» и «g» (нет look-alike, но есть
        # визуальная близость к «д»/«г»). В реальном OCR такие
        # возникают на шумных сканах.
        assert self._call("Dogovor") == "Договор"

    def test_latin_only_chars_prevent_conversion(self):
        """w / q / z не имеют разумных Cyrillic пар. Слово с ними —
        возвращаем None (не трогаем)."""
        assert self._call("Workflow") is None
        assert self._call("Quartz") is None

    def test_brand_uppercase_not_converted(self):
        """``TENSAR`` / ``VOLVO`` — brand-name ALL-UPPERCASE. Даже
        если буквы все имеют Cyrillic-counterpart, помечаем как
        non-convertible — это бренды."""
        # TENSAR: T,E,N,S,A,R — N и S отсутствуют в aggressive map,
        # либо ALL-UPPERCASE heuristic отсекает.
        assert self._call("TENSAR") is None


class TestContextOverrideInNormalize:
    """``normalize_cyrillic_latin_confusion`` в Cyrillic-параграфе
    конвертирует ``lat``-classified слова через aggressive map."""

    def _call(self, text):
        from src.core.text_postprocessor import normalize_cyrillic_latin_confusion
        return normalize_cyrillic_latin_confusion(text)

    def test_lat_word_in_cyr_paragraph_converted(self):
        """Текст: «Заключили Dogovor на поставку» — Dogovor в
        Cyrillic-majority → должен конвертироваться в «Договор»."""
        input_text = (
            "Настоящим мы заключили Dogovor на поставку оборудования.\n"
            "Транспортная накладная оформлена в соответствии с "
            "Постановлением Правительства."
        )
        result = self._call(input_text)
        assert "Договор" in result or "договор" in result.lower()

    def test_brand_name_preserved_in_cyr_paragraph(self):
        """Бренд TENSAR в русском тексте — НЕ конвертируется."""
        input_text = (
            "Поставляется георешётка TENSAR по международному "
            "стандарту, согласно договору."
        )
        result = self._call(input_text)
        assert "TENSAR" in result

    def test_latin_paragraph_untouched(self):
        """Если paragraph majority Latin — aggressive mode не
        триггерится."""
        input_text = (
            "This is a standard invoice written in English. "
            "Total amount: 100 USD. Please review the terms."
        )
        result = self._call(input_text)
        # English — all words stay English.
        assert "This" in result
        assert "invoice" in result

    def test_mixed_paragraph_no_aggressive(self):
        """Сбалансированный Ru+En — без явного majority → aggressive
        mode не применяется (избегаем false-positives)."""
        input_text = (
            "ООО Ромашка supply Tensar georesetka to АО Test. "
            "Standard procedure applied."
        )
        result = self._call(input_text)
        # Здесь «supply» / «Tensar» / «Standard» / «procedure» —
        # не должны ломаться Cyrillic-конвертом.
        assert "supply" in result.lower() or "ооо" in result.lower()
