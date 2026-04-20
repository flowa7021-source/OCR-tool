"""Tests for :mod:`src.core.text_postprocessor`."""

from __future__ import annotations

import logging
import unicodedata

import pytest

from src.core.models import PostprocessConfig, RegexRule
from src.core.text_postprocessor import TextPostprocessor


@pytest.fixture
def processor() -> TextPostprocessor:
    return TextPostprocessor()


def _noop_config(**overrides) -> PostprocessConfig:
    """PostprocessConfig with everything OFF except the flag(s) we pass in."""
    base = {
        "autocorrect_russian": False,
        "autocorrect_english": False,
        "merge_hyphenated": False,
        "normalize_whitespace": False,
        "normalize_unicode": False,
        "remove_artifacts": False,
        "fix_cyrillic_latin_confusion": False,
        "garbage_filter_strictness": "disabled",
        "custom_rules": [],
    }
    base.update(overrides)
    return PostprocessConfig(**base)


def test_merge_hyphenated(processor: TextPostprocessor) -> None:
    text = "hyphen-\nated"
    cfg = _noop_config(merge_hyphenated=True)
    result = processor.process(text, cfg)
    assert result == "hyphenated"


def test_normalize_whitespace_collapses_spaces(processor: TextPostprocessor) -> None:
    text = "foo    bar\t\tbaz"
    cfg = _noop_config(normalize_whitespace=True)
    result = processor.process(text, cfg)
    assert result == "foo bar baz"


def test_normalize_whitespace_collapses_blank_lines(processor: TextPostprocessor) -> None:
    text = "foo\n\n\n\nbar"
    cfg = _noop_config(normalize_whitespace=True)
    result = processor.process(text, cfg)
    assert result == "foo\n\nbar"


def test_normalize_unicode_nfc(processor: TextPostprocessor) -> None:
    # 'é' expressed with combining acute accent ('e' + U+0301)
    decomposed = "cafe\u0301"
    assert decomposed != unicodedata.normalize("NFC", decomposed)
    cfg = _noop_config(normalize_unicode=True)
    result = processor.process(decomposed, cfg)
    assert result == unicodedata.normalize("NFC", decomposed)
    assert result == "café"


def test_custom_rule_literal_replace(processor: TextPostprocessor) -> None:
    text = "Hello FOO and foo"
    rule = RegexRule(
        pattern="foo",
        replacement="BAR",
        is_regex=False,
        case_sensitive=True,
    )
    cfg = _noop_config(custom_rules=[rule])
    assert processor.process(text, cfg) == "Hello FOO and BAR"


def test_custom_rule_regex_replace(processor: TextPostprocessor) -> None:
    text = "order 1234 and 56 items"
    rule = RegexRule(
        pattern=r"\d+",
        replacement="#",
        is_regex=True,
        case_sensitive=True,
    )
    cfg = _noop_config(custom_rules=[rule])
    assert processor.process(text, cfg) == "order # and # items"


def test_custom_rule_case_insensitive(processor: TextPostprocessor) -> None:
    text = "Hello FOO and foo and FoO"
    rule = RegexRule(
        pattern="foo",
        replacement="BAR",
        is_regex=False,
        case_sensitive=False,
    )
    cfg = _noop_config(custom_rules=[rule])
    assert processor.process(text, cfg) == "Hello BAR and BAR and BAR"


def test_disabled_rule_skipped(processor: TextPostprocessor) -> None:
    text = "keep foo intact"
    rule = RegexRule(
        pattern="foo",
        replacement="bar",
        enabled=False,
        is_regex=False,
    )
    cfg = _noop_config(custom_rules=[rule])
    assert processor.process(text, cfg) == text


def test_invalid_regex_rule_logged_not_raised(
    processor: TextPostprocessor, caplog: pytest.LogCaptureFixture
) -> None:
    text = "safe text"
    bad = RegexRule(
        pattern="(unclosed",  # invalid regex
        replacement="x",
        is_regex=True,
    )
    good = RegexRule(
        pattern="safe",
        replacement="SAFE",
        is_regex=False,
    )
    cfg = _noop_config(custom_rules=[bad, good])

    with caplog.at_level(logging.WARNING):
        result = processor.process(text, cfg)

    # The bad rule was skipped, but the good one still ran.
    assert result == "SAFE text"
    # And the error was logged rather than raised.
    assert any("некорректн" in rec.message or "правило" in rec.message.lower()
               for rec in caplog.records)


def test_empty_text_returned_as_is(processor: TextPostprocessor) -> None:
    cfg = PostprocessConfig()
    assert processor.process("", cfg) == ""


# ---------------------------------------------------------------------------
# Cyrillic ↔ Latin word-level look-alike normalisation
# ---------------------------------------------------------------------------


class TestCyrillicLatinFixup:
    """Word-level Latin↔Cyrillic swap covers the failure mode where
    Tesseract picks the wrong script at word boundaries."""

    def test_latin_o_at_end_of_cyrillic_word_is_fixed(
        self, processor: TextPostprocessor,
    ) -> None:
        # "Иванoв" with Latin `o` at position 4 — the trailing `в`
        # is Cyrillic-exclusive so the word classifies as cyr and the
        # lookalike is swapped.
        text = "Иван\u006fв"  # explicit Latin U+006F
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        assert result == "Иванов"
        assert "\u006f" not in result  # Latin o eliminated
        assert "\u043e" in result      # Cyrillic о present

    def test_latin_a_at_start_of_cyrillic_word_is_fixed(
        self, processor: TextPostprocessor,
    ) -> None:
        text = "\u0041рбуз"  # Latin A + "рбуз"
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        assert result == "Арбуз"
        assert result[0] == "\u0410"  # Cyrillic А

    def test_pure_latin_word_untouched(
        self, processor: TextPostprocessor,
    ) -> None:
        text = "Hello World"
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        assert processor.process(text, cfg) == "Hello World"

    def test_ambiguous_all_lookalike_word_untouched(
        self, processor: TextPostprocessor,
    ) -> None:
        # "ABC" — every letter has a Cyrillic look-alike; classifier
        # returns "mixed" and leaves it alone. Applies to both
        # interpretations so we don't corrupt product codes.
        text = "ABC"
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        assert processor.process(text, cfg) == "ABC"

    def test_url_with_cyrillic_neighbours_untouched(
        self, processor: TextPostprocessor,
    ) -> None:
        # Russian body text plus a real URL. The URL must survive
        # intact — we skip tokens matching the URL-shape guard.
        text = "Сайт https://example.com/path открывается"
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        assert "example.com" in result
        assert "открывается" in result  # Cyrillic pass-through

    def test_email_untouched(
        self, processor: TextPostprocessor,
    ) -> None:
        text = "Пишите на info@example.ru"
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        assert "info@example.ru" in result

    def test_disabled_flag_preserves_latin_lookalike(
        self, processor: TextPostprocessor,
    ) -> None:
        # With the flag off the pass is a no-op even on Russian text.
        text = "Иван\u006fв"
        cfg = _noop_config(fix_cyrillic_latin_confusion=False)
        assert processor.process(text, cfg) == text

    def test_mixed_word_with_script_exclusive_from_both_left_alone(
        self, processor: TextPostprocessor,
    ) -> None:
        # "ГОSTи" has Cyrillic Г (exclusive) AND Latin S (exclusive) —
        # ambiguous intent, leave it alone rather than guess.
        text = "ГОSTи"
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        assert processor.process(text, cfg) == text

    def test_runs_before_regex_autocorrect(
        self, processor: TextPostprocessor,
    ) -> None:
        # Regression guard: the word-level pass MUST run before the
        # regex autocorrect so the Cyrillic-context look-arounds see
        # a consistently-scripted word. If ordering regresses, the
        # regex rules won't fire on the fixed-up characters.
        text = "Ивaн\u006fв"  # Latin a (U+0061) + Latin o (U+006F)
        cfg = _noop_config(
            fix_cyrillic_latin_confusion=True,
            autocorrect_russian=True,
        )
        result = processor.process(text, cfg)
        # After fixup both look-alikes should be Cyrillic
        assert result == "Иванов"


class TestCyrillicLatinFixupMonoLookalikeWords:
    """Stage F tightening — words made ENTIRELY of look-alikes.

    ``ru_business_letter`` landed at WER 25 % on the nightly
    benchmark because Tesseract emits short all-look-alike tokens
    like ``Сo`` (Cyrillic С + Latin o) or ``AS`` (Latin
    look-alikes) that the ``_classify_word_script`` "mixed" branch
    refuses to touch. This is correct for ambiguous content but
    wrong for Russian documents where a 2-4 char run of look-
    alikes almost certainly means one slipped script.

    The fix uses the surrounding paragraph: if the paragraph has a
    clear script-exclusive majority, short all-look-alike tokens
    inherit that majority.
    """

    def test_paragraph_context_resolves_short_lookalike_token(
        self, processor: TextPostprocessor,
    ) -> None:
        # Two-word Cyrillic sentence + ``Сo`` fragment with Latin o.
        # Without paragraph context the current fixup leaves ``Сo``
        # alone (pure look-alikes → "mixed" classifier). With
        # paragraph context it inherits the surrounding Cyrillic
        # majority and swaps the Latin o.
        text = "Видно что С\u006f сделано"  # Latin o after С
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        assert result == "Видно что Со сделано", (
            f"Short all-look-alike token not inherited from "
            f"Cyrillic paragraph context: {result!r}"
        )

    def test_latin_majority_paragraph_preserves_latin_token(
        self, processor: TextPostprocessor,
    ) -> None:
        # English sentence with an all-look-alike 2-letter token.
        # Paragraph context is Latin → token stays Latin.
        text = "The quick brown fox jumps over ABC"
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        assert result == text, (
            f"Latin-majority paragraph wrongly modified: {result!r}"
        )

    def test_ambiguous_paragraph_leaves_tokens_alone(
        self, processor: TextPostprocessor,
    ) -> None:
        # No unambiguous script evidence in the paragraph → fall
        # through to the per-word classifier which keeps everything.
        text = "ABC oo"
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        assert result == text


class TestCyrillicLatinFixupNumericContext:
    """Numeric-context heuristic — product codes / invoice numbers
    stay Latin even when the surrounding paragraph is Russian.

    The paragraph-majority fallback on its own would wrongly flip
    ``INV-12345`` to Cyrillic inside a Russian contract body. The
    numeric-context check runs first: a word adjacent to digits
    (via ``-``, ``_``, ``/``, ``.`` or at position-adjacent offset)
    is treated as likely-Latin and the Cyrillic look-alike letters
    are reverse-mapped to their Latin twins.
    """

    def test_hyphen_digit_suffix_keeps_latin(
        self, processor: TextPostprocessor,
    ) -> None:
        # Russian-majority paragraph would normally swing "OCP"
        # toward Cyrillic. With the adjacent "-12345" the numeric-
        # context heuristic pins it to Latin.
        text = (
            "В нашем письме номер ОСР-12345 содержатся важные детали "
            "и дополнительная информация"
        )
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        # The token before the digits was all Cyrillic look-alikes;
        # post-heuristic it should be Latin "OCP".
        assert "OCP-12345" in result, (
            f"Numeric-context heuristic should keep the code Latin: {result!r}"
        )

    def test_dot_digit_prefix_keeps_latin(
        self, processor: TextPostprocessor,
    ) -> None:
        text = "Подробнее смотри в пункте 5.INV данного документа здесь"
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        # "INV" has Latin-exclusive V — stays Latin without the
        # heuristic too. This exercises the "digit before" path.
        assert "5.INV" in result

    def test_space_separated_number_does_not_apply(
        self, processor: TextPostprocessor,
    ) -> None:
        # Whitespace between the word and the number means the word
        # is NOT inside a numeric code — paragraph majority wins.
        # "ОСР" here has no Latin-exclusive evidence, paragraph is
        # Russian → stays Cyrillic.
        text = (
            "Просто слово ОСР 12345 стоит отдельно от числа "
            "в этом предложении на русском языке"
        )
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        assert "ОСР" in result, (
            f"Isolated token with whitespace separator should follow "
            f"paragraph majority: {result!r}"
        )

    def test_numeric_context_before_and_after(
        self, processor: TextPostprocessor,
    ) -> None:
        # Digit on BOTH sides — unambiguous numeric context.
        text = "Значение 77INV99 используется в продуктовом каталоге компании"
        cfg = _noop_config(fix_cyrillic_latin_confusion=True)
        result = processor.process(text, cfg)
        assert "77INV99" in result
