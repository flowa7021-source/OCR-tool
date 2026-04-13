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


def _noop_config(**overrides: bool) -> PostprocessConfig:
    """PostprocessConfig with everything OFF except the flag(s) we pass in."""
    base = dict(
        autocorrect_russian=False,
        autocorrect_english=False,
        merge_hyphenated=False,
        normalize_whitespace=False,
        normalize_unicode=False,
        remove_artifacts=False,
        custom_rules=[],
    )
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
