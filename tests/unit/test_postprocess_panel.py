"""Tests for the postprocess panel UI widget."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.core.models import PostprocessConfig, RegexRule  # noqa: E402
from src.ui.postprocess_panel import PostprocessPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_roundtrip_config(qapp: QApplication) -> None:
    panel = PostprocessPanel()
    cfg = PostprocessConfig(
        autocorrect_russian=False,
        autocorrect_english=True,
        merge_hyphenated=False,
        normalize_whitespace=True,
        normalize_unicode=False,
        remove_artifacts=True,
        custom_rules=[
            RegexRule(
                pattern="foo",
                replacement="bar",
                enabled=True,
                is_regex=False,
                case_sensitive=False,
                description="demo",
            ),
            RegexRule(
                pattern=r"\d+",
                replacement="N",
                enabled=False,
                is_regex=True,
                case_sensitive=True,
                description="nums",
            ),
        ],
    )
    panel.set_config(cfg)
    got = panel.get_config()

    assert got.autocorrect_russian is False
    assert got.autocorrect_english is True
    assert got.merge_hyphenated is False
    assert got.normalize_whitespace is True
    assert got.normalize_unicode is False
    assert got.remove_artifacts is True

    assert len(got.custom_rules) == 2
    r1, r2 = got.custom_rules
    assert r1.pattern == "foo"
    assert r1.replacement == "bar"
    assert r1.enabled is True
    assert r1.is_regex is False
    assert r1.case_sensitive is False
    assert r1.description == "demo"

    assert r2.pattern == r"\d+"
    assert r2.enabled is False
    assert r2.is_regex is True
    assert r2.case_sensitive is True


def test_empty_pattern_filtered(qapp: QApplication) -> None:
    panel = PostprocessPanel()
    panel._add_rule()  # type: ignore[attr-defined]
    got = panel.get_config()
    # A rule with empty pattern should be dropped
    assert got.custom_rules == []


def test_preview_applies_rules(qapp: QApplication) -> None:
    panel = PostprocessPanel()
    panel.set_config(
        PostprocessConfig(
            autocorrect_russian=False,
            autocorrect_english=False,
            merge_hyphenated=False,
            normalize_whitespace=False,
            normalize_unicode=False,
            remove_artifacts=False,
            custom_rules=[
                RegexRule(
                    pattern="hello",
                    replacement="world",
                    is_regex=False,
                    case_sensitive=True,
                    enabled=True,
                )
            ],
        )
    )
    panel.preview_input.setText("say hello!")
    panel._refresh_preview()
    assert panel.preview_output.text() == "say world!"
