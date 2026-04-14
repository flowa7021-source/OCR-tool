"""Tests for the SVG icon loader."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_load_icon_returns_non_null_for_bundled(qapp: QApplication) -> None:
    from src.ui.icons import load_icon

    icon = load_icon("open")
    assert not icon.isNull()
    assert icon.availableSizes()  # at least one size


def test_load_icon_caches_repeated_calls(qapp: QApplication) -> None:
    from src.ui import icons as icons_mod

    icons_mod._ICON_CACHE.clear()
    icon_a = icons_mod.load_icon("start", "#FF0000", 24)
    icon_b = icons_mod.load_icon("start", "#FF0000", 24)
    # Same key → same object due to cache
    assert icon_a is icon_b


def test_load_icon_different_colors_produce_separate_cache_entries(
    qapp: QApplication,
) -> None:
    from src.ui import icons as icons_mod

    icons_mod._ICON_CACHE.clear()
    a = icons_mod.load_icon("start", "#FFFFFF", 24)
    b = icons_mod.load_icon("start", "#000000", 24)
    assert a is not b


def test_load_icon_missing_returns_empty(qapp: QApplication) -> None:
    from src.ui.icons import load_icon

    icon = load_icon("no_such_icon_definitely_does_not_exist")
    assert icon.isNull()


def test_app_icon_has_multiple_sizes(qapp: QApplication) -> None:
    from src.ui.icons import app_icon

    icon = app_icon()
    sizes = [(s.width(), s.height()) for s in icon.availableSizes()]
    assert (16, 16) in sizes
    assert (64, 64) in sizes
    assert (256, 256) in sizes


def test_load_icon_tints_current_color(qapp: QApplication) -> None:
    """The loader should substitute currentColor with the given hex color."""
    from src.ui import icons as icons_mod

    icons_mod._ICON_CACHE.clear()
    # Render a very distinctive hot-pink icon and check pixels contain pink
    icon = icons_mod.load_icon("stop", "#FF00AA", 32)
    pm = icon.pixmap(32, 32)
    img = pm.toImage()
    found = False
    for y in range(img.height()):
        for x in range(img.width()):
            c = img.pixelColor(x, y)
            if c.red() > 200 and c.blue() > 150 and c.green() < 80 and c.alpha() > 0:
                found = True
                break
        if found:
            break
    assert found, "Tinted color should appear in rendered pixmap"
