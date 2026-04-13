"""Helpers for loading tinted SVG icons bundled with the application.

Qt's :class:`QIcon` can render SVG natively, but it has no easy way to recolor
the stroke/fill at load time. Since our icons use ``stroke="currentColor"`` or
``fill="currentColor"`` we do a trivial text substitution before handing the
bytes to :class:`QSvgRenderer`, producing a pixmap in any target color.

This module degrades gracefully — if an icon file is missing we return an
empty :class:`QIcon`. That way the UI never crashes because of a missing
asset during development.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap

from src.shared.constants import COLOR_TEXT_PRIMARY, ICONS_DIR

logger = logging.getLogger(__name__)

_ICON_CACHE: dict[tuple[str, str, int], QIcon] = {}


def _load_svg_bytes(name: str) -> bytes | None:
    """Return raw SVG bytes for ``name`` (without extension), or None."""
    path = _icon_path(name)
    if not path.exists():
        logger.debug("Icon not found: %s", path)
        return None
    try:
        return path.read_bytes()
    except OSError as exc:
        logger.warning("Could not read icon %s: %s", path, exc)
        return None


def _icon_path(name: str) -> Path:
    return ICONS_DIR / f"{name}.svg"


def load_icon(
    name: str, color: str = COLOR_TEXT_PRIMARY, size: int = 24
) -> QIcon:
    """Load a tinted SVG icon.

    Args:
        name: Basename of the icon file under ``resources/icons`` (no .svg).
        color: CSS-style hex color used to substitute ``currentColor``.
        size: Target square size in logical pixels.

    Returns:
        A :class:`QIcon`. Empty if the file is missing.
    """
    key = (name, color, size)
    cached = _ICON_CACHE.get(key)
    if cached is not None:
        return cached

    raw = _load_svg_bytes(name)
    if raw is None:
        icon = QIcon()
        _ICON_CACHE[key] = icon
        return icon

    try:
        tinted = raw.replace(b"currentColor", color.encode("ascii"))
    except Exception:  # noqa: BLE001
        tinted = raw

    from PySide6.QtSvg import QSvgRenderer  # lazy to keep import light

    renderer = QSvgRenderer(QByteArray(tinted))
    if not renderer.isValid():
        logger.warning("Invalid SVG: %s", name)
        icon = QIcon()
        _ICON_CACHE[key] = icon
        return icon

    # Render at 2x for crisp hi-DPI, downscale on display.
    img = QImage(size * 2, size * 2, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    renderer.render(painter)
    painter.end()

    icon = QIcon(QPixmap.fromImage(img))
    _ICON_CACHE[key] = icon
    return icon


def app_icon() -> QIcon:
    """Return the application icon (multi-size)."""
    icon = QIcon()
    raw = _load_svg_bytes("app")
    if raw is None:
        return icon
    try:
        from PySide6.QtSvg import QSvgRenderer

        renderer = QSvgRenderer(QByteArray(raw))
        if not renderer.isValid():
            return icon
        for sz in (16, 24, 32, 48, 64, 128, 256):
            img = QImage(sz, sz, QImage.Format.Format_ARGB32)
            img.fill(Qt.GlobalColor.transparent)
            painter = QPainter(img)
            renderer.render(painter)
            painter.end()
            icon.addPixmap(QPixmap.fromImage(img))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build app icon: %s", exc)
    return icon


__all__ = ["load_icon", "app_icon"]
