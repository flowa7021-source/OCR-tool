"""Helpers for screen-reader-friendly widgets.

Qt's accessibility layer reads ``accessibleName`` (identity) and
``accessibleDescription`` (detail). Tooltips are **not** read by most
screen readers by default, so a control with only a ``setToolTip`` is
invisible to NVDA / JAWS / Narrator. This helper keeps both in sync.

Usage:

    from src.ui.accessibility import describe

    describe(
        self.clahe_clip,
        name="CLAHE clip limit",
        description=(
            "Предел усиления контраста CLAHE. Больше — агрессивнее "
            "подсветка деталей, но и больше шума. Рекомендовано 1.5–3.0."
        ),
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


def describe(
    widget: QWidget,
    *,
    name: str,
    description: str,
    tooltip: str | None = None,
) -> None:
    """Attach an accessible name + description + tooltip to ``widget``.

    ``tooltip`` defaults to ``description`` so sighted users see the
    same hint on hover that screen readers announce.
    """
    widget.setAccessibleName(name)
    widget.setAccessibleDescription(description)
    widget.setToolTip(tooltip if tooltip is not None else description)
