"""Tests for :mod:`src.core.border_remover` — TDD: tests first.

What border_remover does:
  Tesseract mis-reads long horizontal / vertical lines (table borders,
  underlines, form rules) as letters — Ivanov inside a table cell
  loses its final `в` because Tesseract fuses it into the right-
  hand vertical rule, or a ruler line under a signature becomes
  ``_______`` in the output. This module detects those lines before
  OCR and erases them (fills with background colour), preserving
  the glyphs that were next to them.

What it must NOT do:
  * Erase diagonal strokes — those are legitimate glyph components
    (``/``, ``A``, ``K``, Cyrillic ``К``/``Ж``/``Х``).
  * Erase short lines — they're typically letter serifs or glyph
    stems.
  * Touch regions with no detectable lines at all.
  * Modify the image colour space or add artifacts elsewhere on
    the page.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def border_remover_module():
    """Lazy import — module doesn't exist on first collection."""
    from src.core import border_remover

    return border_remover


def _blank(h: int = 800, w: int = 800) -> np.ndarray:
    """White 800x800 grayscale canvas."""
    return np.full((h, w), 255, dtype=np.uint8)


def _add_horizontal_line(
    img: np.ndarray, y: int, x0: int, x1: int, thickness: int = 2,
) -> None:
    img[y : y + thickness, x0:x1] = 0


def _add_vertical_line(
    img: np.ndarray, x: int, y0: int, y1: int, thickness: int = 2,
) -> None:
    img[y0:y1, x : x + thickness] = 0


def _add_diagonal_line(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
) -> None:
    import cv2

    cv2.line(img, (x0, y0), (x1, y1), 0, thickness=2)


def _count_black_pixels(img: np.ndarray) -> int:
    return int((img < 64).sum())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_exports_remove_function(self, border_remover_module) -> None:
        assert callable(border_remover_module.remove_border_lines)

    def test_exports_threshold_constant(self, border_remover_module) -> None:
        # Minimum line length (in pixels) to count as a "border"
        # rather than a glyph stroke. Module-level constant so
        # callers can tune it.
        assert hasattr(border_remover_module, "MIN_LINE_LENGTH_PX")


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


class TestBorderRemoval:
    def test_horizontal_line_removed(self, border_remover_module) -> None:
        img = _blank()
        _add_horizontal_line(img, y=400, x0=50, x1=750, thickness=3)
        black_before = _count_black_pixels(img)

        cleaned = border_remover_module.remove_border_lines(img)

        black_after = _count_black_pixels(cleaned)
        assert black_after < black_before * 0.2, (
            f"Horizontal line NOT removed: {black_before} → {black_after} "
            f"black pixels"
        )

    def test_vertical_line_removed(self, border_remover_module) -> None:
        img = _blank()
        _add_vertical_line(img, x=400, y0=50, y1=750, thickness=3)
        black_before = _count_black_pixels(img)

        cleaned = border_remover_module.remove_border_lines(img)

        black_after = _count_black_pixels(cleaned)
        assert black_after < black_before * 0.2, (
            f"Vertical line NOT removed: {black_before} → {black_after} "
            f"black pixels"
        )

    def test_both_directions_removed(self, border_remover_module) -> None:
        img = _blank()
        _add_horizontal_line(img, y=200, x0=50, x1=750, thickness=3)
        _add_vertical_line(img, x=400, y0=50, y1=750, thickness=3)

        cleaned = border_remover_module.remove_border_lines(img)

        # Near-white after removing both
        black_after = _count_black_pixels(cleaned)
        total_px = cleaned.size
        assert black_after < 0.005 * total_px, (
            f"Grid not fully removed: {black_after}/{total_px} black"
        )


# ---------------------------------------------------------------------------
# Invariants — what must NOT happen
# ---------------------------------------------------------------------------


class TestPreservesGlyphs:
    def test_diagonal_line_preserved(self, border_remover_module) -> None:
        # A diagonal stroke must not be detected as a border. This
        # protects characters like ``A``, ``K``, ``/`` and Cyrillic
        # ``К``, ``Ж``.
        img = _blank()
        _add_diagonal_line(img, x0=100, y0=100, x1=600, y1=600)
        black_before = _count_black_pixels(img)

        cleaned = border_remover_module.remove_border_lines(img)

        black_after = _count_black_pixels(cleaned)
        # Diagonal should be largely preserved — at least 80 %.
        assert black_after > 0.8 * black_before, (
            f"Diagonal erased: {black_before} → {black_after}"
        )

    def test_short_line_preserved(self, border_remover_module) -> None:
        # A 30px "-" shouldn't be mistaken for a ruler border.
        img = _blank()
        _add_horizontal_line(img, y=400, x0=380, x1=410, thickness=2)
        black_before = _count_black_pixels(img)

        cleaned = border_remover_module.remove_border_lines(img)

        black_after = _count_black_pixels(cleaned)
        assert black_after > 0.5 * black_before, (
            f"Short dash mis-detected as border: "
            f"{black_before} → {black_after}"
        )

    def test_empty_page_passes_through(self, border_remover_module) -> None:
        img = _blank()
        cleaned = border_remover_module.remove_border_lines(img)
        # np.array_equal for exact match — no rounding, no drift.
        assert np.array_equal(cleaned, img)

    def test_shape_and_dtype_unchanged(self, border_remover_module) -> None:
        img = _blank()
        _add_horizontal_line(img, y=400, x0=50, x1=750)
        cleaned = border_remover_module.remove_border_lines(img)
        assert cleaned.shape == img.shape
        assert cleaned.dtype == img.dtype


class TestTextNextToBorder:
    """The canonical failure mode: text touching a table border.
    After border removal the glyph strokes must survive — this
    verifies the erasure doesn't bleed into adjacent content."""

    def test_text_next_to_vertical_line_preserved(
        self, border_remover_module,
    ) -> None:
        import cv2

        img = _blank()
        # Add a "TEXT" glyph region via thick black strokes
        cv2.putText(
            img, "TEXT", (60, 410),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3,
        )
        _add_vertical_line(img, x=300, y0=350, y1=450, thickness=3)

        text_region_before = _count_black_pixels(img[380:430, 50:250])

        cleaned = border_remover_module.remove_border_lines(img)

        text_region_after = _count_black_pixels(cleaned[380:430, 50:250])
        # Text region ≥ 90 % preserved — no bleed from line removal.
        assert text_region_after >= 0.9 * text_region_before, (
            f"Line removal eroded adjacent text: "
            f"{text_region_before} → {text_region_after}"
        )


# ---------------------------------------------------------------------------
# Configurable threshold
# ---------------------------------------------------------------------------


class TestMinLineLengthThreshold:
    def test_custom_threshold_respected(self, border_remover_module) -> None:
        # A 100-px horizontal line gets erased at default (50 px
        # threshold) but preserved when we ask for a 300 px minimum.
        img = _blank()
        _add_horizontal_line(img, y=400, x0=350, x1=450, thickness=2)

        default_cleaned = border_remover_module.remove_border_lines(img)
        strict_cleaned = border_remover_module.remove_border_lines(
            img, min_line_length=300,
        )

        default_black = _count_black_pixels(default_cleaned)
        strict_black = _count_black_pixels(strict_cleaned)
        # Default erases it (strictly less black than strict mode)
        assert default_black < strict_black
