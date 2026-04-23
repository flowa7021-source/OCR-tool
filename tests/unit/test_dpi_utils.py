"""Tests for the shared DPI-detection helper."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.shared.dpi_utils import estimate_page_source_dpi


def _mock_page(infos: list[dict] | None) -> MagicMock:
    page = MagicMock()
    if infos is None:
        page.get_image_info.side_effect = AttributeError("no info")
    else:
        page.get_image_info.return_value = infos
    return page


def test_page_with_no_images_returns_none():
    page = _mock_page([])
    assert estimate_page_source_dpi(page) is None


def test_page_raising_on_get_image_info_returns_none():
    page = _mock_page(None)
    assert estimate_page_source_dpi(page) is None


def test_square_image_at_300_dpi():
    # 8.5×11" page at 300 DPI = 2550×3300 px, bbox in points
    # (8.5 × 72 = 612 pt, 11 × 72 = 792 pt).
    page = _mock_page([
        {"width": 2550, "height": 3300, "bbox": (0, 0, 612, 792)},
    ])
    dpi = estimate_page_source_dpi(page)
    assert 298 <= dpi <= 302  # float rounding


def test_low_dpi_scan_detected():
    # 100 DPI on the same A4 page: 850×1100 px
    page = _mock_page([
        {"width": 850, "height": 1100, "bbox": (0, 0, 612, 792)},
    ])
    assert 95 <= estimate_page_source_dpi(page) <= 105


def test_largest_image_chosen():
    # Small logo + big scan — scan wins.
    page = _mock_page([
        {"width": 80, "height": 60, "bbox": (0, 0, 60, 45)},    # logo, ~96 dpi
        {"width": 2550, "height": 3300, "bbox": (0, 0, 612, 792)},  # scan
    ])
    dpi = estimate_page_source_dpi(page)
    assert 290 <= dpi <= 310


def test_malformed_bbox_returns_none():
    page = _mock_page([
        {"width": 100, "height": 100, "bbox": (0, 0, 0, 0)},
    ])
    assert estimate_page_source_dpi(page) is None
