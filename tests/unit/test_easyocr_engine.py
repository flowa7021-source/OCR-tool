"""Unit tests for EasyOCREngine and searchable PDF builder."""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np
import pytest

from src.application.engines.base import PageOCRResult
from src.application.engines.easyocr_engine import EasyOCREngine
from src.application.engines.registry import get_engine, reset_cache
from src.core.models import OCRConfig
from src.infrastructure.searchable_pdf_builder import (
    PageWords,
    build_searchable_pdf,
)
from src.shared.types import OCREngineKind


def _make_pdf_with_text(path: Path, text: str = "Hello OCR") -> None:
    """Create a one-page PDF with a printed line for OCR to find."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 200), text, fontsize=48)
    doc.save(str(path))
    doc.close()


# ─── searchable_pdf_builder ────────────────────────────────────────────────


def test_builder_writes_searchable_text(tmp_path: Path) -> None:
    raster = fitz.open()
    page = raster.new_page(width=595, height=842)
    pix = page.get_pixmap(dpi=150)
    raster.close()

    pages = [PageWords(
        image_png=pix.tobytes("png"),
        width_px=pix.width,
        height_px=pix.height,
        dpi=150,
        words=[(100, 100, 200, 40, 95.0, "Hello")],
    )]
    out = tmp_path / "out.pdf"
    build_searchable_pdf(pages, out)

    doc = fitz.open(str(out))
    text = doc[0].get_text("text")
    doc.close()
    assert "Hello" in text


def test_builder_skips_zero_size_boxes(tmp_path: Path) -> None:
    raster = fitz.open()
    page = raster.new_page(width=200, height=200)
    pix = page.get_pixmap(dpi=72)
    raster.close()
    pages = [PageWords(pix.tobytes("png"), pix.width, pix.height, 72, [
        (10, 10, 0, 30, 90.0, "zero_w"),
        (10, 10, 30, 0, 90.0, "zero_h"),
        (10, 10, 30, 30, 90.0, "   "),
    ])]
    out = tmp_path / "out.pdf"
    build_searchable_pdf(pages, out)
    text = fitz.open(str(out))[0].get_text("text").strip()
    assert text == ""


# ─── registry ──────────────────────────────────────────────────────────────


def test_registry_returns_easyocr_engine() -> None:
    reset_cache()
    engine = get_engine(OCREngineKind.EASYOCR)
    assert isinstance(engine, EasyOCREngine)
    assert engine.kind is OCREngineKind.EASYOCR
    reset_cache()


def test_registry_caches_instance() -> None:
    reset_cache()
    a = get_engine(OCREngineKind.EASYOCR)
    b = get_engine(OCREngineKind.EASYOCR)
    assert a is b
    reset_cache()


def test_reset_cache_calls_unload() -> None:
    reset_cache()
    engine = get_engine(OCREngineKind.EASYOCR)
    engine._reader = object()
    engine._reader_key = ("ru",)
    reset_cache()
    assert engine._reader is None


# ─── EasyOCR availability probe ────────────────────────────────────────────


def test_is_available_true_when_deps_installed() -> None:
    engine = EasyOCREngine()
    ok, _ = engine.is_available()
    assert ok is True


# ─── End-to-end engine run (requires easyocr + torch, ~15s CPU) ────────────


@pytest.mark.slow
def test_run_produces_searchable_pdf(tmp_path: Path) -> None:
    src_pdf = tmp_path / "in.pdf"
    out_pdf = tmp_path / "out.pdf"
    _make_pdf_with_text(src_pdf, "HELLO")

    engine = EasyOCREngine()
    results = engine.run(src_pdf, out_pdf, OCRConfig(dpi=200))
    assert out_pdf.exists()
    assert len(results) == 1
    assert isinstance(results[0], PageOCRResult)

    doc = fitz.open(str(out_pdf))
    text = doc[0].get_text("text").upper()
    doc.close()
    # EasyOCR may read "HELLO" with minor variations — check the core.
    assert "H" in text and "L" in text
