"""Real-OCR: paths with Cyrillic names, spaces, and nested directories.

The app ships Windows-only and Russian users typically store documents
in paths like ``C:\\Users\\Иванов И.И.\\Документы\\Накладные\\...``
with spaces and Cyrillic glyphs at every level. OpenCV's ``cv2.imread``
/ ``cv2.imwrite`` are known to silently fail on non-ASCII Windows paths
(ANSI-encoded internally). The pipeline works around this via
``Path.write_bytes`` + ``cv2.imdecode(raw)``, but that workaround is
easy to regress when someone "cleans up" the I/O code.

These tests force the whole real-OCR pipeline through paths that
would break a naive implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.types import JobStatus
from tests.integration._real_ocr_helpers import (
    make_realistic_profile,
    render_clean_text_pdf,
    requires_real_ocr,
    run_pipeline,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


class TestCyrillicPath:
    """Cyrillic folder + Cyrillic filename must not break real OCR."""

    def test_cyrillic_dir_and_filename(
        self,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        cyr_dir = tmp_path / "Иванов И.И" / "Документы" / "Накладные"
        cyr_dir.mkdir(parents=True)
        input_pdf = render_clean_text_pdf(
            cyr_dir / "накладная_№36.pdf", "invoice page", pages=1,
        )
        output_pdf = cyr_dir / "накладная_№36_ocr.pdf"

        profile = make_realistic_profile(binarization="otsu", dpi=200)
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        assert result.status is JobStatus.COMPLETED, (
            f"Cyrillic path pipeline FAILED: {result.error}"
        )
        assert output_pdf.exists(), (
            "Output PDF missing — Windows ANSI-path bug likely regressed"
        )


class TestPathWithSpaces:
    """Spaces in every path segment — classic shell-escaping trap."""

    def test_spaces_throughout_path(
        self,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        space_dir = (
            tmp_path / "Program Files" / "OCR Studio" / "test inputs"
        )
        space_dir.mkdir(parents=True)
        input_pdf = render_clean_text_pdf(
            space_dir / "my document 2024.pdf", "spaces work", pages=1,
        )
        output_pdf = space_dir / "my document 2024 ocr.pdf"

        profile = make_realistic_profile(binarization="otsu", dpi=200)
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        assert result.status is JobStatus.COMPLETED, result.error
        assert output_pdf.exists()


class TestMixedCyrillicAndAscii:
    """One path with both Cyrillic and ASCII segments + numbers + symbols."""

    def test_mixed_segments(
        self,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        mixed_dir = (
            tmp_path / "2024" / "Q1-Отчёты" / "sub dir" / "Накладные№1-12"
        )
        mixed_dir.mkdir(parents=True)
        input_pdf = render_clean_text_pdf(
            mixed_dir / "doc-42_ТН.pdf", "mixed path test", pages=1,
        )
        output_pdf = mixed_dir / "doc-42_ТН_ocr.pdf"

        profile = make_realistic_profile(binarization="otsu", dpi=200)
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        assert result.status is JobStatus.COMPLETED, result.error
