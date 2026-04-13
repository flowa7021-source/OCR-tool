"""Tests for :mod:`src.infrastructure.file_utils`."""

from __future__ import annotations

from pathlib import Path

from src.infrastructure.file_utils import (
    bytes_human,
    ensure_dir,
    is_valid_pdf,
    safe_unique_path,
    suggest_output_path,
)


class TestSafeUniquePath:
    def test_returns_original_when_free(self, tmp_path: Path) -> None:
        p = tmp_path / "new.txt"
        assert safe_unique_path(p) == p

    def test_returns_variant_when_exists(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("first")
        uniq = safe_unique_path(p)
        assert uniq != p
        assert uniq.name == "file (1).txt"
        assert not uniq.exists()

    def test_increments_counter(self, tmp_path: Path) -> None:
        p = tmp_path / "file.txt"
        p.write_text("a")
        (tmp_path / "file (1).txt").write_text("b")
        uniq = safe_unique_path(p)
        assert uniq.name == "file (2).txt"


class TestSuggestOutputPath:
    def test_default_suffix(self, tmp_path: Path) -> None:
        src = tmp_path / "doc.pdf"
        result = suggest_output_path(src)
        assert result.name == "doc_ocr.pdf"
        assert result.parent == tmp_path

    def test_custom_suffix(self, tmp_path: Path) -> None:
        src = tmp_path / "doc.pdf"
        result = suggest_output_path(src, suffix="_processed")
        assert result.name == "doc_processed.pdf"

    def test_custom_extension(self, tmp_path: Path) -> None:
        src = tmp_path / "doc.pdf"
        result = suggest_output_path(src, ext=".txt")
        assert result.name == "doc_ocr.txt"

    def test_extension_without_dot(self, tmp_path: Path) -> None:
        src = tmp_path / "doc.pdf"
        result = suggest_output_path(src, ext="txt")
        assert result.name == "doc_ocr.txt"


class TestBytesHuman:
    def test_zero(self) -> None:
        assert bytes_human(0) == "0 B"

    def test_kb(self) -> None:
        assert bytes_human(1024) == "1.00 KB"

    def test_mb(self) -> None:
        assert bytes_human(1024 * 1024) == "1.00 MB"

    def test_bytes_integer(self) -> None:
        assert bytes_human(500) == "500 B"


class TestIsValidPdf:
    def test_real_pdf_header(self, fake_pdf_path: Path) -> None:
        assert is_valid_pdf(fake_pdf_path) is True

    def test_fake_bytes(self, tmp_path: Path) -> None:
        p = tmp_path / "fake.pdf"
        p.write_bytes(b"Not a PDF at all")
        assert is_valid_pdf(p) is False

    def test_missing_file(self, tmp_path: Path) -> None:
        assert is_valid_pdf(tmp_path / "nope.pdf") is False


class TestEnsureDir:
    def test_creates_nested_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c"
        assert not target.exists()
        result = ensure_dir(target)
        assert result == target
        assert target.is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        target = tmp_path / "existing"
        target.mkdir()
        ensure_dir(target)  # must not raise
        assert target.is_dir()
