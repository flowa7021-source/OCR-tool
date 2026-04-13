"""Tests for the ExportManager: TXT, DOCX, PDF passthrough, dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.export_manager import ExportError, ExportManager
from src.core.models import JobResult, PageResult
from src.shared.types import ExportFormat, JobStatus


def _sample_result(output_path: Path) -> JobResult:
    return JobResult(
        job_id="j1",
        status=JobStatus.COMPLETED,
        input_path="/tmp/in.pdf",
        output_path=str(output_path),
        pages=[
            PageResult(page_number=1, text="Первая страница.\n\nВторой абзац.", mean_confidence=92.5),
            PageResult(page_number=2, text="Hello world.", mean_confidence=88.1),
            PageResult(page_number=3, error="Timeout"),
        ],
        total_time_sec=5.5,
    )


class TestExportTxt:
    def test_writes_file(self, tmp_path: Path) -> None:
        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "out.txt"
        written = ExportManager().export_txt(result, target)
        assert written == target
        assert target.exists()

    def test_contains_all_page_text(self, tmp_path: Path) -> None:
        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "out.txt"
        ExportManager().export_txt(result, target)
        body = target.read_text(encoding="utf-8")
        assert "Первая страница." in body
        assert "Второй абзац." in body
        assert "Hello world." in body
        # Page headers present
        assert "Page 1" in body
        assert "Page 2" in body
        assert "Page 3" in body

    def test_error_marker_preserved(self, tmp_path: Path) -> None:
        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "out.txt"
        ExportManager().export_txt(result, target)
        body = target.read_text(encoding="utf-8")
        assert "[ERROR: Timeout]" in body

    def test_unicode_roundtrip(self, tmp_path: Path) -> None:
        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "out.txt"
        ExportManager().export_txt(result, target)
        # Ensure we didn't accidentally double-encode Cyrillic
        raw = target.read_bytes()
        assert "Первая".encode("utf-8") in raw

    def test_bom_when_utf8_sig(self, tmp_path: Path) -> None:
        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "out.txt"
        ExportManager().export_txt(result, target, encoding="utf-8-sig")
        raw = target.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")

    def test_creates_parent_dir(self, tmp_path: Path) -> None:
        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "sub" / "deep" / "out.txt"
        ExportManager().export_txt(result, target)
        assert target.exists()


class TestExportDocx:
    def test_writes_file(self, tmp_path: Path) -> None:
        pytest.importorskip("docx")
        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "out.docx"
        written = ExportManager().export_docx(result, target)
        assert written == target
        assert target.exists()
        # DOCX is a ZIP archive
        assert target.read_bytes().startswith(b"PK")

    def test_contains_page_headings(self, tmp_path: Path) -> None:
        pytest.importorskip("docx")
        from docx import Document

        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "out.docx"
        ExportManager().export_docx(result, target)

        doc = Document(str(target))
        all_text = "\n".join(p.text for p in doc.paragraphs)
        assert "Page 1" in all_text
        assert "Первая страница." in all_text
        assert "Hello world." in all_text

    def test_error_page_marked(self, tmp_path: Path) -> None:
        pytest.importorskip("docx")
        from docx import Document

        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "out.docx"
        ExportManager().export_docx(result, target)
        doc = Document(str(target))
        all_text = "\n".join(p.text for p in doc.paragraphs)
        assert "[ERROR: Timeout]" in all_text


class TestExportDispatcher:
    def test_pdf_passthrough_returns_output_path(self, tmp_path: Path) -> None:
        target = tmp_path / "searchable.pdf"
        target.write_bytes(b"%PDF-1.7\n")
        result = _sample_result(target)
        written = ExportManager().export(result, target, ExportFormat.PDF)
        assert written == target

    def test_txt_via_dispatcher(self, tmp_path: Path) -> None:
        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "out.txt"
        written = ExportManager().export(result, target, ExportFormat.TXT)
        assert written == target
        assert "Hello world." in target.read_text(encoding="utf-8")

    def test_docx_via_dispatcher(self, tmp_path: Path) -> None:
        pytest.importorskip("docx")
        result = _sample_result(tmp_path / "out.pdf")
        target = tmp_path / "out.docx"
        written = ExportManager().export(result, target, ExportFormat.DOCX)
        assert written == target
        assert target.exists()

    def test_clipboard_without_qapp_raises(self, tmp_path: Path) -> None:
        # No QApplication running; clipboard should refuse clearly
        result = _sample_result(tmp_path / "out.pdf")
        with pytest.raises(RuntimeError):
            ExportManager().export(result, tmp_path / "dummy", ExportFormat.CLIPBOARD)


class TestConcatenateText:
    def test_includes_all_pages(self, tmp_path: Path) -> None:
        result = _sample_result(tmp_path / "out.pdf")
        text = ExportManager._concatenate_text(result)
        assert "Первая страница." in text
        assert "Hello world." in text
        assert "[ERROR: Timeout]" in text
        assert "Page 1" in text and "Page 3" in text
