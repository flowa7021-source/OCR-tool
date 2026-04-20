"""Tests for the low-confidence word annotation helpers in export_manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.export_manager import (
    ExportManager,
    _mark_low_conf_words_txt,
    _split_for_docx_annotation,
)
from src.core.models import JobResult, PageResult
from src.shared.types import JobStatus


def _page(text: str, low: list[str]) -> PageResult:
    return PageResult(page_number=1, text=text, low_confidence_words=low)


class TestMarkLowConfWordsTxt:
    def test_wraps_whole_word_occurrences(self) -> None:
        page = _page("The quick brown fox", ["quick", "fox"])
        assert _mark_low_conf_words_txt(page) == "The [?quick?] brown [?fox?]"

    def test_empty_list_returns_text_unchanged(self) -> None:
        page = _page("Nothing to mark", [])
        assert _mark_low_conf_words_txt(page) == "Nothing to mark"

    def test_empty_text_returns_empty_string(self) -> None:
        page = _page("", ["foo"])
        assert _mark_low_conf_words_txt(page) == ""

    def test_substring_not_marked(self) -> None:
        # "ain" is a low-conf word but "brain" shouldn't be rewritten
        # to "br[?ain?]" — word-boundary guards prevent that.
        page = _page("brain drain", ["ain"])
        assert _mark_low_conf_words_txt(page) == "brain drain"

    def test_cyrillic_word_boundaries(self) -> None:
        page = _page("Привет мир", ["Привет"])
        assert _mark_low_conf_words_txt(page) == "[?Привет?] мир"

    def test_duplicate_entries_deduplicated(self) -> None:
        # Same word listed twice shouldn't produce double-wrapping.
        page = _page("repeat repeat repeat", ["repeat", "repeat"])
        assert (
            _mark_low_conf_words_txt(page)
            == "[?repeat?] [?repeat?] [?repeat?]"
        )

    def test_custom_markers(self) -> None:
        page = _page("one two three", ["two"])
        out = _mark_low_conf_words_txt(
            page, marker_open="<", marker_close=">",
        )
        assert out == "one <two> three"


class TestSplitForDocxAnnotation:
    def test_interleaved_spans(self) -> None:
        page = _page("Alpha beta gamma delta", ["beta", "delta"])
        assert _split_for_docx_annotation(page) == [
            ("Alpha ", False),
            ("beta", True),
            (" gamma ", False),
            ("delta", True),
        ]

    def test_leading_match(self) -> None:
        page = _page("beta tail", ["beta"])
        assert _split_for_docx_annotation(page) == [
            ("beta", True),
            (" tail", False),
        ]

    def test_no_matches(self) -> None:
        page = _page("Nothing to flag", ["missing"])
        assert _split_for_docx_annotation(page) == [
            ("Nothing to flag", False),
        ]

    def test_empty_low_conf_returns_single_span(self) -> None:
        page = _page("Some text", [])
        assert _split_for_docx_annotation(page) == [("Some text", False)]


class TestExportTxtAnnotation:
    def test_annotation_off_by_default(self, tmp_path: Path) -> None:
        result = JobResult(
            job_id="j",
            status=JobStatus.COMPLETED,
            input_path="/in.pdf",
            output_path="/out.pdf",
            pages=[
                PageResult(
                    page_number=1,
                    text="The quick brown fox",
                    low_confidence_words=["quick", "fox"],
                ),
            ],
        )
        out = tmp_path / "out.txt"
        ExportManager().export_txt(result, out)
        body = out.read_text(encoding="utf-8")
        assert "The quick brown fox" in body
        assert "[?quick?]" not in body

    def test_annotation_on_wraps_low_conf_words(
        self, tmp_path: Path
    ) -> None:
        result = JobResult(
            job_id="j",
            status=JobStatus.COMPLETED,
            input_path="/in.pdf",
            output_path="/out.pdf",
            pages=[
                PageResult(
                    page_number=1,
                    text="Hello world fox",
                    low_confidence_words=["world"],
                ),
            ],
        )
        out = tmp_path / "out.txt"
        ExportManager().export_txt(result, out, annotate_low_conf=True)
        body = out.read_text(encoding="utf-8")
        assert "Hello [?world?] fox" in body


class TestExportDocxAnnotation:
    def test_annotation_produces_red_run(self, tmp_path: Path) -> None:
        pytest.importorskip("docx")
        from docx import Document
        from docx.shared import RGBColor

        result = JobResult(
            job_id="j",
            status=JobStatus.COMPLETED,
            input_path="/in.pdf",
            output_path="/out.pdf",
            pages=[
                PageResult(
                    page_number=1,
                    text="The quick brown fox",
                    low_confidence_words=["quick"],
                ),
            ],
        )
        out = tmp_path / "out.docx"
        ExportManager().export_docx(result, out, annotate_low_conf=True)

        doc = Document(str(out))
        red_runs = [
            run.text
            for p in doc.paragraphs
            for run in p.runs
            if run.font.color is not None
            and run.font.color.rgb == RGBColor(0xCC, 0x00, 0x00)
        ]
        assert "quick" in red_runs, (
            f"Expected 'quick' as a red run; got red_runs={red_runs!r}"
        )
