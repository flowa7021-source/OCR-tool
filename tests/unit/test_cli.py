"""Tests for the command-line interface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import cli
from src.core.models import JobResult, PageResult
from src.shared.types import JobStatus

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_parser_accepts_single_pdf(self) -> None:
        args = cli.build_parser().parse_args(["file.pdf"])
        assert args.inputs == [Path("file.pdf")]
        assert args.profile == "default"
        assert args.workers == 1
        assert args.txt is False
        assert args.docx is False

    def test_output_flag(self) -> None:
        args = cli.build_parser().parse_args(["in.pdf", "-o", "out.pdf"])
        assert args.output == Path("out.pdf")

    def test_profile_flag(self) -> None:
        args = cli.build_parser().parse_args(["in.pdf", "-p", "contracts_ru"])
        assert args.profile == "contracts_ru"

    def test_verbose_counts(self) -> None:
        args = cli.build_parser().parse_args(["in.pdf", "-vv"])
        assert args.verbose == 2

    def test_list_profiles_standalone(self) -> None:
        args = cli.build_parser().parse_args(["--list-profiles"])
        assert args.list_profiles is True

    def test_version_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli.build_parser().parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_multiple_inputs(self) -> None:
        args = cli.build_parser().parse_args(["a.pdf", "b.pdf", "./dir/"])
        assert args.inputs == [Path("a.pdf"), Path("b.pdf"), Path("./dir/")]


# ---------------------------------------------------------------------------
# discover_inputs
# ---------------------------------------------------------------------------


class TestDiscoverInputs:
    def test_single_pdf_passthrough(self, tmp_path: Path) -> None:
        pdf = tmp_path / "one.pdf"
        pdf.write_bytes(b"%PDF-1.7\n")
        result = cli.discover_inputs([pdf])
        assert result == [pdf.resolve()]

    def test_directory_expansion(self, tmp_path: Path) -> None:
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.7\n")
        (tmp_path / "b.pdf").write_bytes(b"%PDF-1.7\n")
        (tmp_path / "not-a-pdf.txt").write_text("x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.pdf").write_bytes(b"%PDF-1.7\n")
        found = cli.discover_inputs([tmp_path])
        names = sorted(p.name for p in found)
        assert names == ["a.pdf", "b.pdf", "c.pdf"]

    def test_deduplicates(self, tmp_path: Path) -> None:
        pdf = tmp_path / "x.pdf"
        pdf.write_bytes(b"%PDF-1.7\n")
        result = cli.discover_inputs([pdf, pdf, tmp_path])
        # One physical file, regardless of reference count
        assert len([p for p in result if p.name == "x.pdf"]) == 1

    def test_non_pdf_is_skipped(self, tmp_path: Path) -> None:
        txt = tmp_path / "file.txt"
        txt.write_text("hi")
        assert cli.discover_inputs([txt]) == []

    def test_missing_file_is_skipped(self, tmp_path: Path) -> None:
        assert cli.discover_inputs([tmp_path / "nope.pdf"]) == []


# ---------------------------------------------------------------------------
# main() dispatcher
# ---------------------------------------------------------------------------


class TestMain:
    def test_list_profiles_returns_zero(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # Monkey-patch profile storage location via ProfileStorage arg
        rc = cli.main(["--list-profiles"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "профилей" in out.lower()

    def test_no_inputs_returns_nonzero(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            cli.main([])
        # argparse.error() exits with code 2
        assert exc_info.value.code == 2

    def test_no_pdfs_found_returns_nonzero(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        rc = cli.main([str(empty_dir)])
        assert rc == 2

    def test_output_with_multiple_inputs_errors(self, tmp_path: Path) -> None:
        a = tmp_path / "a.pdf"
        b = tmp_path / "b.pdf"
        a.write_bytes(b"%PDF-1.7\n")
        b.write_bytes(b"%PDF-1.7\n")
        with pytest.raises(SystemExit) as exc_info:
            cli.main([str(a), str(b), "-o", "out.pdf"])
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# process_single (mocked pipeline)
# ---------------------------------------------------------------------------


class TestProcessSingle:
    def test_unknown_profile_returns_2(self, tmp_path: Path) -> None:
        pdf = tmp_path / "in.pdf"
        pdf.write_bytes(b"%PDF-1.7\n")
        out = tmp_path / "out.pdf"
        rc = cli.process_single(
            pdf, out, "this_profile_does_not_exist_anywhere", False, False
        )
        assert rc == 2

    def test_successful_run_with_mocked_pipeline(self, tmp_path: Path) -> None:
        pdf = tmp_path / "in.pdf"
        pdf.write_bytes(b"%PDF-1.7\n")
        out = tmp_path / "out.pdf"

        fake_result = JobResult(
            job_id="j",
            status=JobStatus.COMPLETED,
            input_path=str(pdf),
            output_path=str(out),
            pages=[PageResult(page_number=1, text="hi", mean_confidence=90.0)],
            total_time_sec=0.1,
        )

        with patch("src.application.pipeline.OCRPipeline") as pipeline_cls:
            instance = MagicMock()
            instance.run.return_value = fake_result
            pipeline_cls.return_value = instance
            with patch("src.infrastructure.tesseract_wrapper.TesseractWrapper") as tw:
                tw.return_value.configure_pytesseract.return_value = None
                rc = cli.process_single(pdf, out, "default", False, False)

        assert rc == 0

    def test_failed_run_returns_1(self, tmp_path: Path) -> None:
        pdf = tmp_path / "in.pdf"
        pdf.write_bytes(b"%PDF-1.7\n")
        out = tmp_path / "out.pdf"

        fake_result = JobResult(
            job_id="j",
            status=JobStatus.FAILED,
            input_path=str(pdf),
            output_path=str(out),
            error="boom",
        )
        with patch("src.application.pipeline.OCRPipeline") as pipeline_cls:
            instance = MagicMock()
            instance.run.return_value = fake_result
            pipeline_cls.return_value = instance
            with patch("src.infrastructure.tesseract_wrapper.TesseractWrapper"):
                rc = cli.process_single(pdf, out, "default", False, False)
        assert rc == 1

    def test_txt_export_uses_export_manager(self, tmp_path: Path) -> None:
        pdf = tmp_path / "in.pdf"
        pdf.write_bytes(b"%PDF-1.7\n")
        out = tmp_path / "out.pdf"

        fake_result = JobResult(
            job_id="j",
            status=JobStatus.COMPLETED,
            input_path=str(pdf),
            output_path=str(out),
            pages=[PageResult(page_number=1, text="hi")],
        )

        with patch("src.application.pipeline.OCRPipeline") as pipeline_cls, \
             patch("src.application.export_manager.ExportManager") as em_cls, \
             patch("src.infrastructure.tesseract_wrapper.TesseractWrapper"):
            pipeline_cls.return_value.run.return_value = fake_result
            em_instance = MagicMock()
            em_cls.return_value = em_instance

            rc = cli.process_single(pdf, out, "default", True, False)

        assert rc == 0
        assert em_instance.export.called
