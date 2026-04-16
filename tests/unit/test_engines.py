"""Tests for the OCR engine abstraction layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.application.engines import get_engine, list_engines
from src.application.engines.base import (
    EngineNotAvailableError,
    OCREngine,
    PageOCRResult,
)
from src.application.engines.registry import reset_cache
from src.application.engines.tesseract_engine import TesseractEngine
from src.core.models import OCRConfig
from src.shared.types import OCREngineKind


@pytest.fixture(autouse=True)
def _clean_engine_cache() -> None:
    """Each test starts with a fresh engine cache so probes are honest."""
    reset_cache()
    yield
    reset_cache()


# ---------------------------------------------------------------------------
# OCRConfig.engine field
# ---------------------------------------------------------------------------


class TestOCRConfigEngineField:
    def test_default_is_tesseract(self) -> None:
        cfg = OCRConfig()
        assert cfg.engine is OCREngineKind.TESSERACT

    def test_engine_serializes_in_dict(self) -> None:
        from src.core.models import ProfileData

        profile = ProfileData(name="test")
        profile.ocr.engine = OCREngineKind.GOT_OCR2
        d = profile.to_dict()
        assert d["ocr"]["engine"] == "got_ocr2"

    def test_engine_deserializes_from_dict(self) -> None:
        from src.core.models import ProfileData

        original = ProfileData(name="test")
        original.ocr.engine = OCREngineKind.GOT_OCR2
        restored = ProfileData.from_dict(original.to_dict())
        assert restored.ocr.engine is OCREngineKind.GOT_OCR2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_get_tesseract_returns_engine(self) -> None:
        engine = get_engine(OCREngineKind.TESSERACT)
        assert isinstance(engine, TesseractEngine)
        assert engine.kind is OCREngineKind.TESSERACT

    def test_get_caches_instance(self) -> None:
        a = get_engine(OCREngineKind.TESSERACT)
        b = get_engine(OCREngineKind.TESSERACT)
        assert a is b

    def test_got_ocr_resolves_but_reports_unavailable(self) -> None:
        # Module is registered but torch/transformers + weights are not
        # installed by default, so the engine must self-report as
        # unavailable with an actionable hint.
        engine = get_engine(OCREngineKind.GOT_OCR2)
        assert engine.kind is OCREngineKind.GOT_OCR2
        ok, msg = engine.is_available()
        assert ok is False
        assert msg  # non-empty Russian hint

    def test_list_engines_includes_all_kinds(self) -> None:
        listing = list_engines()
        kinds = [item[0] for item in listing]
        assert OCREngineKind.TESSERACT in kinds
        assert OCREngineKind.GOT_OCR2 in kinds

    def test_list_engines_marks_unavailable(self) -> None:
        listing = list_engines()
        got_entry = next(item for item in listing if item[0] is OCREngineKind.GOT_OCR2)
        # GOT-OCR2 engine module isn't shipped yet → must be marked
        # unavailable with a non-empty hint message.
        _, _name, available, msg = got_entry
        assert available is False
        assert len(msg) > 0


# ---------------------------------------------------------------------------
# TesseractEngine
# ---------------------------------------------------------------------------


class TestTesseractEngine:
    def test_metadata(self) -> None:
        engine = TesseractEngine()
        assert engine.kind is OCREngineKind.TESSERACT
        assert "Tesseract" in engine.name
        assert engine.description

    def test_run_raises_when_unavailable(self, tmp_path: Path) -> None:
        engine = TesseractEngine()
        with patch.object(engine, "is_available", return_value=(False, "stub: missing")):
            with pytest.raises(EngineNotAvailableError) as exc_info:
                engine.run(
                    preprocessed_pdf=tmp_path / "in.pdf",
                    output_pdf=tmp_path / "out.pdf",
                    config=OCRConfig(),
                )
            assert "stub: missing" in str(exc_info.value)

    def test_run_invokes_ocrmypdf(self, tmp_path: Path) -> None:
        """Happy-path: engine wires straight through to run_ocrmypdf."""
        engine = TesseractEngine()
        out = tmp_path / "out.pdf"
        out.write_bytes(b"%PDF-1.7\n")  # so fitz.open succeeds afterwards

        fake_doc = MagicMock()
        fake_doc.page_count = 3

        with patch.object(engine, "is_available", return_value=(True, "")), \
             patch("src.application.engines.tesseract_engine.run_ocrmypdf") as run_mock, \
             patch("src.application.engines.tesseract_engine.map_ocr_config") as map_mock, \
             patch("fitz.open", return_value=fake_doc):
            map_mock.return_value = "OPTS"
            results = engine.run(
                preprocessed_pdf=tmp_path / "in.pdf",
                output_pdf=out,
                config=OCRConfig(),
            )

        run_mock.assert_called_once_with("OPTS")
        assert len(results) == 3
        assert all(isinstance(r, PageOCRResult) for r in results)
        assert [r.page_number for r in results] == [1, 2, 3]

    def test_progress_callback_fires(self, tmp_path: Path) -> None:
        engine = TesseractEngine()
        out = tmp_path / "out.pdf"
        out.write_bytes(b"%PDF-1.7\n")
        events: list[tuple[int, int, str]] = []
        fake_doc = MagicMock()
        fake_doc.page_count = 1

        with patch.object(engine, "is_available", return_value=(True, "")), \
             patch("src.application.engines.tesseract_engine.run_ocrmypdf"), \
             patch("src.application.engines.tesseract_engine.map_ocr_config"), \
             patch("fitz.open", return_value=fake_doc):
            engine.run(
                preprocessed_pdf=tmp_path / "in.pdf",
                output_pdf=out,
                config=OCRConfig(),
                progress_callback=lambda c, t, s: events.append((c, t, s)),
            )
        assert events == [(0, 1, "ocr"), (1, 1, "ocr")]


class TestRunOcrmypdfIntegration:
    """Guards around ``run_ocrmypdf`` — the OCRmyPDF wrapper.

    Separate from TesseractEngine tests because they patch
    ``run_ocrmypdf`` wholesale; these poke the wrapper directly.
    """

    def test_ocr_called_positionally(self, tmp_path: Path) -> None:
        """input/output paths must be passed POSITIONALLY to ocrmypdf.ocr.

        Regression: OCRmyPDF 17 renamed the first parameter from
        ``input_file`` to ``input_file_or_options``. Passing either
        name as a keyword breaks on one or both versions. Passing the
        paths positionally is the only forward-compatible call
        convention. End-user logs showed:
          ``TypeError: ocr() missing 1 required positional argument:
          'input_file_or_options'``
        when the packaged OCRmyPDF was 17.x.
        """
        from src.application.ocrmypdf_integration import (
            OCRmyPDFOptions,
            run_ocrmypdf,
        )

        in_pdf = tmp_path / "in.pdf"
        in_pdf.write_bytes(b"%PDF-1.7\n")
        out_pdf = tmp_path / "out.pdf"

        options = OCRmyPDFOptions(
            input_file=in_pdf,
            output_file=out_pdf,
            language="eng",
            oem=1,
            psm=3,
            optimize=1,
            skip_text=True,
            tesseract_timeout=600,
        )

        fake_ocrmypdf = MagicMock()
        fake_ocrmypdf.ocr = MagicMock()

        class _FakeExitCodeError(Exception):  # stand-in for ExitCodeException
            exit_code = 0

        fake_exceptions = MagicMock()
        fake_exceptions.ExitCodeException = _FakeExitCodeError

        import sys

        with patch.dict(
            sys.modules,
            {
                "ocrmypdf": fake_ocrmypdf,
                "ocrmypdf.exceptions": fake_exceptions,
            },
        ):
            run_ocrmypdf(options)

        assert fake_ocrmypdf.ocr.call_count == 1
        call = fake_ocrmypdf.ocr.call_args
        # First two args MUST be positional input/output paths, not kwargs.
        assert len(call.args) == 2, (
            f"ocrmypdf.ocr should be called with 2 positional args "
            f"(input_file, output_file), got {len(call.args)}: {call.args!r}"
        )
        assert call.args[0] == str(in_pdf)
        assert call.args[1] == str(out_pdf)
        # And neither name should appear in kwargs — both are
        # positional, and any stale ``input_file`` kwarg would crash on
        # OCRmyPDF 17+ with the same TypeError.
        assert "input_file" not in call.kwargs
        assert "output_file" not in call.kwargs
        assert "input_file_or_options" not in call.kwargs
        # Sanity: our preprocessing-disabling kwargs survived.
        assert call.kwargs.get("deskew") is False
        assert call.kwargs.get("language") == "eng"


    def test_graft_hocr_miss_maps_to_tesseract_timeout_hint(
        self, tmp_path: Path
    ) -> None:
        """A timed-out page triggers a clear error, not the raw WinError 2.

        Regression: when Tesseract exceeds ``tesseract_timeout`` on a
        page, OCRmyPDF logs ``took too long to OCR - skipping`` and
        then crashes deep in the graft phase because it still tries
        to stat the per-page HOCR that was never produced::

            File "ocrmypdf/_graft.py", line 350, in _parse_hocr_pages
              File "pathlib.py", line 1013, in stat
            FileNotFoundError: [WinError 2] ... '000003_ocr_hocr.hocr'

        End users saw ``OCRmyPDF failed: [WinError 2] ...`` pointing at
        a temp path they cannot do anything about. The wrapper now
        intercepts that specific FileNotFoundError shape and emits
        Russian guidance naming ``tesseract_timeout`` and DPI.
        """
        from src.application.ocrmypdf_integration import (
            OCRmyPDFError,
            OCRmyPDFOptions,
            run_ocrmypdf,
        )

        in_pdf = tmp_path / "in.pdf"
        in_pdf.write_bytes(b"%PDF-1.7\n")
        out_pdf = tmp_path / "out.pdf"

        options = OCRmyPDFOptions(
            input_file=in_pdf,
            output_file=out_pdf,
            language="rus+eng",
            oem=1,
            psm=3,
            optimize=1,
            skip_text=True,
            tesseract_timeout=120,
        )

        # Build a FileNotFoundError that matches what _graft raises —
        # errno=2 with a ``*_ocr_hocr.hocr`` filename. The wrapper
        # keys on the filename, not on any message string, so this is
        # robust to locale changes in the underlying WinError text.
        graft_path = (
            r"C:\Users\USER~1.020\AppData\Local\Temp\ocrmypdf.io.abcd\000003_ocr_hocr.hocr"
        )
        graft_err = FileNotFoundError(2, "No such file", graft_path)

        fake_ocrmypdf = MagicMock()
        fake_ocrmypdf.ocr = MagicMock(side_effect=graft_err)

        class _FakeExitCodeError(Exception):
            exit_code = 0

        fake_exceptions = MagicMock()
        fake_exceptions.ExitCodeException = _FakeExitCodeError

        import sys

        with patch.dict(
            sys.modules,
            {
                "ocrmypdf": fake_ocrmypdf,
                "ocrmypdf.exceptions": fake_exceptions,
            },
        ), pytest.raises(OCRmyPDFError) as excinfo:
            run_ocrmypdf(options)

        message = str(excinfo.value)
        # Must name the knob the user can turn.
        assert "tesseract_timeout" in message, message
        # Must include the configured value so the hint is concrete.
        assert "120" in message, message
        # Must NOT leak the raw temp path to the user (that's what the
        # old error did).
        assert "000003_ocr_hocr" not in message
        assert "WinError" not in message
        # Exception chain is preserved so debug logs still show the
        # original cause.
        assert excinfo.value.__cause__ is graft_err

    def test_unrelated_filenotfounderror_still_surfaces(
        self, tmp_path: Path
    ) -> None:
        """A non-graft FileNotFoundError must NOT get the timeout hint.

        If OCRmyPDF bails because its own dependency is missing
        (e.g. ``tesseract.exe`` gone from PATH mid-run), we must not
        mislead the user into tuning ``tesseract_timeout``.
        """
        from src.application.ocrmypdf_integration import (
            OCRmyPDFError,
            OCRmyPDFOptions,
            run_ocrmypdf,
        )

        in_pdf = tmp_path / "in.pdf"
        in_pdf.write_bytes(b"%PDF-1.7\n")
        out_pdf = tmp_path / "out.pdf"

        options = OCRmyPDFOptions(
            input_file=in_pdf,
            output_file=out_pdf,
            language="eng",
            oem=1,
            psm=3,
            optimize=1,
            skip_text=True,
            tesseract_timeout=60,
        )

        unrelated_err = FileNotFoundError(
            2, "No such file", "C:/Program Files/tesseract.exe"
        )

        fake_ocrmypdf = MagicMock()
        fake_ocrmypdf.ocr = MagicMock(side_effect=unrelated_err)

        class _FakeExitCodeError(Exception):
            exit_code = 0

        fake_exceptions = MagicMock()
        fake_exceptions.ExitCodeException = _FakeExitCodeError

        import sys

        with patch.dict(
            sys.modules,
            {
                "ocrmypdf": fake_ocrmypdf,
                "ocrmypdf.exceptions": fake_exceptions,
            },
        ), pytest.raises(OCRmyPDFError) as excinfo:
            run_ocrmypdf(options)

        message = str(excinfo.value)
        # The generic wrapper kicks in — no misleading timeout hint.
        assert "tesseract_timeout" not in message, message
        # But the original file path does surface so the user knows
        # what's actually missing.
        assert "tesseract.exe" in message, message


# ---------------------------------------------------------------------------
# Custom engine for the abstraction itself
# ---------------------------------------------------------------------------


class TestCustomEngine:
    def test_can_implement_subclass(self, tmp_path: Path) -> None:
        """Down-stream code can subclass OCREngine without touching internals."""

        class _Stub(OCREngine):
            kind = OCREngineKind.TESSERACT

            @property
            def name(self) -> str:
                return "Stub"

            @property
            def description(self) -> str:
                return "for tests"

            def is_available(self) -> tuple[bool, str]:
                return True, ""

            def run(self, preprocessed_pdf, output_pdf, config, progress_callback=None):
                return [PageOCRResult(page_number=1, text="hello")]

        e = _Stub()
        out = e.run(tmp_path / "in.pdf", tmp_path / "out.pdf", OCRConfig())
        assert out[0].text == "hello"
