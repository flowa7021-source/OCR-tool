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
        profile.ocr.engine = OCREngineKind.TESSERACT
        d = profile.to_dict()
        assert d["ocr"]["engine"] == "tesseract"

    def test_engine_deserializes_from_dict(self) -> None:
        from src.core.models import ProfileData

        original = ProfileData(name="test")
        original.ocr.engine = OCREngineKind.TESSERACT
        restored = ProfileData.from_dict(original.to_dict())
        assert restored.ocr.engine is OCREngineKind.TESSERACT


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

    def test_list_engines_includes_tesseract(self) -> None:
        listing = list_engines()
        kinds = [item[0] for item in listing]
        assert OCREngineKind.TESSERACT in kinds


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
        """Happy-path: engine OCRs each page individually."""
        import shutil

        import fitz

        engine = TesseractEngine()

        # Build a real 3-page PDF so the per-page splitting works.
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page(width=200, height=200)
            page.insert_text((10, 50), f"page {i + 1}", fontsize=12)
        in_pdf = tmp_path / "in.pdf"
        doc.save(str(in_pdf))
        doc.close()

        out = tmp_path / "out.pdf"

        # run_ocrmypdf mock: just copy input → output so the merge
        # finds a valid PDF.
        def _fake_run(opts):
            shutil.copy2(str(opts.input_file), str(opts.output_file))

        with patch.object(engine, "is_available", return_value=(True, "")), \
             patch("src.application.engines.tesseract_engine.run_ocrmypdf", side_effect=_fake_run):
            results = engine.run(
                preprocessed_pdf=in_pdf,
                output_pdf=out,
                config=OCRConfig(),
            )

        assert len(results) == 3
        assert all(isinstance(r, PageOCRResult) for r in results)
        assert [r.page_number for r in results] == [1, 2, 3]
        assert out.exists()

    def test_progress_callback_fires(self, tmp_path: Path) -> None:
        import fitz

        engine = TesseractEngine()

        doc = fitz.open()
        doc.new_page(width=100, height=100)
        in_pdf = tmp_path / "in.pdf"
        doc.save(str(in_pdf))
        doc.close()

        out = tmp_path / "out.pdf"
        events: list[tuple[int, int, str]] = []

        def _fake_run(opts):
            # Write a PDF with a real text layer — the engine now
            # verifies the output PDF isn't an empty shell before
            # accepting the primary attempt. A plain ``copy2`` of
            # an empty raster would make the engine treat the
            # primary as failed and try the retry tiers.
            d = fitz.open()
            try:
                page = d.new_page(width=200, height=200)
                page.insert_text((10, 50), "ocr'd text", fontsize=12)
                d.save(str(opts.output_file))
            finally:
                d.close()

        with patch.object(engine, "is_available", return_value=(True, "")), \
             patch("src.application.engines.tesseract_engine.run_ocrmypdf", side_effect=_fake_run):
            engine.run(
                preprocessed_pdf=in_pdf,
                output_pdf=out,
                config=OCRConfig(),
                progress_callback=lambda c, t, s: events.append((c, t, s)),
            )
        # Should have start (0, 1, "ocr") and end (1, 1, "ocr").
        assert events[0] == (0, 1, "ocr")
        assert events[-1] == (1, 1, "ocr")

    def test_failing_page_is_retried_with_aggressive_preprocessing(
        self, tmp_path: Path
    ) -> None:
        """When a page crashes on primary settings, the engine MUST
        first retry with aggressive preprocessing at the original DPI
        (Sauvola + CLAHE 3.0 + background removal + NLM) before
        dropping to the simpler-settings tier.

        User requirement: every page must end up with a text layer —
        no silent "kept as raster" for pages the user expects to
        search. The aggressive retry targets faded / noisy scans
        where the user's profile preprocessing wasn't aggressive
        enough; the subsequent simpler-settings retry targets
        layout-crash cases where the problem is too MUCH pixel
        density.
        """
        import fitz

        engine = TesseractEngine()

        doc = fitz.open()
        for i in range(3):
            page = doc.new_page(width=200, height=200)
            page.insert_text((10, 50), f"page {i + 1}", fontsize=12)
        in_pdf = tmp_path / "in.pdf"
        doc.save(str(in_pdf))
        doc.close()

        out = tmp_path / "out.pdf"
        call_log: list[tuple[str, int]] = []
        aggressive_called = [0]

        def _write_text_pdf(path):
            """Write a tiny PDF with a real text layer."""
            d = fitz.open()
            try:
                page = d.new_page(width=200, height=200)
                page.insert_text((10, 50), "mocked ocr", fontsize=12)
                d.save(str(path))
            finally:
                d.close()

        def _fake_run(opts):
            """Fail on page 2 primary attempt; succeed everywhere else.
            The aggressive retry no longer goes through run_ocrmypdf —
            it uses pytesseract.image_to_pdf_or_hocr directly — so
            run_ocrmypdf is NEVER called on the ``*_aggressive.pdf``
            name and the fake's job is just to fail once on the
            failing page."""
            name = opts.input_file.name
            call_log.append((name, opts.psm))
            if name == "page_0002.pdf":
                raise RuntimeError("simulated Tesseract layout crash")
            _write_text_pdf(opts.output_file)

        def _fake_image_to_pdf(*args, **kwargs):
            """Aggressive retry's direct pytesseract call. Return a
            minimal 1-page PDF with a text layer so the retry is
            recorded as a successful recovery."""
            aggressive_called[0] += 1
            d = fitz.open()
            try:
                page = d.new_page(width=200, height=200)
                page.insert_text(
                    (10, 50), "aggressive recovery", fontsize=12,
                )
                buf = d.tobytes()
            finally:
                d.close()
            return buf

        with patch.object(engine, "is_available", return_value=(True, "")), \
             patch(
                 "src.application.engines.tesseract_engine.run_ocrmypdf",
                 side_effect=_fake_run,
             ), \
             patch(
                 "pytesseract.image_to_pdf_or_hocr",
                 side_effect=_fake_image_to_pdf,
             ):
            results = engine.run(
                preprocessed_pdf=in_pdf,
                output_pdf=out,
                config=OCRConfig(),
            )

        # All 3 pages reported back.
        assert len(results) == 3
        assert out.exists()

        # The aggressive-preprocessing retry must have been issued —
        # pytesseract.image_to_pdf_or_hocr is called once per retry
        # attempt, which should be exactly once (on the failing page 2).
        assert aggressive_called[0] >= 1, (
            "Expected the aggressive-preprocessing retry to invoke "
            f"pytesseract.image_to_pdf_or_hocr at least once, but "
            f"got {aggressive_called[0]} calls; run_ocrmypdf log was "
            f"{call_log!r}"
        )

    def test_failing_page_retry_escalates_to_last_resort_tier(
        self, tmp_path: Path
    ) -> None:
        """If both primary and simplified retry fail, the engine must
        try one final last-resort tier (PSM=SPARSE_TEXT, 150 DPI)
        before falling back to raster-only.

        User requirement: "все страницы всегда распознаны" — every
        page gets a text layer. Sparse-text PSM is the most tolerant
        mode Tesseract offers; it almost never crashes on weird
        layouts (stamps, rotated tables, mixed handwriting).
        """
        import fitz

        from src.shared.types import PSM

        engine = TesseractEngine()

        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.insert_text((10, 50), "one page", fontsize=12)
        in_pdf = tmp_path / "in.pdf"
        doc.save(str(in_pdf))
        doc.close()

        out = tmp_path / "out.pdf"
        call_log: list[tuple[str, int]] = []

        def _fake_run(opts):
            name = opts.input_file.name
            call_log.append((name, opts.psm))
            # Primary and simplified both crash; last-resort succeeds
            # by writing a PDF with a REAL text layer (the engine now
            # rejects empty-output success as a silent-timeout).
            if "lastresort" in name:
                d = fitz.open()
                try:
                    p = d.new_page(width=200, height=200)
                    p.insert_text((10, 50), "sparse text recovery", fontsize=12)
                    d.save(str(opts.output_file))
                finally:
                    d.close()
                return
            raise RuntimeError("simulated layout crash")

        def _fake_image_to_pdf(*args, **kwargs):
            """Aggressive retry goes through pytesseract directly.
            Fail it by returning a minimal PDF without any text layer
            so the engine escalates to the simpler-settings tier."""
            d = fitz.open()
            try:
                d.new_page(width=200, height=200)
                buf = d.tobytes()
            finally:
                d.close()
            return buf

        with patch.object(engine, "is_available", return_value=(True, "")), \
             patch("src.application.engines.tesseract_engine.run_ocrmypdf", side_effect=_fake_run), \
             patch("pytesseract.image_to_pdf_or_hocr", side_effect=_fake_image_to_pdf):
            engine.run(
                preprocessed_pdf=in_pdf,
                output_pdf=out,
                config=OCRConfig(),
            )

        # Last-resort tier must have run with PSM=SPARSE_TEXT.
        last_resort = [entry for entry in call_log if "lastresort" in entry[0]]
        assert last_resort, (
            f"Expected last-resort retry, but call log was {call_log!r}"
        )
        assert last_resort[0][1] == int(PSM.SPARSE_TEXT), (
            f"Last-resort tier should use PSM=SPARSE_TEXT, got "
            f"psm={last_resort[0][1]}"
        )
        assert out.exists()

    def test_all_tiers_fail_raises_only_when_every_page_failed(
        self, tmp_path: Path
    ) -> None:
        """0/N error fires ONLY when every tier failed on every page.

        Mixed results (some pages recovered via retry, some raster)
        still produce a usable PDF and must not raise — the user
        gets a searchable PDF for the pages Tesseract could handle.
        """
        import fitz

        from src.application.ocrmypdf_integration import OCRmyPDFError

        engine = TesseractEngine()

        doc = fitz.open()
        for _ in range(2):
            doc.new_page(width=200, height=200)
        in_pdf = tmp_path / "in.pdf"
        doc.save(str(in_pdf))
        doc.close()

        out = tmp_path / "out.pdf"

        def _always_fail(opts):
            raise RuntimeError("every attempt crashes")

        with patch.object(engine, "is_available", return_value=(True, "")), \
             patch("src.application.engines.tesseract_engine.run_ocrmypdf", side_effect=_always_fail), \
             pytest.raises(OCRmyPDFError) as excinfo:
            engine.run(
                preprocessed_pdf=in_pdf,
                output_pdf=out,
                config=OCRConfig(),
            )

        message = str(excinfo.value)
        # The refreshed message must mention that even the retry
        # with simplified settings couldn't recover the doc.
        assert "автоматическ" in message.lower() or "упрощ" in message.lower()


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

        # Write a non-empty PDF as the "OCRmyPDF output" — otherwise
        # the retry-on-empty-output path (added for silent-tesseract-
        # timeout detection) would kick in and we'd see 2 calls.
        def _write_non_empty_output(input_f, output_f, **kw):
            import fitz

            doc = fitz.open()
            try:
                page = doc.new_page(width=200, height=200)
                page.insert_text((10, 50), "ok", fontsize=12)
                doc.save(str(output_f))
            finally:
                doc.close()

        fake_ocrmypdf = MagicMock()
        fake_ocrmypdf.ocr = MagicMock(side_effect=_write_non_empty_output)

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


    def test_graft_hocr_miss_triggers_auto_retry_with_longer_timeout(
        self, tmp_path: Path
    ) -> None:
        """First-attempt graft-hocr-miss → automatic second call with 3× timeout.

        When Tesseract exceeds ``tesseract_timeout`` on a page, OCRmyPDF
        logs ``took too long to OCR - skipping`` and then crashes deep
        in the graft phase with a ``FileNotFoundError`` pointing at a
        missing ``*_ocr_hocr.hocr``. The wrapper must NOT surface that
        error straight to the UI — it must first retry once with a
        longer timeout. Only if the retry also fails do we bubble up
        to the user.

        This test exercises the "retry succeeds" path: first call
        raises the graft-hocr-miss shape, second call returns cleanly,
        and ``run_ocrmypdf`` completes without raising.
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
            language="rus+eng",
            oem=1,
            psm=3,
            optimize=1,
            skip_text=True,
            tesseract_timeout=120,
        )

        graft_path = (
            r"C:\Users\USER~1.020\AppData\Local\Temp\ocrmypdf.io.abcd\000003_ocr_hocr.hocr"
        )
        graft_err = FileNotFoundError(2, "No such file", graft_path)

        fake_ocrmypdf = MagicMock()
        # First call fails with graft-hocr-miss, second call succeeds.
        fake_ocrmypdf.ocr = MagicMock(side_effect=[graft_err, None])

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
        ):
            run_ocrmypdf(options)  # must NOT raise

        # Exactly two ocrmypdf.ocr calls: the original, then the retry.
        assert fake_ocrmypdf.ocr.call_count == 2
        first_call = fake_ocrmypdf.ocr.call_args_list[0]
        retry_call = fake_ocrmypdf.ocr.call_args_list[1]

        # First call used the user-configured timeout and (by default)
        # use_threads=True.
        assert first_call.kwargs["tesseract_timeout"] == 120
        assert first_call.kwargs["use_threads"] is True

        # Retry: timeout escalated to ≥ max(base*3, 600) = 600.
        # ``use_threads`` is NOT changed (OCRmyPDF 16.x upstream bug
        # crashes when ``use_threads=False``).
        assert retry_call.kwargs["tesseract_timeout"] >= 600
        assert retry_call.kwargs["tesseract_timeout"] == 600, (
            "escalation should be min(max(base*3, 600), 900); "
            f"got {retry_call.kwargs['tesseract_timeout']}"
        )
        assert retry_call.kwargs["use_threads"] is True

    def test_graft_hocr_miss_persists_through_retry_raises_clear_error(
        self, tmp_path: Path
    ) -> None:
        """If the retry also fails, emit the user-facing Russian hint.

        The user-facing message must name the knobs they can turn (DPI,
        ``tesseract_timeout``) and must NOT leak the raw temp path that
        OCRmyPDF's graft phase points at.
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

        graft_path = (
            r"C:\Users\USER~1.020\AppData\Local\Temp\ocrmypdf.io.abcd\000003_ocr_hocr.hocr"
        )
        fake_ocrmypdf = MagicMock()
        # Both calls fail with the same shape.
        fake_ocrmypdf.ocr = MagicMock(
            side_effect=[
                FileNotFoundError(2, "No such file", graft_path),
                FileNotFoundError(2, "No such file", graft_path),
            ]
        )

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
        # Must mention retry happened + name the user-tunable knobs.
        assert "повтор" in message.lower() or "автоматическ" in message.lower()
        assert "DPI" in message, message
        assert "tesseract_timeout" in message, message
        # Must NOT leak the raw temp path.
        assert "000003_ocr_hocr" not in message
        assert "WinError" not in message
        # Both attempts ran.
        assert fake_ocrmypdf.ocr.call_count == 2

    def test_empty_output_pdf_triggers_auto_retry_with_longer_timeout(
        self, tmp_path: Path
    ) -> None:
        """First attempt returns COMPLETED but an empty-text PDF →
        auto-retry with escalated timeout.

        Regression: Tesseract silently times out on aggressive
        preprocessing by writing an empty hOCR that OCRmyPDF grafts
        without complaint. Result: JobStatus.COMPLETED with zero
        recognised text, no FileNotFoundError, nothing for the
        ``_is_graft_hocr_miss`` heuristic to catch. The user sees an
        empty searchable PDF and can't tell what went wrong.

        Fix: ``_invoke_ocrmypdf_with_timeout_retry`` now checks the
        output PDF for any text after a "successful" attempt. Empty
        → treat as silent timeout → retry with 600 s + single-threaded.
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
            language="rus+eng",
            oem=1,
            psm=3,
            optimize=1,
            skip_text=True,
            tesseract_timeout=120,
        )

        # First call: "succeeds" — write an empty-text PDF on disk.
        # Second call: "succeeds" — write a non-empty PDF.
        import fitz

        def write_empty_pdf(input_f, output_f, **kwargs):
            doc = fitz.open()
            try:
                doc.new_page(width=100, height=100)  # blank page, no text
                doc.save(str(output_f))
            finally:
                doc.close()

        def write_pdf_with_text(input_f, output_f, **kwargs):
            doc = fitz.open()
            try:
                page = doc.new_page(width=200, height=200)
                page.insert_text((10, 50), "retry worked", fontsize=12)
                doc.save(str(output_f))
            finally:
                doc.close()

        fake_ocrmypdf = MagicMock()
        fake_ocrmypdf.ocr = MagicMock(
            side_effect=[write_empty_pdf(in_pdf, out_pdf), write_pdf_with_text(in_pdf, out_pdf)]
        )
        # MagicMock above already consumed the writes — reset to use
        # side_effect functions instead so fake_ocrmypdf.ocr calls
        # our callables when invoked.
        fake_ocrmypdf.ocr = MagicMock(
            side_effect=[write_empty_pdf, write_pdf_with_text]
        )

        # Wrap each side_effect fn with the ``(input, output, **kw)``
        # calling convention the wrapper uses.
        def _first(input_f, output_f, **kw):
            write_empty_pdf(input_f, output_f, **kw)

        def _second(input_f, output_f, **kw):
            write_pdf_with_text(input_f, output_f, **kw)

        fake_ocrmypdf.ocr = MagicMock(side_effect=[_first, _second])
        # MagicMock side_effect as a list of callables: each call
        # invokes the next callable with the same args. Wrap each
        # callable so MagicMock's side_effect dispatch works.
        fake_ocrmypdf.ocr = MagicMock(
            side_effect=lambda input_f, output_f, **kw: (
                _first(input_f, output_f, **kw) if fake_ocrmypdf.ocr.call_count == 1
                else _second(input_f, output_f, **kw)
            )
        )

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
        ):
            run_ocrmypdf(options)  # must NOT raise

        assert fake_ocrmypdf.ocr.call_count == 2, (
            "Expected exactly 2 calls — first empty, then retry"
        )
        # Second call kwargs reflect the escalation.
        retry_kwargs = fake_ocrmypdf.ocr.call_args_list[1].kwargs
        assert retry_kwargs["tesseract_timeout"] >= 600
        # use_threads must NOT be changed — OCRmyPDF 16.x crashes
        # with use_threads=False.
        assert retry_kwargs["use_threads"] is True

    def test_non_empty_output_skips_retry(self, tmp_path: Path) -> None:
        """The happy path must not retry — that would double every
        job's wall time."""
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
            tesseract_timeout=60,
        )

        import fitz

        def _first(input_f, output_f, **kw):
            doc = fitz.open()
            try:
                page = doc.new_page(width=200, height=200)
                page.insert_text((10, 50), "clean output", fontsize=12)
                doc.save(str(output_f))
            finally:
                doc.close()

        fake_ocrmypdf = MagicMock()
        fake_ocrmypdf.ocr = MagicMock(side_effect=_first)

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
        ):
            run_ocrmypdf(options)

        # Exactly ONE call — no wasteful retry on the happy path.
        assert fake_ocrmypdf.ocr.call_count == 1

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
        # And CRITICALLY — no wasteful retry for this failure shape.
        # Retrying with a longer timeout wouldn't help when the
        # Tesseract binary itself is missing; it would just double
        # the user's wait before giving up.
        assert fake_ocrmypdf.ocr.call_count == 1


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
