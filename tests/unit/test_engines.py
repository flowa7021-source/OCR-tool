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
