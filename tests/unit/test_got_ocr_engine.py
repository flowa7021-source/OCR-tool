"""Tests for the GOT-OCR2 engine wrapper.

Heavy ML deps (torch, transformers) are not required to run these —
imports are lazy and patched.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.application.engines.base import EngineNotAvailableError, PageOCRResult
from src.application.engines.got_ocr_engine import GOTOCREngine
from src.core.models import OCRConfig
from src.infrastructure.model_manager import GOT_OCR2_SPEC, ModelManager
from src.shared.types import OCREngineKind


@pytest.fixture
def manager(tmp_path: Path) -> ModelManager:
    return ModelManager(models_dir=tmp_path / "models")


def _seed_manifest(manager: ModelManager) -> None:
    """Pretend the GOT-OCR2 weights have already been downloaded."""
    target = manager.model_dir(GOT_OCR2_SPEC.model_id)
    target.mkdir(parents=True)
    for f in GOT_OCR2_SPEC.files:
        # 64-byte placeholder per file — large enough to defeat the
        # ModelManager's "tiny file" sanity check but small enough to
        # keep the test cheap.
        path = target / f.name
        if f.size_bytes:
            # Match the expected size precisely so size validation passes.
            path.write_bytes(b"\x00" * f.size_bytes)
        else:
            path.write_bytes(b"\x00" * 2048)


def _stub_torch_transformers(monkeypatch) -> None:
    """Inject minimal torch + transformers modules so import succeeds.

    Also stubs ``einops`` and ``accelerate`` — these are transitive
    deps of GOT-OCR 2.0's ``trust_remote_code`` scripts that the
    engine's ``is_available`` probes for. And stubs ``torch.zeros``
    so the DLL-load smoke test (``torch.zeros(1)``) in ``is_available``
    doesn't fail on the mock.
    """
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        # Smoke test calls torch.zeros(1) to catch bundled-DLL failures.
        zeros=lambda *a, **kw: object(),
    )
    fake_torch.__name__ = "torch"
    fake_transformers = types.SimpleNamespace(
        AutoModel=types.SimpleNamespace(from_pretrained=lambda *a, **kw: MagicMock()),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: MagicMock()),
    )
    fake_transformers.__name__ = "transformers"
    fake_einops = types.SimpleNamespace()
    fake_einops.__name__ = "einops"
    fake_accelerate = types.SimpleNamespace()
    fake_accelerate.__name__ = "accelerate"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "einops", fake_einops)
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)


# ---------------------------------------------------------------------------
# Metadata + availability
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_kind_and_name(self, manager: ModelManager) -> None:
        e = GOTOCREngine(model_manager=manager)
        assert e.kind is OCREngineKind.GOT_OCR2
        assert "GOT" in e.name
        assert e.description


class TestAvailability:
    def test_missing_torch_returns_install_hint(
        self, manager: ModelManager, monkeypatch
    ) -> None:
        # Ensure neither torch nor transformers is importable
        monkeypatch.setitem(sys.modules, "torch", None)
        e = GOTOCREngine(model_manager=manager)
        ok, msg = e.is_available()
        assert ok is False
        assert "htr" in msg.lower() or "torch" in msg.lower()

    def test_missing_weights_returns_download_hint(
        self, manager: ModelManager, monkeypatch
    ) -> None:
        _stub_torch_transformers(monkeypatch)
        # No files seeded
        e = GOTOCREngine(model_manager=manager)
        ok, msg = e.is_available()
        assert ok is False
        assert "Скачать" in msg or "модель" in msg.lower()

    def test_all_present_returns_ok(
        self, manager: ModelManager, monkeypatch
    ) -> None:
        _stub_torch_transformers(monkeypatch)
        _seed_manifest(manager)
        e = GOTOCREngine(model_manager=manager)
        ok, msg = e.is_available()
        assert ok is True
        assert msg == ""


# ---------------------------------------------------------------------------
# run() with mocked model + PyMuPDF
# ---------------------------------------------------------------------------


class TestRun:
    def test_raises_when_unavailable(
        self, manager: ModelManager, tmp_path: Path
    ) -> None:
        e = GOTOCREngine(model_manager=manager)
        with pytest.raises(EngineNotAvailableError):
            e.run(
                preprocessed_pdf=tmp_path / "in.pdf",
                output_pdf=tmp_path / "out.pdf",
                config=OCRConfig(),
            )

    def test_happy_path_emits_per_page_text(
        self, manager: ModelManager, monkeypatch, tmp_path: Path
    ) -> None:
        _stub_torch_transformers(monkeypatch)
        _seed_manifest(manager)

        # Skip _load_model (its signature requires real torch internals)
        monkeypatch.setattr(GOTOCREngine, "_load_model", lambda self: None)
        # Stub _recognize_page so we don't need PIL or the real model.
        monkeypatch.setattr(
            GOTOCREngine,
            "_recognize_page",
            lambda self, page: (f"page {id(page) % 100}", 80.0),
        )
        # Stub _append_page_with_overlay so PyMuPDF doesn't need to draw.
        monkeypatch.setattr(
            GOTOCREngine, "_append_page_with_overlay", lambda *a, **kw: None
        )

        # Fake the source PDF as a 3-page document
        fake_src = MagicMock()
        fake_src.page_count = 3
        fake_src.load_page.side_effect = lambda i: MagicMock(name=f"p{i}")
        fake_out = MagicMock()
        with patch("fitz.open") as fitz_open:
            fitz_open.side_effect = [fake_src, fake_out]
            e = GOTOCREngine(model_manager=manager)
            events: list[tuple[int, int, str]] = []
            results = e.run(
                preprocessed_pdf=tmp_path / "in.pdf",
                output_pdf=tmp_path / "out.pdf",
                config=OCRConfig(),
                progress_callback=lambda c, t, s: events.append((c, t, s)),
            )

        assert len(results) == 3
        assert all(isinstance(r, PageOCRResult) for r in results)
        assert all(r.text.startswith("page ") for r in results)
        assert all(r.mean_confidence == 80.0 for r in results)
        # Progress reports start (0, n) and end (n, n)
        assert events[0] == (0, 3, "got-ocr")
        assert events[-1] == (3, 3, "got-ocr")
        # Output document was saved + closed
        fake_out.save.assert_called_once()
        fake_out.close.assert_called_once()
