"""Tests for ``GOTOCREngine._safe_model_path``.

Regression guard for the HTR equivalent of the Tesseract Unicode bug:
when the bundled GOT-OCR 2.0 weights sit under a non-ASCII path like
``C:\\Users\\Т.Н. 020\\...``, some transformers/safetensors code paths
(C++/Rust extensions) have historically struggled. We preemptively
resolve to a Win32 short-path on Windows, and pass through untouched
elsewhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.exercise_preflight


def _engine():
    """Create the engine without hitting the model manager."""
    # model_manager is only used at load time; pass a plain MagicMock.
    from unittest.mock import MagicMock

    from src.application.engines.got_ocr_engine import GOTOCREngine

    return GOTOCREngine(model_manager=MagicMock())


def test_safe_path_ascii_passthrough(tmp_path: Path) -> None:
    """ASCII-only paths are returned unchanged on every platform."""
    engine = _engine()
    got = engine._safe_model_path(tmp_path / "models" / "got_ocr2")
    assert got == str(tmp_path / "models" / "got_ocr2")


def test_safe_path_posix_non_ascii_passthrough() -> None:
    """On POSIX we never try to shorten — pathlib is Unicode-safe."""
    if sys.platform == "win32":
        pytest.skip("Windows-specific behaviour covered in another test")
    engine = _engine()
    p = Path("/home/Т.Н. 020/models")
    assert engine._safe_model_path(p) == str(p)


def test_safe_path_falls_back_when_short_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    """If ``GetShortPathNameW`` can't resolve (returns 0), fall back.

    Simulated here by forcing ``os.name != "nt"`` so the Windows
    branch is skipped on the CI runner regardless of its platform.
    The test's real value is documenting that a failing short-path
    lookup must not crash ``_load_model`` — we return the original
    path and let Python's Unicode-aware I/O handle it.
    """
    engine = _engine()
    monkeypatch.setattr("src.application.engines.got_ocr_engine.os.name", "posix")
    p = tmp_path / "Т.Н._models"
    p.mkdir()
    assert engine._safe_model_path(p) == str(p)
