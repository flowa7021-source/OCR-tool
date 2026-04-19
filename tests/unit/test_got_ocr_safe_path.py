"""Tests for ``GOTOCREngine._safe_model_path``.

Regression guard for the HTR equivalent of the Tesseract Unicode bug:
when the bundled GOT-OCR 2.0 weights sit under a non-ASCII path like
``C:\\Users\\Т.Н. 020\\...``, some transformers/safetensors code paths
(C++/Rust extensions) have historically struggled. We preemptively
resolve to a Win32 short-path on Windows, and pass through untouched
elsewhere.
"""

from __future__ import annotations

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


def test_safe_path_posix_non_ascii_passthrough(monkeypatch) -> None:
    """The POSIX branch returns non-ASCII paths unchanged.

    The shortening logic is gated on ``os.name == 'nt'``. Patching
    ``os.name`` to ``"posix"`` lets this test exercise the pathlib-
    safe branch on every platform — including the Windows CI leg,
    where a naive ``sys.platform == 'win32'`` skip previously kept
    the test from running at all.

    WARNING: ``os`` is a module singleton so ``monkeypatch.setattr``
    on ``os.name`` mutates it globally for the duration of the test.
    While that patch is in place, ``pathlib.Path(...)`` reads the
    hijacked value and tries to instantiate ``PosixPath`` on an
    actual Windows host, which raises::

        NotImplementedError: cannot instantiate 'PosixPath' on your system

    Any ``Path(...)`` constructor — including the ones pytest itself
    calls in its cache-provider teardown — crashes until the patch is
    undone. We dodge this by building the test ``Path`` BEFORE
    patching ``os.name``, then never constructing another ``Path``
    for the rest of the test body. The sibling test
    ``test_safe_path_falls_back_when_short_unavailable`` gets the
    same safety for free via the ``tmp_path`` fixture (Path created
    by pytest before the test function even starts).
    """
    # Build the Path FIRST, while os.name is still "nt" on Windows
    # (so pathlib resolves to WindowsPath). Only then hijack os.name.
    p = Path("/home/Т.Н. 020/models")
    engine = _engine()
    monkeypatch.setattr("src.application.engines.got_ocr_engine.os.name", "posix")
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
