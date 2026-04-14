"""Tests for src.shared.constants path resolution under PyInstaller."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _reload_constants_with_frozen(
    meipass: Path | None,
    executable: Path,
    monkeypatch,
):
    """Reload shared.constants with patched PyInstaller attributes."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable), raising=False)
    if meipass is not None:
        monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    else:
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    # Force re-import so module-level constants are re-evaluated
    import src.shared.constants as c  # noqa: PLC0415

    return importlib.reload(c)


def test_source_checkout_returns_repo_root(monkeypatch) -> None:
    """When not frozen, APP_ROOT is the project root."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    import src.shared.constants as c

    reloaded = importlib.reload(c)
    # The project root must contain pyproject.toml (sanity marker)
    assert (reloaded.APP_ROOT / "pyproject.toml").exists()


def test_frozen_pyinstaller_6_uses_meipass(tmp_path: Path, monkeypatch) -> None:
    """PyInstaller 6 --onedir: APP_ROOT == sys._MEIPASS, not executable dir."""
    internal = tmp_path / "OCRStudio" / "_internal"
    internal.mkdir(parents=True)
    (internal / "resources").mkdir()
    (internal / "resources" / "tesseract").mkdir()
    (internal / "resources" / "tesseract" / "tesseract.exe").write_text("stub")

    executable = tmp_path / "OCRStudio" / "OCRStudio.exe"
    executable.write_text("stub")

    reloaded = _reload_constants_with_frozen(internal, executable, monkeypatch)

    assert internal == reloaded.APP_ROOT
    # Tesseract must be findable at the derived path
    assert (reloaded.TESSERACT_BIN_DIR / "tesseract.exe").exists()


def test_frozen_no_meipass_falls_back_to_executable_dir(
    tmp_path: Path, monkeypatch
) -> None:
    """Old PyInstaller --onedir (<6.0): no _MEIPASS, fall back to exe dir."""
    app_dir = tmp_path / "OCRStudio"
    app_dir.mkdir()
    (app_dir / "resources").mkdir()
    (app_dir / "resources" / "tesseract").mkdir()

    executable = app_dir / "OCRStudio.exe"
    executable.write_text("stub")

    reloaded = _reload_constants_with_frozen(None, executable, monkeypatch)

    assert app_dir == reloaded.APP_ROOT


def test_unfreeze_between_tests(monkeypatch) -> None:
    """Safety: make sure later tests see a non-frozen environment."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    import src.shared.constants as c

    reloaded = importlib.reload(c)
    # sys.frozen should be False again
    assert not getattr(sys, "frozen", False)
    assert reloaded.APP_ROOT.name  # still resolvable
