"""Tests for ``src.infrastructure.external_tools``.

These tests exercise the discovery + PATH registration logic that
ensures OCRmyPDF's ``shutil.which`` calls hit our bundled Tesseract /
Ghostscript binaries rather than failing with ``MissingDependencyError``.

Every test in this module is marked ``exercise_preflight`` so the
autouse bypass in ``tests/conftest.py`` does NOT stub out the real
``verify_required_for_ocrmypdf`` function — these tests want to call
the real implementation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.exercise_preflight


def _make_fake_binary(dir_path: Path, name: str) -> Path:
    """Create a file that ``locate`` will accept as a binary.

    ``is_file()`` is all we check; no need for a real executable bit
    since the tests never actually run the file.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    target = dir_path / name
    target.write_bytes(b"")
    return target


def test_locate_finds_binary_in_bundle(tmp_path: Path, monkeypatch) -> None:
    """Direct hit — binary sits straight in the bundle_dir."""
    from src.infrastructure import external_tools as ext

    name = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    _make_fake_binary(tmp_path, name)

    tool = ext.ExternalTool(
        name="tesseract",
        bundle_dir=tmp_path,
        candidate_names=(name,),
        required_for_ocrmypdf=True,
    )
    found = ext.locate(tool)
    assert found is not None
    assert found.name == name
    assert found.parent == tmp_path


def test_locate_finds_binary_in_nested_subdir(tmp_path: Path) -> None:
    """Some installers drop under ``bin/``; we search one level deep."""
    from src.infrastructure import external_tools as ext

    nested = tmp_path / "bin"
    _make_fake_binary(nested, "gswin64c.exe")

    tool = ext.ExternalTool(
        name="ghostscript",
        bundle_dir=tmp_path,
        candidate_names=("gswin64c.exe",),
        required_for_ocrmypdf=True,
    )
    found = ext.locate(tool)
    assert found is not None
    assert found.parent == nested


def test_locate_returns_none_when_absent(tmp_path: Path, monkeypatch) -> None:
    """Nothing in the bundle and nothing on the system PATH."""
    from src.infrastructure import external_tools as ext

    tool = ext.ExternalTool(
        name="made-up-tool",
        bundle_dir=tmp_path / "nonexistent",
        candidate_names=("definitely-not-a-real-binary-xyz123",),
        required_for_ocrmypdf=False,
    )
    # Don't inherit the host PATH so shutil.which can't find anything.
    monkeypatch.setenv("PATH", "")
    assert ext.locate(tool) is None


def test_ensure_on_path_prepends_bundle_dirs(
    tmp_path: Path, monkeypatch
) -> None:
    """Discovered bundle dirs must land at the FRONT of PATH."""
    from src.infrastructure import external_tools as ext

    tess_dir = tmp_path / "tesseract"
    gs_dir = tmp_path / "ghostscript"

    tess_name = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    gs_name = "gswin64c.exe" if sys.platform == "win32" else "gs"
    _make_fake_binary(tess_dir, tess_name)
    _make_fake_binary(gs_dir, gs_name)

    fake_registry = (
        ext.ExternalTool(
            name="tesseract",
            bundle_dir=tess_dir,
            candidate_names=(tess_name,),
            required_for_ocrmypdf=True,
        ),
        ext.ExternalTool(
            name="ghostscript",
            bundle_dir=gs_dir,
            candidate_names=(gs_name,),
            required_for_ocrmypdf=True,
        ),
    )
    monkeypatch.setattr(ext, "REGISTRY", fake_registry)

    monkeypatch.setenv("PATH", f"/pre/existing{os.pathsep}/also/preexisting")
    resolved = ext.ensure_on_path()

    assert resolved["tesseract"] is not None
    assert resolved["ghostscript"] is not None

    parts = os.environ["PATH"].split(os.pathsep)
    # Both bundle dirs must appear, and BEFORE the preexisting ones.
    assert str(tess_dir) in parts
    assert str(gs_dir) in parts
    assert parts.index(str(tess_dir)) < parts.index("/pre/existing")
    assert parts.index(str(gs_dir)) < parts.index("/pre/existing")


def test_ensure_on_path_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Re-running must not stack duplicate entries on PATH."""
    from src.infrastructure import external_tools as ext

    tess_dir = tmp_path / "tesseract"
    tess_name = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    _make_fake_binary(tess_dir, tess_name)

    fake_registry = (
        ext.ExternalTool(
            name="tesseract",
            bundle_dir=tess_dir,
            candidate_names=(tess_name,),
            required_for_ocrmypdf=True,
        ),
    )
    monkeypatch.setattr(ext, "REGISTRY", fake_registry)

    monkeypatch.setenv("PATH", "")
    ext.ensure_on_path()
    first_path = os.environ["PATH"]
    ext.ensure_on_path()
    second_path = os.environ["PATH"]
    assert first_path == second_path, (
        f"PATH changed on second call — not idempotent.\n"
        f"first={first_path!r}\n"
        f"second={second_path!r}"
    )
    assert second_path.split(os.pathsep).count(str(tess_dir)) == 1


def test_verify_required_for_ocrmypdf_reports_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """A missing required tool is returned by name; optional missing is not."""
    from src.infrastructure import external_tools as ext

    present = tmp_path / "have-this"
    _make_fake_binary(present, "present-tool.exe")

    fake_registry = (
        ext.ExternalTool(
            name="present-tool",
            bundle_dir=present,
            candidate_names=("present-tool.exe",),
            required_for_ocrmypdf=True,
        ),
        ext.ExternalTool(
            name="missing-required",
            bundle_dir=tmp_path / "nowhere",
            candidate_names=("nope.exe",),
            required_for_ocrmypdf=True,
        ),
        ext.ExternalTool(
            name="missing-optional",
            bundle_dir=tmp_path / "also-nowhere",
            candidate_names=("also-nope.exe",),
            required_for_ocrmypdf=False,
        ),
    )
    monkeypatch.setattr(ext, "REGISTRY", fake_registry)
    monkeypatch.setenv("PATH", "")  # no system fallback

    missing = ext.verify_required_for_ocrmypdf()
    assert missing == ["missing-required"], (
        f"Expected ['missing-required'] (required + absent), got {missing!r}"
    )


def test_configure_pytesseract_prepends_bundle_dir_to_path(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression for end-user log:
    ``MissingDependencyError: Could not find program 'tesseract' on
    the PATH`` from OCRmyPDF. Root cause: we configured
    ``pytesseract.tesseract_cmd`` but OCRmyPDF uses
    ``shutil.which`` which walks PATH, not pytesseract's setting.
    ``configure_pytesseract`` must prepend the bundled bin dir to
    os.environ['PATH'] so OCRmyPDF picks up our binary.
    """
    pytest.importorskip("pytesseract")
    from src.infrastructure.tesseract_wrapper import TesseractWrapper

    bin_dir = tmp_path / "tesseract_bundle"
    tess_name = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    bin_path = _make_fake_binary(bin_dir, tess_name)
    tessdata_dir = tmp_path / "tessdata"
    tessdata_dir.mkdir()
    (tessdata_dir / "rus.traineddata").write_bytes(b"fake")

    TesseractWrapper.reset()
    TesseractWrapper._binary_path = bin_path
    TesseractWrapper._tessdata_path = tessdata_dir
    monkeypatch.setenv("PATH", "/unrelated")
    # ``configure_pytesseract`` writes ``os.environ["TESSDATA_PREFIX"]``
    # to the chosen tessdata dir. If we don't undo that mutation,
    # subsequent subprocess-worker tests (ParallelProcessor, fork mode)
    # inherit the stale tmp path and ``find_tessdata_dir`` picks it up
    # via the env-var fallback — ``verify()`` then reports
    # "Отсутствуют языки: eng" because the tmp dir only has
    # ``rus.traineddata``. Save + restore manually below.
    original_tessdata_prefix = os.environ.pop("TESSDATA_PREFIX", None)

    try:
        TesseractWrapper().configure_pytesseract()
    finally:
        TesseractWrapper.reset()
        if original_tessdata_prefix is not None:
            os.environ["TESSDATA_PREFIX"] = original_tessdata_prefix
        else:
            os.environ.pop("TESSDATA_PREFIX", None)

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert str(bin_dir) in path_entries, (
        f"configure_pytesseract did not prepend {bin_dir} to PATH. "
        f"PATH={os.environ['PATH']!r}"
    )
    # Must be in FRONT of /unrelated, not appended.
    assert path_entries.index(str(bin_dir)) < path_entries.index("/unrelated")
