"""Tests for :mod:`src.infrastructure.installed_app`.

The helper walks standard Windows install locations looking for an
already-installed OCR Studio, so a dev working in a source checkout
can reuse its bundled Tesseract / Ghostscript without a separate
system install. POSIX always returns None; the tests verify both
branches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.infrastructure import installed_app


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Clear the ``lru_cache`` between tests so each test runs its own
    monkeypatched environment against a fresh probe."""
    installed_app.find_installed_ocr_studio_resources.cache_clear()
    yield
    installed_app.find_installed_ocr_studio_resources.cache_clear()


def test_posix_returns_none(monkeypatch) -> None:
    """Non-Windows platforms unconditionally return None — the helper
    is only meaningful for the Windows-installer layout."""
    monkeypatch.setattr(sys, "platform", "linux")
    assert installed_app.find_installed_ocr_studio_resources() is None


def test_returns_none_when_no_install_found(
    monkeypatch, tmp_path: Path
) -> None:
    """If none of the well-known install roots contain an
    ``_internal/resources`` directory, return None cleanly — no
    exception even when LOCALAPPDATA is missing entirely."""
    monkeypatch.setattr(sys, "platform", "win32")
    # Point env vars at an empty tmp dir so every candidate misses.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(empty))
    monkeypatch.setenv("PROGRAMFILES", str(empty))
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)

    assert installed_app.find_installed_ocr_studio_resources() is None


def test_finds_localappdata_install(monkeypatch, tmp_path: Path) -> None:
    """Per-user install layout (LOCALAPPDATA\\Programs\\OCR Studio\\_internal)
    is the default for Inno Setup without admin elevation."""
    monkeypatch.setattr(sys, "platform", "win32")
    appdata = tmp_path / "LocalAppData"
    resources = (
        appdata
        / "Programs"
        / "OCR Studio"
        / "_internal"
        / "resources"
    )
    resources.mkdir(parents=True)

    monkeypatch.setenv("LOCALAPPDATA", str(appdata))
    # Drop PROGRAMFILES so the search clearly lands on LOCALAPPDATA.
    monkeypatch.delenv("PROGRAMFILES", raising=False)
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)

    found = installed_app.find_installed_ocr_studio_resources()
    assert found == resources


def test_finds_programfiles_install_when_localappdata_missing(
    monkeypatch, tmp_path: Path
) -> None:
    """Per-machine install (Program Files\\OCR Studio) found when the
    per-user location is empty."""
    monkeypatch.setattr(sys, "platform", "win32")
    pf = tmp_path / "Program Files"
    resources = pf / "OCR Studio" / "_internal" / "resources"
    resources.mkdir(parents=True)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(empty))
    monkeypatch.setenv("PROGRAMFILES", str(pf))

    found = installed_app.find_installed_ocr_studio_resources()
    assert found == resources


def test_localappdata_wins_over_programfiles(
    monkeypatch, tmp_path: Path
) -> None:
    """When both per-user and per-machine installs are present,
    prefer per-user — that's the install the dev most likely uses
    day-to-day (no admin needed to update it)."""
    monkeypatch.setattr(sys, "platform", "win32")

    appdata = tmp_path / "LocalAppData"
    user_resources = (
        appdata / "Programs" / "OCR Studio" / "_internal" / "resources"
    )
    user_resources.mkdir(parents=True)

    pf = tmp_path / "Program Files"
    machine_resources = pf / "OCR Studio" / "_internal" / "resources"
    machine_resources.mkdir(parents=True)

    monkeypatch.setenv("LOCALAPPDATA", str(appdata))
    monkeypatch.setenv("PROGRAMFILES", str(pf))

    found = installed_app.find_installed_ocr_studio_resources()
    assert found == user_resources


def test_result_is_lru_cached(monkeypatch, tmp_path: Path) -> None:
    """The probe is LRU-cached so repeated UI queries don't stat the
    filesystem repeatedly."""
    monkeypatch.setattr(sys, "platform", "win32")
    appdata = tmp_path / "LocalAppData"
    resources = (
        appdata / "Programs" / "OCR Studio" / "_internal" / "resources"
    )
    resources.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(appdata))

    first = installed_app.find_installed_ocr_studio_resources()
    assert first == resources

    # If the resources dir is deleted after the first call, the cached
    # value still comes back — the second call doesn't re-stat.
    import shutil
    shutil.rmtree(resources)
    assert installed_app.find_installed_ocr_studio_resources() == first
