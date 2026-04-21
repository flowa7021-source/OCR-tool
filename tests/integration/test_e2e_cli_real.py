"""Real-OCR via CLI: subprocess ``python -m src.cli input.pdf``.

The CLI is a separate entry point from the GUI and uses its own
config bootstrapping. A CLI-only regression is invisible to GUI
tests, so we run the CLI as a real subprocess with a real PDF.

This catches:
  * Missing ``__main__`` entry / incorrect import path
  * CLI doesn't initialise built-in profiles before use
  * --output path with Cyrillic / spaces breaks when passed via argv
  * Exit code isn't 0 on success (breaks CI scripts users write)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration._real_ocr_helpers import (
    render_clean_text_pdf,
    requires_real_ocr,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


def _run_cli(*args: str, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    """Invoke ``python -m src.cli ...`` with the repo root on PYTHONPATH."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    env = {"PATH": __import__("os").environ.get("PATH", "")}
    # Pass through anything that real-OCR wrappers rely on (e.g.
    # TESSDATA_PREFIX on Windows).
    import os

    for key in ("TESSDATA_PREFIX", "SYSTEMROOT", "HOME", "USERPROFILE"):
        if key in os.environ:
            env[key] = os.environ[key]
    env["PYTHONPATH"] = str(repo_root)
    # PYTHONIOENCODING forces the child Python's stdout/stderr to UTF-8,
    # which matters on Windows where the default is cp1252. Russian text
    # in error messages (e.g. "Отсутствуют языки: rus") would otherwise
    # crash subprocess._readerthread with UnicodeDecodeError when we
    # try to capture it, killing the test before the real assertion runs.
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


class TestCliSingleFile:
    """``python -m src.cli in.pdf -o out.pdf`` exits 0 and writes output."""

    def test_single_pdf_end_to_end(
        self, tmp_path: Path, real_tesseract_wrapper,
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", "cli test", pages=1,
        )
        output_pdf = tmp_path / "out.pdf"

        result = _run_cli(
            str(input_pdf),
            "-o", str(output_pdf),
            "--profile", "universal_accurate",
            cwd=tmp_path,
        )

        assert result.returncode == 0, (
            f"CLI exited {result.returncode}.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert output_pdf.exists(), "CLI claimed success but no output PDF"


class TestCliListProfiles:
    """``--list-profiles`` exits 0 and names the built-in profiles."""

    def test_list_profiles(self, tmp_path: Path) -> None:
        result = _run_cli("--list-profiles", cwd=tmp_path)
        assert result.returncode == 0, (
            f"--list-profiles failed: {result.stderr}"
        )
        # Output should include at least a couple of built-ins
        combined = result.stdout + result.stderr
        assert "universal_accurate" in combined, (
            f"--list-profiles did not show universal_accurate:\n{combined}"
        )


class TestCliCyrillicOutputPath:
    """CLI must accept Cyrillic output paths without encoding errors.

    On Windows the default stdio encoding is CP-1251 and argv arrives
    decoded via Windows wide-char APIs; ``_force_utf8_stdio`` in the
    CLI reconfigures stdio to UTF-8 so subprocess capture works. If
    that setup regresses, this test catches it on the Linux CI leg
    too (LC_ALL defaults to C on runners, so encoding matters there).
    """

    def test_cyrillic_output_path(
        self, tmp_path: Path, real_tesseract_wrapper,
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", "test", pages=1,
        )
        cyr_dir = tmp_path / "Мои документы"
        cyr_dir.mkdir()
        output_pdf = cyr_dir / "результат.pdf"

        result = _run_cli(
            str(input_pdf),
            "-o", str(output_pdf),
            "--profile", "universal_accurate",
            cwd=tmp_path,
        )
        assert result.returncode == 0, (
            f"CLI failed on Cyrillic output path: {result.stderr}"
        )
        assert output_pdf.exists()
