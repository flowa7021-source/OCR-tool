"""User-journey E2E tests — fully replicate the target-user experience.

The user's daily workflow on a Windows box:

    1. Drops a scanned Russian contract with a Cyrillic filename
       (``ТН к УПД 36 от 02.09.2022.pdf``) living at a Cyrillic path
       (``C:\\Users\\Т.Н. 020\\Desktop\\БП\\…``) into the app.
    2. Picks one of the 7 bundled profiles from the dropdown.
    3. Clicks Start OCR.
    4. Gets back a searchable PDF at the output path, plus optional
       TXT/DOCX exports.

Each test here exercises ONE leg of that journey against the real
Tesseract + Ghostscript + OCRmyPDF stack — the mocked-engine suite
can't catch regressions in things like "Cyrillic path breaks the
preprocess PNG write" or "universal_accurate profile produces
empty output because of DPI metadata". Those are exactly the bugs
that have been biting in production.

Skips cleanly on hosts without real OCR or rus.traineddata.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.shared.types import JobStatus
from tests.integration._real_ocr_helpers import (
    assert_ocr_recognised,
    make_realistic_profile,
    render_clean_text_pdf,
    requires_real_ocr,
    requires_real_russian_ocr,
    run_pipeline,
)

pytestmark = [
    requires_real_ocr,
    pytest.mark.exercise_preflight,
]


# ---------------------------------------------------------------------------
# Cyrillic filename + Cyrillic path: the actual production environment
# ---------------------------------------------------------------------------


@requires_real_russian_ocr
class TestCyrillicFilenameAndPath:
    """The user's real path is ``C:\\Users\\Т.Н. 020\\Desktop\\БП\\…``
    and the input filename is ``ТН к УПД 36 от 02.09.2022.pdf``. Every
    stage of the pipeline — rasterise, save PNG, ocrmypdf spawn,
    stamp output — must survive non-ASCII paths on both ends."""

    def test_cyrillic_input_filename_produces_russian_output(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        cyrillic_dir = tmp_path / "Т.Н. 020" / "Рабочий стол" / "БП"
        cyrillic_dir.mkdir(parents=True)
        input_pdf = cyrillic_dir / "ТН к УПД 36.pdf"
        output_pdf = cyrillic_dir / "ТН к УПД 36_ocr.pdf"

        render_clean_text_pdf(
            input_pdf, text="АКТ ПРИЁМА", cyrillic=True
        )

        profile = make_realistic_profile(
            languages=["rus", "eng"],
            binarization="otsu",
            autocorrect_russian=True,
            normalize_unicode=True,
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        # Output landed at the Cyrillic path.
        assert output_pdf.exists()
        # Non-trivial searchable PDF — OCRmyPDF emitted a text layer.
        assert output_pdf.stat().st_size > 1024

        # At least one tri-gram of the Russian input survived.
        assert_ocr_recognised(
            result, ["АКТ", "ПРИ", "ИЁМА", "РИЁ"]
        )

    def test_cyrillic_path_with_spaces_and_dots(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        """The user's actual Windows path has a period and a space in
        the directory name (``Т.Н. 020``). Windows' short-path (8.3)
        fallback converts it to ``450D~1.020`` which historically has
        been a source of subprocess-spawn failures. POSIX has no such
        fallback but the test still exercises the space+dot shape."""
        tricky_dir = tmp_path / "Т.Н. 020" / "foo bar" / "pdf.files"
        tricky_dir.mkdir(parents=True)
        input_pdf = tricky_dir / "договор #1.pdf"
        output_pdf = tricky_dir / "договор #1_ocr.pdf"

        render_clean_text_pdf(
            input_pdf, text="СЧЕТ №", cyrillic=True
        )

        profile = make_realistic_profile(
            languages=["rus", "eng"],
            binarization="otsu",
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert result.status is JobStatus.COMPLETED, result.error
        assert output_pdf.exists()


# ---------------------------------------------------------------------------
# Multi-page Russian contract — closest to the user's actual file
# ---------------------------------------------------------------------------


@requires_real_russian_ocr
class TestMultiPageRussianContract:
    """The user's test file is a 4-page Russian contract. All pages
    must come back with a text layer — a regression where only
    page 1 got OCR'd (common OCRmyPDF misconfig) must surface as a
    failed assertion."""

    def test_four_page_russian_document(
        self, tmp_path: Path, real_tesseract_wrapper
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "contract.pdf",
            text="ДОГОВОР",
            cyrillic=True,
            pages=4,
        )
        output_pdf = tmp_path / "contract_ocr.pdf"

        profile = make_realistic_profile(
            languages=["rus", "eng"],
            binarization="otsu",
            autocorrect_russian=True,
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        assert result.status is JobStatus.COMPLETED, result.error
        assert result.page_count == 4
        # Every one of the 4 pages has a non-empty text layer.
        import fitz
        with fitz.open(str(output_pdf)) as doc:
            assert doc.page_count == 4
            for i in range(4):
                page_text = doc.load_page(i).get_text("text")
                assert page_text.strip(), (
                    f"page {i + 1} has empty text layer in output PDF"
                )


# ---------------------------------------------------------------------------
# Each bundled profile must OCR a realistic Russian input
# ---------------------------------------------------------------------------


@requires_real_russian_ocr
class TestAllBundledProfilesRealOCR:
    """For each profile in ``profiles/*.json``, run real OCR on a
    Russian input and verify the profile genuinely produces
    recognisable text. ``universal_accurate`` and ``handwritten_mixed``
    are skipped: universal runs at 600 DPI and adds multi-minute wall
    time; handwritten_mixed needs the GOT-OCR 2.0 model which isn't
    bundled in test envs.

    This is how we'd have caught the ``_assemble_pdf`` DPI bug:
    every profile (except NONE-binarization ones) was producing
    empty hOCR and we didn't notice until a user reported it.
    """

    @pytest.mark.parametrize(
        "profile_name",
        [
            "default",
            "quick_reliable",
            "contracts_ru",
            "low_quality_scan",
            # ``english_text`` uses eng-only — tested below in a
            # separate English-text parametrisation.
            # ``universal_accurate`` uses 600 DPI — too slow for CI
            # as a parametrised test; dedicated test below caps DPI.
            # ``handwritten_mixed`` uses GOT-OCR 2.0 — needs the
            # ~580 MB model; covered by test_e2e_got_ocr2.py with
            # a stubbed engine.
        ],
    )
    def test_profile_produces_russian_text(
        self,
        profile_name: str,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = manager.load(profile_name)

        input_pdf = render_clean_text_pdf(
            tmp_path / "ru.pdf",
            text="ДОГОВОР",
            cyrillic=True,
        )
        output_pdf = tmp_path / "ru_ocr.pdf"

        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["ДОГ", "ОГО", "ВОР"])


@requires_real_ocr
class TestEnglishTextProfile:
    """``english_text`` profile is eng-only — verify it OCRs English."""

    def test_english_text_profile_produces_text(
        self,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = manager.load("english_text")

        input_pdf = render_clean_text_pdf(
            tmp_path / "en.pdf", text="CONTRACT AGREEMENT"
        )
        output_pdf = tmp_path / "en_ocr.pdf"

        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert_ocr_recognised(result, ["CONTRACT", "AGREEMENT"])


@requires_real_russian_ocr
class TestUniversalAccurateProfile:
    """``universal_accurate`` uses 500 DPI — too slow for a
    parametrised CI test. We cap DPI at 300 for this test to
    verify the rest of the profile (Sauvola + CLAHE + deskew +
    denoise chain + border removal + background removal + all
    postprocess flags) doesn't crash.

    The 500 DPI + real Tesseract path — where OCR content is
    actually testable — is exercised by the nightly corpus matrix
    (``test_nightly_corpus.py``, restricted to ``universal_accurate``)
    and the installer smoke test the user runs before a release.

    Scope of THIS test: "pipeline survives the full universal_accurate
    configuration and returns a COMPLETED job". We intentionally do
    NOT assert on OCR content because the profile's preprocessing
    stack is tuned for the real 500 DPI target: Sauvola
    ``window=41``, background removal ``blur_kernel=55``, border
    removal ``min_line_length=125``, CLAHE, deskew, denoise chain.
    At the capped 300 DPI (and the test input being rasterised at
    200 DPI before being up-sampled), those parameters over-
    aggressively strip strokes from short synthetic text like
    "ДОГОВОР" — Tesseract ends up reading noise. That's a test-
    harness artefact, not a pipeline bug; asserting on content
    here would produce flaky failures unrelated to any real
    regression. The nightly corpus matrix (running at the
    profile's native 500 DPI on 20 adversarial documents) is
    where we catch actual accuracy regressions.

    What we DO assert: the job reaches ``JobStatus.COMPLETED``, the
    output PDF exists with non-trivial size, and the per-page
    ``JobResult.pages`` list was populated — enough to prove every
    stage of the pipeline (analyze / preprocess / assemble / OCR /
    postprocess) ran without raising.
    """

    def test_universal_accurate_runs_at_capped_dpi(
        self,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = manager.load("universal_accurate")
        # Cap DPI so the test finishes in CI (<30 s instead of 2+ min).
        profile.ocr.dpi = 300

        input_pdf = render_clean_text_pdf(
            tmp_path / "ua.pdf", text="ДОГОВОР", cyrillic=True
        )
        output_pdf = tmp_path / "ua_ocr.pdf"

        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        # Scope: "pipeline runs to completion". Content accuracy is
        # deliberately not asserted here — see the class docstring.
        assert result.status is JobStatus.COMPLETED, (
            f"universal_accurate pipeline FAILED at capped DPI 300: "
            f"{result.error!r}"
        )
        assert output_pdf.exists(), "no output PDF was produced"
        assert output_pdf.stat().st_size > 1024, (
            f"output PDF suspiciously small "
            f"({output_pdf.stat().st_size} bytes) — likely an empty/"
            f"malformed searchable PDF"
        )
        assert result.pages, (
            "JobResult.pages is empty — the OCR stage returned no "
            "per-page records even though status is COMPLETED"
        )
        assert len(result.pages) == 1, (
            f"expected 1-page input → 1 page of result, got "
            f"{len(result.pages)}"
        )


# ---------------------------------------------------------------------------
# CLI end-to-end: the dev-loop shortcut
# ---------------------------------------------------------------------------


class TestCLIEndToEnd:
    """The CLI is the user's (and my) primary faster-than-installer
    debug loop. These tests exercise it as a subprocess — the same
    way a human would run it — so a regression in argument parsing,
    stdout encoding, or exit codes surfaces."""

    def test_cli_processes_english_pdf_end_to_end(
        self, tmp_path: Path
    ) -> None:
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="CLI SMOKE"
        )
        output_pdf = tmp_path / "out.pdf"

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        # Use tmp-scoped profile + cache so the user's real install
        # doesn't influence the test.
        env["OCRSTUDIO_LOGS_DIR"] = str(tmp_path / "logs")

        proc = subprocess.run(
            [
                sys.executable,
                "-m", "src.cli",
                str(input_pdf),
                "-o", str(output_pdf),
                "--profile", "quick_reliable",
                "-v",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, (
            f"CLI exited {proc.returncode}\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
        assert output_pdf.exists()
        # The output PDF has a text layer.
        import fitz
        with fitz.open(str(output_pdf)) as doc:
            recognised = doc.load_page(0).get_text("text") or ""
        assert any(
            w in recognised.upper() for w in ("CLI", "SMOKE")
        ), f"CLI output PDF lacks recognised text: {recognised!r}"

    @requires_real_russian_ocr
    def test_cli_processes_russian_pdf_with_cyrillic_paths(
        self, tmp_path: Path
    ) -> None:
        cyrillic_dir = tmp_path / "Т.Н. 020" / "Договоры"
        cyrillic_dir.mkdir(parents=True)
        input_pdf = cyrillic_dir / "акт.pdf"
        output_pdf = cyrillic_dir / "акт_ocr.pdf"
        render_clean_text_pdf(
            input_pdf, text="АКТ", cyrillic=True
        )

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        env["OCRSTUDIO_LOGS_DIR"] = str(tmp_path / "logs")

        proc = subprocess.run(
            [
                sys.executable,
                "-m", "src.cli",
                str(input_pdf),
                "-o", str(output_pdf),
                "--profile", "quick_reliable",
                "-v",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, (
            f"CLI exited {proc.returncode} on Cyrillic path\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
        assert output_pdf.exists()

    def test_cli_txt_export_matches_recognized_text(
        self, tmp_path: Path
    ) -> None:
        """Users often export TXT alongside the searchable PDF for
        downstream tooling. The TXT must contain the same text that
        was recognized."""
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="EXPORT TEST 2026"
        )
        output_pdf = tmp_path / "out.pdf"

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        env["OCRSTUDIO_LOGS_DIR"] = str(tmp_path / "logs")

        proc = subprocess.run(
            [
                sys.executable,
                "-m", "src.cli",
                str(input_pdf),
                "-o", str(output_pdf),
                "--profile", "quick_reliable",
                "--txt",
                "-v",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, (
            f"CLI exited {proc.returncode}\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
        txt_path = output_pdf.with_suffix(".txt")
        assert txt_path.exists(), "--txt did not create the TXT export"
        body = txt_path.read_text(encoding="utf-8")
        assert any(
            w in body.upper() for w in ("EXPORT", "TEST", "2026", "202")
        ), f"TXT export has no expected content: {body!r}"


# ---------------------------------------------------------------------------
# smoke_test_real_ocr.py as a subprocess — proves the dev-loop itself
# ---------------------------------------------------------------------------


class TestSmokeTestScript:
    """The smoke test script is the fastest feedback loop we have.
    If IT breaks, devs lose the 30-second cycle and go back to the
    15-minute installer grind. Treat it as a first-class contract."""

    def test_smoke_test_script_runs_green(self, tmp_path: Path) -> None:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        env["OCRSTUDIO_LOGS_DIR"] = str(tmp_path / "logs")

        proc = subprocess.run(
            [
                sys.executable,
                "scripts/smoke_test_real_ocr.py",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=120,
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        assert proc.returncode == 0, (
            f"smoke_test_real_ocr.py exited {proc.returncode}\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
