"""E2E tests on REALISTIC synthetic PDFs that model real scan problems.

Previous tests used clean, crisp synthetic text — perfect line-art on
a white background. Real user scans have:
  * Faded photocopies (low contrast)
  * Crooked scans (skew) + speckle noise
  * Multi-page documents with mixed content per page (headers, body
    text, tables with rules, signatures, fine print, numbers)

These fixtures are generated at test time (not committed to git —
they're ~60 MB) by ``tests/fixtures/generate_fixtures.py``. Each
test runs a real pipeline with real Tesseract on a fixture that
would have caught the bugs we hit in production.

Skips cleanly when Tesseract isn't on PATH.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.types import JobStatus
from tests.integration._real_ocr_helpers import (
    make_realistic_profile,
    requires_real_ocr,
    requires_real_russian_ocr,
    run_pipeline,
)

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


# ---------------------------------------------------------------------------
# Fixtures: generate PDFs once per session (slow to render, reuse across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> Path:
    """Generate all fixture PDFs once per test session."""
    import sys

    repo = Path(__file__).resolve().parent.parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from tests.fixtures.generate_fixtures import (
        FIXTURES_DIR,
        generate_contract_ru_4page,
        generate_faded_scan,
        generate_skewed_noisy,
    )

    # Generate into the repo's fixtures dir (gitignored).
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    if not (FIXTURES_DIR / "contract_ru_4page.pdf").exists():
        generate_contract_ru_4page()
    if not (FIXTURES_DIR / "faded_scan.pdf").exists():
        generate_faded_scan()
    if not (FIXTURES_DIR / "skewed_noisy.pdf").exists():
        generate_skewed_noisy()
    return FIXTURES_DIR


# ---------------------------------------------------------------------------
# Multi-page contract — the user's exact scenario
# ---------------------------------------------------------------------------


@requires_real_russian_ocr
class TestContractRu4Page:
    """A 4-page Russian contract with headers, body, table, signatures.

    This is the closest model we have to the user's actual
    ``ТН к УПД 36 от 02.09.2022.pdf`` without using their real data.
    Every profile that claims to handle Russian contracts must produce
    non-empty text on ALL 4 pages.
    """

    @pytest.mark.parametrize(
        "profile_name",
        ["default", "universal_accurate", "universal_accurate", "universal_accurate"],
    )
    def test_every_page_has_text(
        self,
        profile_name: str,
        fixtures_dir: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = manager.load(profile_name)

        input_pdf = fixtures_dir / "contract_ru_4page.pdf"
        output_pdf = tmp_path / f"contract_{profile_name}_ocr.pdf"

        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        assert result.status is JobStatus.COMPLETED, (
            f"profile={profile_name} FAILED: {result.error}"
        )
        assert result.page_count == 4, (
            f"expected 4 pages, got {result.page_count}"
        )
        # Every page must have SOME recognised text — a page with
        # zero text means the preprocessing destroyed the content
        # or the DPI/timeout was misconfigured.
        for i, page in enumerate(result.pages):
            assert (page.text or "").strip(), (
                f"profile={profile_name} page {i + 1} has empty text"
            )


# ---------------------------------------------------------------------------
# Faded scan — CLAHE is mandatory for recognition
# ---------------------------------------------------------------------------


class TestFadedScan:
    """Low-contrast page (foreground 140, background 210). Without
    CLAHE preprocessing, OTSU threshold collapses everything to
    white and Tesseract returns nothing. Profiles with CLAHE on
    must produce recognisable text."""

    def test_clahe_profile_reads_faded_text(
        self,
        fixtures_dir: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        input_pdf = fixtures_dir / "faded_scan.pdf"
        output_pdf = tmp_path / "faded_ocr.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            clahe=True,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert result.status is JobStatus.COMPLETED, result.error
        text = (result.pages[0].text or "").upper()
        assert any(
            w in text for w in ("FADED", "INVOICE", "1234", "2026")
        ), f"CLAHE profile couldn't read faded scan: {text!r}"


# ---------------------------------------------------------------------------
# Skewed + noisy — deskew + denoise required
# ---------------------------------------------------------------------------


class TestSkewedNoisy:
    """3° rotation + 2% salt-pepper noise. Deskew + denoise must
    recover enough text for Tesseract to read."""

    def test_deskew_denoise_reads_skewed_noisy_text(
        self,
        fixtures_dir: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        input_pdf = fixtures_dir / "skewed_noisy.pdf"
        output_pdf = tmp_path / "skewed_ocr.pdf"

        profile = make_realistic_profile(
            binarization="otsu",
            deskew=True,
            denoise_steps=[("median", {"ksize": 3})],
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert result.status is JobStatus.COMPLETED, result.error
        text = (result.pages[0].text or "").upper()
        assert any(
            w in text for w in ("SKEWED", "NOISY", "DOCUMENT", "2026")
        ), f"deskew+denoise couldn't read skewed/noisy scan: {text!r}"


# ---------------------------------------------------------------------------
# All profiles on all fixtures — the matrix
# ---------------------------------------------------------------------------


class TestAllProfilesAllFixtures:
    """Every Tesseract profile × every fixture → COMPLETED.

    We don't assert specific text here (that's done in the targeted
    tests above). We assert that NO combination crashes the pipeline.
    A crash = a code bug, not an accuracy issue.
    """

    @pytest.mark.parametrize("fixture_name", [
        "faded_scan.pdf",
        "skewed_noisy.pdf",
    ])
    @pytest.mark.parametrize("profile_name", [
        "universal_accurate",
        "default",
        "universal_accurate",
        "universal_accurate",
        "universal_accurate",
    ])
    def test_no_crash(
        self,
        profile_name: str,
        fixture_name: str,
        fixtures_dir: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = manager.load(profile_name)

        input_pdf = fixtures_dir / fixture_name
        output_pdf = tmp_path / f"{profile_name}_{fixture_name}"

        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )
        assert result.status is JobStatus.COMPLETED, (
            f"{profile_name} × {fixture_name} FAILED: {result.error}"
        )
        assert output_pdf.exists()
