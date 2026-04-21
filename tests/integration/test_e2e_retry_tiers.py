"""Real-OCR tests for the per-page retry-tier system.

Unlike ``tests/unit/test_engines.py`` which mocks ``run_ocrmypdf`` to
verify the retry logic is *wired*, this module runs the **full real
stack** (Tesseract + Ghostscript + OCRmyPDF) on adversarial fixtures
that historically crash Tesseract's layout analyser.

The invariant every test enforces:

    Every page of the input PDF ends up with SOME recognised text in
    the output PDF — via the primary attempt, the simplified-settings
    retry, or the last-resort sparse-text tier. Zero raster-only pages.

This matches the user requirement: "мне нужно чтобы все страницы
всегда были распознаны и с текстовым слоем".

Fixtures exercised:

    * invoice_typical_form.pdf  — the user's failure shape
    * dense_small_text.pdf      — 7pt Russian body stress test
    * stamped_page.pdf          — circular stamp overlay (crashes AUTO)
    * rotated_table.pdf         — 8° skew past auto-deskew threshold
    * contract_ru_4page.pdf     — the 4-page contract case

Skips cleanly when Tesseract isn't on PATH or rus.traineddata is
missing; CI has both pre-installed on both the Linux and Windows legs.
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
# Fixtures: generate adversarial PDFs once per session (slow; reuse across tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def adversarial_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate all adversarial fixture PDFs once per test session.

    The baseline fixtures (contract, faded, skewed-noisy) and the new
    adversarial set (invoice form, dense small text, stamped page,
    rotated table) are both generated into ``tests/fixtures/``. The
    generator short-circuits when a fixture file already exists, so
    this is a no-op on the second test run.
    """
    import sys

    repo = Path(__file__).resolve().parent.parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from tests.fixtures.generate_fixtures import (
        FIXTURES_DIR,
        generate_contract_ru_4page,
        generate_dense_small_text,
        generate_invoice_typical_form,
        generate_rotated_table,
        generate_stamped_page,
    )

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    _ensure(FIXTURES_DIR / "contract_ru_4page.pdf", generate_contract_ru_4page)
    _ensure(FIXTURES_DIR / "invoice_typical_form.pdf",
            generate_invoice_typical_form)
    _ensure(FIXTURES_DIR / "dense_small_text.pdf", generate_dense_small_text)
    _ensure(FIXTURES_DIR / "stamped_page.pdf", generate_stamped_page)
    _ensure(FIXTURES_DIR / "rotated_table.pdf", generate_rotated_table)
    return FIXTURES_DIR


def _ensure(path: Path, generator) -> None:
    """Run ``generator`` only when ``path`` doesn't already exist."""
    if not path.exists():
        generator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_pdf_has_text_on_every_page(pdf_path: Path) -> tuple[bool, list[int]]:
    """Return ``(all_pages_have_text, empty_page_numbers)``.

    We walk the output PDF page-by-page with PyMuPDF and check each
    page for any non-whitespace extractable text. A page with zero
    recognised characters is what the retry-tier system must prevent;
    if the assertion fails, the list of empty page numbers tells us
    which pages slipped through.
    """
    import fitz

    empty_pages: list[int] = []
    with fitz.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            if not (page.get_text("text") or "").strip():
                empty_pages.append(i + 1)
    return (not empty_pages), empty_pages


def _assert_every_page_searchable(
    pdf_path: Path, fixture_name: str, profile_name: str
) -> None:
    """Assert the retry-tier invariant: zero empty pages in the output."""
    ok, empty = _output_pdf_has_text_on_every_page(pdf_path)
    assert ok, (
        f"Retry-tier system failed on {fixture_name} × {profile_name}: "
        f"pages {empty} have no text layer. Every page must be recovered "
        f"by the primary attempt, the simplified-settings retry, or the "
        f"last-resort PSM=SPARSE_TEXT tier."
    )


# ---------------------------------------------------------------------------
# Invoice / накладная — the user's real failure shape
# ---------------------------------------------------------------------------


@requires_real_russian_ocr
class TestInvoiceTypicalForm:
    """User-reported failure: page 4 of a typical накладная crashes
    Tesseract's layout analyser even after timeout escalation.

    The new retry-tier system should recover it — if the primary
    PSM=AUTO attempt crashes, PSM=SINGLE_BLOCK at 200 DPI grayscale
    takes over; if that also fails, PSM=SPARSE_TEXT at 150 DPI is the
    final net. No page should end up without text.
    """

    @pytest.mark.parametrize(
        "profile_name",
        ["default", "universal_accurate", "universal_accurate", "universal_accurate"],
    )
    def test_every_page_has_text_layer(
        self,
        profile_name: str,
        adversarial_dir: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = manager.load(profile_name)

        input_pdf = adversarial_dir / "invoice_typical_form.pdf"
        output_pdf = tmp_path / f"invoice_{profile_name}.pdf"

        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        assert result.status is JobStatus.COMPLETED, (
            f"profile={profile_name} FAILED on invoice form: {result.error}"
        )
        _assert_every_page_searchable(
            output_pdf, "invoice_typical_form", profile_name,
        )


# ---------------------------------------------------------------------------
# Dense small text — Tesseract timeout/layout-crash stress
# ---------------------------------------------------------------------------


@requires_real_russian_ocr
class TestDenseSmallText:
    """Full page of 7pt Russian body text. Historically timed out on
    the user's hardware at 600 DPI; the retry tier at 200 DPI grayscale
    must finish in time."""

    def test_output_has_text_layer(
        self,
        adversarial_dir: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        input_pdf = adversarial_dir / "dense_small_text.pdf"
        output_pdf = tmp_path / "dense_small.pdf"

        profile = make_realistic_profile(
            languages=["rus", "eng"],
            binarization="otsu",
            clahe=True,
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        assert result.status is JobStatus.COMPLETED, result.error
        _assert_every_page_searchable(
            output_pdf, "dense_small_text", "otsu+clahe",
        )


# ---------------------------------------------------------------------------
# Stamped page — circular stamp crashes PSM=AUTO layout analyser
# ---------------------------------------------------------------------------


@requires_real_ocr
class TestStampedPage:
    """Circular stamp overlay historically crashes ``PSM.AUTO``. The
    retry tier at ``PSM.SINGLE_BLOCK`` / ``PSM.SPARSE_TEXT`` must
    still recover the body text."""

    def test_body_text_recovered_despite_stamp(
        self,
        adversarial_dir: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        input_pdf = adversarial_dir / "stamped_page.pdf"
        output_pdf = tmp_path / "stamped.pdf"

        profile = make_realistic_profile(
            languages=["eng"],  # use ENG to avoid Cyrillic font coupling
            binarization="otsu",
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        assert result.status is JobStatus.COMPLETED, result.error
        _assert_every_page_searchable(
            output_pdf, "stamped_page", "otsu-eng",
        )


# ---------------------------------------------------------------------------
# Rotated table — 8° rotation past auto-deskew ±5° threshold
# ---------------------------------------------------------------------------


@requires_real_ocr
class TestRotatedTable:
    """A bordered table rotated 8° exceeds the ±5° auto-deskew cap
    and reaches Tesseract still tilted. The retry tier at lower DPI
    with ``PSM.SINGLE_BLOCK`` handles rotated tables more gracefully
    than ``PSM.AUTO``."""

    def test_table_text_recovered(
        self,
        adversarial_dir: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        input_pdf = adversarial_dir / "rotated_table.pdf"
        output_pdf = tmp_path / "rot_table.pdf"

        profile = make_realistic_profile(
            languages=["eng"],
            binarization="otsu",
            deskew=True,
            dpi=300,
        )
        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        assert result.status is JobStatus.COMPLETED, result.error
        _assert_every_page_searchable(
            output_pdf, "rotated_table", "otsu+deskew",
        )


# ---------------------------------------------------------------------------
# 4-page contract — the original failure case this system was built for
# ---------------------------------------------------------------------------


@requires_real_russian_ocr
class TestContract4PageEveryPage:
    """The 4-page Russian contract is the regression test for the
    original "page 4 crashes Tesseract" production bug. After the
    retry-tier fix, every one of the 4 pages must have text regardless
    of which profile processed the document."""

    @pytest.mark.parametrize(
        "profile_name",
        ["default", "universal_accurate", "universal_accurate", "universal_accurate"],
    )
    def test_all_4_pages_searchable(
        self,
        profile_name: str,
        adversarial_dir: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = manager.load(profile_name)

        input_pdf = adversarial_dir / "contract_ru_4page.pdf"
        output_pdf = tmp_path / f"contract4_{profile_name}.pdf"

        result = run_pipeline(
            input_pdf, output_pdf, profile, real_tesseract_wrapper
        )

        assert result.status is JobStatus.COMPLETED, result.error
        assert result.page_count == 4
        _assert_every_page_searchable(
            output_pdf, "contract_ru_4page", profile_name,
        )
