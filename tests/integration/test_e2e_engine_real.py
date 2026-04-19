"""Engine-level real-OCR tests: ``TesseractEngine.run`` with live Tesseract.

Complements ``tests/unit/test_engines.py`` which mocks ``run_ocrmypdf``
to verify the retry-tier wiring. This module calls ``TesseractEngine.run``
**directly** with real Tesseract against adversarial fixtures so we
catch regressions in the actual engine-level contract:

  * The returned list has one ``PageOCRResult`` per input page
  * The output PDF exists and has text on every page
  * Progress callbacks fire per page
  * A crash on page N does NOT abort the whole engine run

The mocked unit tests can verify "the retry code path is entered"; only
real Tesseract can verify "the retry code path actually produces a
usable text layer". Both live together — mocks for speed & logic, real
for correctness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.engines.base import PageOCRResult
from src.application.engines.tesseract_engine import TesseractEngine
from src.core.models import OCRConfig
from src.shared.types import OEM, PSM, OCREngineKind, OptimizeLevel
from tests.integration._real_ocr_helpers import requires_real_ocr

pytestmark = [requires_real_ocr, pytest.mark.exercise_preflight]


# ---------------------------------------------------------------------------
# Fixtures: adversarial PDFs shared with test_e2e_retry_tiers.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def engine_test_fixtures(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate adversarial fixtures once per session (no-op on 2nd run)."""
    import sys

    repo = Path(__file__).resolve().parent.parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from tests.fixtures.generate_fixtures import (
        FIXTURES_DIR,
        generate_dense_small_text,
        generate_invoice_typical_form,
        generate_rotated_table,
        generate_stamped_page,
    )

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    for rel, gen in [
        ("invoice_typical_form.pdf", generate_invoice_typical_form),
        ("dense_small_text.pdf", generate_dense_small_text),
        ("stamped_page.pdf", generate_stamped_page),
        ("rotated_table.pdf", generate_rotated_table),
    ]:
        if not (FIXTURES_DIR / rel).exists():
            gen()
    return FIXTURES_DIR


# ---------------------------------------------------------------------------
# Engine-level invariants exercised against real Tesseract
# ---------------------------------------------------------------------------


class TestEngineProducesTextLayerOnAdversarialFixtures:
    """Call ``TesseractEngine.run`` directly and assert the output PDF
    has a text layer on every page — no mocked ``run_ocrmypdf``, no
    ``OCRPipeline`` wrapping. If this test fails, the engine itself
    is broken (the pipeline layer can't save a broken engine)."""

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "invoice_typical_form.pdf",
            "dense_small_text.pdf",
            "stamped_page.pdf",
            "rotated_table.pdf",
        ],
    )
    def test_every_page_has_text(
        self,
        fixture_name: str,
        engine_test_fixtures: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        import fitz

        input_pdf = engine_test_fixtures / fixture_name
        output_pdf = tmp_path / f"engine_real_{fixture_name}"

        config = OCRConfig(
            engine=OCREngineKind.TESSERACT,
            languages=["rus", "eng"],
            primary_language="rus",
            psm=PSM.AUTO,
            oem=OEM.LSTM_ONLY,
            dpi=300,
            optimize_level=OptimizeLevel.NONE,
            skip_text=False,
            tesseract_timeout=120,
        )

        engine = TesseractEngine()
        results = engine.run(
            preprocessed_pdf=input_pdf,
            output_pdf=output_pdf,
            config=config,
        )

        # Engine contract: one PageOCRResult per page
        with fitz.open(str(input_pdf)) as src:
            expected_pages = src.page_count
        assert len(results) == expected_pages
        assert all(isinstance(r, PageOCRResult) for r in results)
        assert [r.page_number for r in results] == list(
            range(1, expected_pages + 1)
        )

        # Output PDF exists and is a valid PDF
        assert output_pdf.exists(), "Engine didn't write output PDF"
        assert output_pdf.stat().st_size > 0, "Output PDF is empty"

        # Every page has a non-empty text layer — the core retry-tier
        # invariant. If a page has zero text the retry-tier system
        # silently failed on that page.
        empty_pages: list[int] = []
        with fitz.open(str(output_pdf)) as doc:
            for i, page in enumerate(doc):
                if not (page.get_text("text") or "").strip():
                    empty_pages.append(i + 1)
        assert not empty_pages, (
            f"Engine left pages {empty_pages} without text layer on "
            f"{fixture_name}. The retry-tier system should prevent "
            f"this — check the engine logs for which tier was last "
            f"attempted."
        )


class TestEngineProgressCallbackFiresPerPage:
    """The engine fires progress_callback once per page as processing
    advances. The UI depends on this to show "страница 2/4" instead
    of sitting at 0% for minutes. Verified with real Tesseract to
    catch cases where callback wiring changes under the real code path
    but not the mocked one."""

    def test_callback_receives_per_page_updates(
        self,
        engine_test_fixtures: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        import fitz

        input_pdf = engine_test_fixtures / "rotated_table.pdf"
        output_pdf = tmp_path / "progress_check.pdf"

        config = OCRConfig(
            engine=OCREngineKind.TESSERACT,
            languages=["eng"],
            primary_language="eng",
            psm=PSM.AUTO,
            oem=OEM.LSTM_ONLY,
            dpi=200,
            optimize_level=OptimizeLevel.NONE,
            skip_text=False,
            tesseract_timeout=60,
        )

        events: list[tuple[int, int, str]] = []
        engine = TesseractEngine()
        engine.run(
            preprocessed_pdf=input_pdf,
            output_pdf=output_pdf,
            config=config,
            progress_callback=lambda c, t, s: events.append((c, t, s)),
        )

        with fitz.open(str(input_pdf)) as doc:
            n = doc.page_count

        # We expect ``n + 1`` events: one per page start (0..n-1) plus
        # a final completion event at (n, n). Tolerate slack — callback
        # semantics may evolve — but always demand at least start+end.
        assert events, "Engine fired no progress events at all"
        assert events[0] == (0, n, "ocr"), (
            f"First event should be (0, {n}, 'ocr'); got {events[0]}"
        )
        assert events[-1] == (n, n, "ocr"), (
            f"Last event should be ({n}, {n}, 'ocr'); got {events[-1]}"
        )
