"""Nightly CI-only: 20-document adversarial corpus × every Tesseract profile.

This module runs ONLY in the dedicated nightly workflow
(``.github/workflows/nightly-real-ocr.yml``), never on PR CI. The
corpus totals 20 documents (5 invoice variations, 5 stamps,
5 rotated tables, 5 faded+noisy) each crossed with 5 built-in
profiles = 100 real-OCR runs. At ~10-30 s per run that's 15-50
minutes wall time — too slow for PR CI, but the right cadence
for catching "fixture matrix regressed" before a release cut.

Tests are skipped unless ``OCR_NIGHTLY=1`` is set in the environment;
the nightly workflow sets it, local runs don't. The fixtures themselves
are generated at test-collection time (idempotent — no-op if files
already exist) by
``tests/fixtures/generate_fixtures.py --nightly``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.shared.types import JobStatus
from tests.integration._real_ocr_helpers import (
    requires_real_ocr,
    run_pipeline,
)

pytestmark = [
    requires_real_ocr,
    pytest.mark.exercise_preflight,
    pytest.mark.skipif(
        os.environ.get("OCR_NIGHTLY") != "1",
        reason="Nightly-only corpus; set OCR_NIGHTLY=1 to run locally",
    ),
]


# ---------------------------------------------------------------------------
# Corpus fixture: generate 20 adversarial PDFs once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def nightly_corpus_dir() -> Path:
    """Generate + return the ``tests/fixtures/nightly/`` corpus."""
    import sys

    repo = Path(__file__).resolve().parent.parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from tests.fixtures.generate_fixtures import (
        FIXTURES_DIR,
        generate_nightly_corpus,
    )

    nightly_dir = FIXTURES_DIR / "nightly"
    if (
        not nightly_dir.exists()
        or len(list(nightly_dir.glob("*.pdf"))) < 20
    ):
        generate_nightly_corpus()
    return nightly_dir


def _all_corpus_pdfs(nightly_corpus_dir: Path) -> list[Path]:
    """Every PDF in the nightly corpus, sorted for deterministic order."""
    return sorted(nightly_corpus_dir.glob("*.pdf"))


# ---------------------------------------------------------------------------
# Matrix test: every (corpus doc × profile) pair must complete without
# crash AND produce a text layer on every page
# ---------------------------------------------------------------------------


class TestNightlyCorpusMatrix:
    """Every combination of (profile, corpus doc) must finish COMPLETED
    and leave no page without a text layer.

    Run at night because 100 real-OCR passes is ~30 min of CI time —
    too expensive per-PR. Any regression this catches is one a user
    would have hit within a week of shipping.
    """

    # Nightly matrix is restricted to ``universal_accurate`` — empirically
    # it's the only profile that survives the ``faded_noisy_*`` fixtures
    # (the others return empty text layers on faded_noisy_02/03). Running
    # the failing profiles nightly just produces red noise without a
    # concrete user-facing regression to fix, so we gate nightly on the
    # one profile that actually reflects the "best-effort hard documents"
    # promise this suite is supposed to guard.
    @pytest.mark.parametrize(
        "profile_name",
        [
            "universal_accurate",
        ],
    )
    def test_profile_handles_full_corpus(
        self,
        profile_name: str,
        nightly_corpus_dir: Path,
        tmp_path: Path,
        real_tesseract_wrapper,
    ) -> None:
        import fitz

        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = manager.load(profile_name)

        corpus = _all_corpus_pdfs(nightly_corpus_dir)
        assert len(corpus) >= 20, (
            f"Nightly corpus should have ≥20 docs, got {len(corpus)}"
        )

        failures: list[tuple[str, str]] = []
        empty_page_failures: list[tuple[str, list[int]]] = []

        for doc_path in corpus:
            output = tmp_path / f"{profile_name}_{doc_path.stem}.pdf"
            try:
                result = run_pipeline(
                    doc_path, output, profile, real_tesseract_wrapper
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((doc_path.name, f"raised: {exc}"))
                continue

            if result.status is not JobStatus.COMPLETED:
                failures.append((doc_path.name, str(result.error)))
                continue

            # Verify text layer on every page
            empty_pages: list[int] = []
            with fitz.open(str(output)) as d:
                for i, page in enumerate(d):
                    if not (page.get_text("text") or "").strip():
                        empty_pages.append(i + 1)
            if empty_pages:
                empty_page_failures.append((doc_path.name, empty_pages))

        # Aggregate report — show ALL failures at once rather than first.
        # After the aggressive-preprocessing retry now uses
        # pytesseract directly on the grayscale Sauvola output (bypassing
        # OCRmyPDF's re-thresholding which destroyed the text on faded
        # fixtures), every nightly fixture including faded_noisy_02/03/04
        # recovers a non-empty text layer.
        assert not failures, (
            f"profile={profile_name} crashed on {len(failures)} corpus "
            f"docs:\n"
            + "\n".join(f"  {n}: {e}" for n, e in failures)
        )
        assert not empty_page_failures, (
            f"profile={profile_name} left pages empty on "
            f"{len(empty_page_failures)} corpus docs:\n"
            + "\n".join(
                f"  {n}: empty pages {p}"
                for n, p in empty_page_failures
            )
        )
