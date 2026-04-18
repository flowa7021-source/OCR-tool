"""Accuracy benchmark — measures OCR quality with CER / WER metrics.

This test is the foundation of **every** accuracy-related change in
the codebase. Before introducing a new preprocessing step, tuning a
binariser, or swapping an OCR engine, we run this benchmark to
capture the current CER/WER on our ground-truth corpus. After the
change lands, we run it again: if accuracy regressed, the test
fails. If accuracy improved, we update the baseline explicitly
(via ``--regenerate-baseline`` — see the class docstring below).

The benchmark is INTENTIONALLY slow (≤ 2 min) and gated behind
``OCR_ACCURACY_BENCHMARK=1`` so it only runs in the nightly CI
workflow, never on per-PR CI. Local developers can run it on demand.

Corpus design:
  * Clean synthetic documents only (no adversarial stress tests —
    those live in ``test_e2e_retry_tiers.py``).
  * Ground truth comes from the generator source itself, so the
    expected text is guaranteed to match what was rendered.
  * Covers four realistic shapes: English prose, Russian prose,
    mixed rus+eng memo, Russian invoice with units/codes, dense
    small Russian body.

Metric rationale:
  * **CER (Character Error Rate)** — primary metric. Fine-grained,
    catches single-character slips that WER rounds off.
  * **WER (Word Error Rate)** — secondary. Easier to interpret at
    a glance ("3 out of 100 words wrong") and matches user
    perception of accuracy.
  * Both computed via ``jiwer``, the de-facto standard for OCR/ASR
    evaluation.

Regression guardrail:
  * A change that raises ANY document's CER by > 1 percentage point
    is flagged as a regression — the tolerance window is tight
    because document-level CER on clean synthetic scans should be
    stable to ~0.1% across runs.
  * Hints in the failure message point at where to look
    (preprocessing? postprocessing? engine?).

How to bless a new baseline:
  * After an intentional accuracy improvement, run
    ``pytest tests/integration/test_accuracy_benchmark.py
     --regenerate-baseline``
    — this overwrites ``baseline.json`` with the new numbers.
    Commit the updated baseline with the change that earned it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

pytest.importorskip("jiwer")

from src.shared.types import JobStatus  # noqa: E402
from tests.integration._real_ocr_helpers import (  # noqa: E402
    requires_real_russian_ocr,
    run_pipeline,
)

# Ensure the corpus generator is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


BASELINE_PATH = (
    _REPO_ROOT / "tests" / "fixtures" / "accuracy_corpus" / "baseline.json"
)

#: Tolerance applied on top of each baseline number. If the measured
#: CER for ``ru_business_letter`` is baseline 0.030 and we measure
#: 0.041, the delta (+0.011) exceeds this tolerance (+0.010) and the
#: test fails. Values below tolerance are treated as noise.
CER_REGRESSION_TOLERANCE: float = 0.01  # 1 percentage point

#: Same for WER. Word-level metric is noisier, especially on short
#: documents, so we allow slightly more slack.
WER_REGRESSION_TOLERANCE: float = 0.02  # 2 percentage points


@dataclass(frozen=True)
class DocumentResult:
    """Per-document measurement captured for each benchmark run."""

    name: str
    cer: float
    wer: float
    recognised_chars: int
    ground_truth_chars: int


# ``--regenerate-baseline`` is registered by tests/integration/conftest.py
# (pytest refuses to register ``pytest_addoption`` from a test module
# because the hook is discovered during plugin loading, before test
# modules are imported).

pytestmark = [
    requires_real_russian_ocr,
    pytest.mark.exercise_preflight,
    pytest.mark.skipif(
        # Nightly-only — too slow for per-PR CI, too important to
        # leave out of any cadence.
        __import__("os").environ.get("OCR_ACCURACY_BENCHMARK") != "1",
        reason=(
            "Set OCR_ACCURACY_BENCHMARK=1 to run the accuracy "
            "benchmark locally; nightly CI sets it automatically."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Fixture: build the corpus once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def accuracy_corpus():
    """Generate (or reuse) the ground-truth corpus and return the docs."""
    from tests.fixtures.accuracy_corpus.generate import generate_corpus

    return generate_corpus(force=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_baseline() -> dict[str, dict]:
    """Return the baseline JSON contents, or an empty dict on first run."""
    if not BASELINE_PATH.exists():
        return {}
    try:
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_baseline(results: dict[str, DocumentResult]) -> None:
    """Serialise measurements to ``baseline.json`` for future comparisons."""
    payload = {name: asdict(r) for name, r in results.items()}
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _measure_document(
    doc, tmp_path: Path, real_tesseract_wrapper,
) -> DocumentResult:
    """Run the full pipeline on ``doc`` and compute CER / WER."""
    import jiwer

    from src.application.profile_manager import ProfileManager
    from src.infrastructure.config_storage import ProfileStorage

    storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
    manager = ProfileManager(storage)
    manager.initialize_builtins()
    profile = manager.load("universal_accurate")

    output_pdf = tmp_path / f"{doc.name}_ocr.pdf"
    result = run_pipeline(
        doc.pdf_path, output_pdf, profile, real_tesseract_wrapper,
    )
    assert result.status is JobStatus.COMPLETED, (
        f"Pipeline failed on {doc.name}: {result.error}"
    )

    recognised = "\n".join(
        (p.text or "").strip() for p in result.pages
    ).strip()

    # Normalise whitespace on both sides before metric computation —
    # jiwer is sensitive to line breaks, and neither OCR output nor
    # our ground-truth strings carry semantic meaning in where the
    # newlines happen to fall.
    truth_norm = " ".join(doc.ground_truth.split())
    ocr_norm = " ".join(recognised.split())

    cer = jiwer.cer(truth_norm, ocr_norm)
    wer = jiwer.wer(truth_norm, ocr_norm)

    return DocumentResult(
        name=doc.name,
        cer=round(cer, 4),
        wer=round(wer, 4),
        recognised_chars=len(ocr_norm),
        ground_truth_chars=len(truth_norm),
    )


# ---------------------------------------------------------------------------
# The benchmark test
# ---------------------------------------------------------------------------


class TestAccuracyBenchmark:
    """One document per test for cleaner failure reports.

    Each test measures one corpus document, writes the result into
    the session-scoped ``bench_results`` dict, and compares the
    measurement to the baseline.

    On ``--regenerate-baseline``, the comparison is replaced with
    "always pass" and the collected results overwrite the baseline
    file at session end.
    """

    @pytest.fixture(scope="class", autouse=True)
    def _bench_results(self, request):
        """Shared dict that aggregates per-test measurements."""
        bag: dict[str, DocumentResult] = {}
        yield bag
        # At class teardown: write the baseline if the user asked.
        if request.config.getoption("--regenerate-baseline"):
            _save_baseline(bag)
            print(
                f"\nBaseline regenerated: "
                f"{len(bag)} documents → {BASELINE_PATH}"
            )

    @pytest.mark.parametrize(
        "doc_name",
        [
            "en_business_letter",
            "ru_business_letter",
            "ru_en_mixed_memo",
            "ru_invoice_body",
            "ru_dense_small",
        ],
    )
    def test_document_accuracy(
        self,
        doc_name: str,
        accuracy_corpus,
        tmp_path: Path,
        real_tesseract_wrapper,
        request,
        _bench_results: dict[str, DocumentResult],
    ) -> None:
        doc = next((d for d in accuracy_corpus if d.name == doc_name), None)
        assert doc is not None, f"Corpus missing document {doc_name!r}"

        result = _measure_document(doc, tmp_path, real_tesseract_wrapper)
        _bench_results[doc_name] = result

        # Log for visibility even when the assertion passes.
        print(
            f"\n{doc_name}: CER={result.cer:.4f} WER={result.wer:.4f} "
            f"({result.recognised_chars} / {result.ground_truth_chars} chars)"
        )

        if request.config.getoption("--regenerate-baseline"):
            pytest.skip(
                "--regenerate-baseline: skipping assertion, will "
                "rewrite baseline at class teardown"
            )

        baseline = _load_baseline().get(doc_name)
        if baseline is None:
            pytest.skip(
                f"No baseline for {doc_name!r} yet. Run with "
                "--regenerate-baseline once to capture it."
            )

        baseline_cer = float(baseline["cer"])
        baseline_wer = float(baseline["wer"])

        assert result.cer <= baseline_cer + CER_REGRESSION_TOLERANCE, (
            f"CER regression on {doc_name}: measured "
            f"{result.cer:.4f}, baseline {baseline_cer:.4f}, "
            f"tolerance +{CER_REGRESSION_TOLERANCE:.4f}. "
            f"If the regression is intentional, rerun with "
            f"--regenerate-baseline."
        )
        assert result.wer <= baseline_wer + WER_REGRESSION_TOLERANCE, (
            f"WER regression on {doc_name}: measured "
            f"{result.wer:.4f}, baseline {baseline_wer:.4f}, "
            f"tolerance +{WER_REGRESSION_TOLERANCE:.4f}. "
            f"If the regression is intentional, rerun with "
            f"--regenerate-baseline."
        )
