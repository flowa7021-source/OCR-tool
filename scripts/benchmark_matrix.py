"""Benchmark matrix — measure OCR accuracy across profiles × DPIs × docs.

Companion to :mod:`scripts.benchmark_universal`. Where that script
iterates on ONE profile for ONE document to tune knobs, this one
sweeps the whole matrix and spits out a table so you can answer the
"which profile and DPI are best for my documents?" question from
hard numbers rather than spot-checks.

Usage::

    # Run the bundled synthetic corpus through the 6 builtin profiles
    # at 200 / 300 / 400 / 500 / 600 DPI and print a table.
    python scripts/benchmark_matrix.py

    # Point at a directory of user documents (see "Corpus layout"
    # below) and restrict the sweep to two profiles + two DPIs.
    python scripts/benchmark_matrix.py \\
        --corpus ~/ocr-test-docs \\
        --profiles universal_accurate quick_reliable \\
        --dpis 300 400

    # Write the full table to CSV for spreadsheet import.
    python scripts/benchmark_matrix.py --csv results.csv

Corpus layout (custom ``--corpus``)::

    ~/ocr-test-docs/
        invoice_01.pdf
        invoice_01.gt.txt        ← ground-truth plain text, UTF-8
        contract_05.pdf
        contract_05.gt.txt
        ...

Each PDF must have a matching ``.gt.txt`` (same stem). The ground
truth is the text you'd expect a perfect OCR to produce — paragraph
breaks don't matter, the script normalises whitespace on both sides.

Metrics per cell:
    * **CER** — character error rate (lower is better; 0.01 = 1%)
    * **WER** — word error rate
    * **mean_conf** — Tesseract's self-reported per-word confidence,
      averaged across pages
    * **sec** — wall-clock time for the OCR run

Output:
    * Console — compact table per document + a "best cell" summary
    * Optional CSV — full grid for post-hoc analysis

Non-goals:
    * Does not run preflight or touch ``baseline.json``.
    * Does not parallelise across cells; you can ``parallel`` the
      outer loop in a shell if a multi-hour sweep is needed.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Repo-root import path bootstrap so the script works via
# ``python scripts/benchmark_matrix.py`` from a fresh checkout
# without ``pip install -e``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class Cell:
    """One measurement cell in the matrix."""

    document: str
    profile: str
    dpi: int
    cer: float
    wer: float
    mean_conf: float
    elapsed_sec: float
    error: str = ""


@dataclass
class CorpusDoc:
    """A document + its ground-truth text."""

    name: str
    pdf_path: Path
    ground_truth: str


def _load_user_corpus(corpus_dir: Path) -> list[CorpusDoc]:
    """Discover PDF / .gt.txt pairs under ``corpus_dir``.

    Raises :class:`ValueError` when no matched pair is found so the
    user gets a clear "your corpus directory is empty or malformed"
    signal instead of an ambiguous zero-doc run.
    """
    docs: list[CorpusDoc] = []
    for pdf in sorted(corpus_dir.glob("*.pdf")):
        gt_path = pdf.with_suffix(".gt.txt")
        if not gt_path.is_file():
            print(
                f"[warn] {pdf.name} has no matching .gt.txt — skipping",
                file=sys.stderr,
            )
            continue
        docs.append(
            CorpusDoc(
                name=pdf.stem,
                pdf_path=pdf,
                ground_truth=gt_path.read_text(encoding="utf-8"),
            )
        )
    if not docs:
        raise ValueError(
            f"No PDF / .gt.txt pairs found under {corpus_dir}. Expected "
            "each PDF to have a sibling .gt.txt containing the ground "
            "truth text."
        )
    return docs


def _load_synthetic_corpus() -> list[CorpusDoc]:
    """Reuse the bundled synthetic accuracy corpus.

    Lets a fresh checkout run the matrix without any data prep — the
    corpus generator will also lazily materialise the PDFs on first
    call.
    """
    from tests.fixtures.accuracy_corpus.generate import generate_corpus

    return [
        CorpusDoc(
            name=doc.name,
            pdf_path=doc.pdf_path,
            ground_truth=doc.ground_truth,
        )
        for doc in generate_corpus(force=False)
    ]


def _build_pipeline():
    """Construct :class:`OCRPipeline` with confidence computation on.

    Mirrors :func:`benchmark_universal._build_pipeline` — kept copy-
    pasted rather than factored out because the matrix script ships
    separately from the dev-iteration one and the duplication is
    shorter than a third shared module.
    """
    from src.application.pipeline import OCRPipeline
    from src.core.image_preprocessor import ImagePreprocessor
    from src.core.text_postprocessor import TextPostprocessor
    from src.infrastructure.tesseract_wrapper import TesseractWrapper

    try:
        from src.infrastructure.external_tools import ensure_on_path

        ensure_on_path()
    except Exception:  # noqa: BLE001
        pass

    tess = TesseractWrapper()
    try:
        tess.configure_pytesseract()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Tesseract not fully configured: {exc}", file=sys.stderr)

    return OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        tesseract=tess,
        compute_confidence=True,
    )


def _load_profile(name: str, profiles_dir: Path):
    """Return a built-in profile by name from a clean per-run dir."""
    from src.application.profile_manager import ProfileManager
    from src.infrastructure.config_storage import ProfileStorage

    storage = ProfileStorage(profiles_dir=profiles_dir)
    manager = ProfileManager(storage)
    manager.initialize_builtins()
    return storage.load(name)


def _run_cell(
    pipeline, profile, doc: CorpusDoc, dpi: int, output_pdf: Path
) -> Cell:
    """Run one (profile, dpi, doc) combination and return the cell."""
    import dataclasses

    import jiwer

    from src.core.models import OCRJobConfig
    from src.shared.types import JobStatus

    # Override DPI on the profile without mutating the stored copy —
    # ``dataclasses.replace`` gives us a fresh OCRConfig with just the
    # DPI changed, keeping every other profile setting intact.
    ocr_override = dataclasses.replace(profile.ocr, dpi=dpi)
    profile_override = dataclasses.replace(profile, ocr=ocr_override)

    started = time.time()
    try:
        result = pipeline.run(
            OCRJobConfig(
                input_path=str(doc.pdf_path),
                output_path=str(output_pdf),
                profile=profile_override,
            )
        )
    except Exception as exc:  # noqa: BLE001
        return Cell(
            document=doc.name,
            profile=profile.name,
            dpi=dpi,
            cer=float("nan"),
            wer=float("nan"),
            mean_conf=0.0,
            elapsed_sec=time.time() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    elapsed = time.time() - started

    if result.status is not JobStatus.COMPLETED:
        return Cell(
            document=doc.name,
            profile=profile.name,
            dpi=dpi,
            cer=float("nan"),
            wer=float("nan"),
            mean_conf=0.0,
            elapsed_sec=elapsed,
            error=result.error or "unknown failure",
        )

    recognised = "\n".join(
        (p.text or "").strip() for p in result.pages
    ).strip()
    truth_norm = " ".join(doc.ground_truth.split())
    ocr_norm = " ".join(recognised.split())

    # Guard against empty OCR — jiwer treats an empty hypothesis as
    # "delete everything" which gives CER=1.0; that's technically
    # correct but we want to also flag zero-recognition as a
    # distinct failure mode in the error column.
    if not ocr_norm:
        return Cell(
            document=doc.name,
            profile=profile.name,
            dpi=dpi,
            cer=1.0,
            wer=1.0,
            mean_conf=0.0,
            elapsed_sec=elapsed,
            error="empty OCR output",
        )

    cer = float(jiwer.cer(truth_norm, ocr_norm))
    wer = float(jiwer.wer(truth_norm, ocr_norm))
    mean_conf = float(result.average_confidence)

    return Cell(
        document=doc.name,
        profile=profile.name,
        dpi=dpi,
        cer=round(cer, 4),
        wer=round(wer, 4),
        mean_conf=round(mean_conf, 1),
        elapsed_sec=round(elapsed, 1),
    )


def _print_document_block(doc_name: str, cells: list[Cell]) -> None:
    """Print one compact table for the document, sorted by CER."""
    print()
    print(f"=== {doc_name} " + "=" * max(1, 60 - len(doc_name) - 5))
    print(
        f"{'profile':<22} {'dpi':>5} {'CER%':>7} {'WER%':>7} "
        f"{'conf':>6} {'sec':>6}  note"
    )
    for cell in sorted(cells, key=lambda c: (c.cer, c.wer)):
        note = cell.error or ""
        cer_pct = cell.cer * 100 if cell.cer == cell.cer else float("nan")
        wer_pct = cell.wer * 100 if cell.wer == cell.wer else float("nan")
        print(
            f"{cell.profile:<22} {cell.dpi:>5} "
            f"{cer_pct:>6.2f}% {wer_pct:>6.2f}% "
            f"{cell.mean_conf:>5.1f}% {cell.elapsed_sec:>5.1f}s  {note}"
        )


def _print_best_summary(cells: list[Cell]) -> None:
    """Aggregate winners per document and per profile/dpi."""
    by_doc: dict[str, list[Cell]] = {}
    for c in cells:
        by_doc.setdefault(c.document, []).append(c)

    print()
    print("=" * 70)
    print("Best cell per document (lowest CER; ties broken by WER):")
    print("=" * 70)
    print(f"{'document':<26} {'profile':<22} {'dpi':>5} {'CER%':>7} {'WER%':>7}")
    for doc_name, doc_cells in by_doc.items():
        valid = [c for c in doc_cells if not c.error]
        if not valid:
            print(f"{doc_name:<26} (every cell errored)")
            continue
        best = min(valid, key=lambda c: (c.cer, c.wer))
        print(
            f"{doc_name:<26} {best.profile:<22} "
            f"{best.dpi:>5} {best.cer * 100:>6.2f}% {best.wer * 100:>6.2f}%"
        )

    print()
    print("Average CER per (profile, dpi):")
    by_combo: dict[tuple[str, int], list[float]] = {}
    for c in cells:
        if c.error or c.cer != c.cer:
            continue
        by_combo.setdefault((c.profile, c.dpi), []).append(c.cer)
    ranked = sorted(
        (
            (profile, dpi, statistics.fmean(cers), len(cers))
            for (profile, dpi), cers in by_combo.items()
        ),
        key=lambda t: t[2],
    )
    print(f"{'profile':<22} {'dpi':>5} {'avg CER%':>9} {'n':>3}")
    for profile, dpi, avg_cer, n in ranked:
        print(f"{profile:<22} {dpi:>5} {avg_cer * 100:>8.2f}% {n:>3}")


def _write_csv(cells: list[Cell], csv_path: Path) -> None:
    """Dump every cell to ``csv_path`` in a spreadsheet-friendly format."""
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["document", "profile", "dpi", "cer", "wer",
             "mean_conf", "elapsed_sec", "error"]
        )
        for c in cells:
            writer.writerow(
                [c.document, c.profile, c.dpi, c.cer, c.wer,
                 c.mean_conf, c.elapsed_sec, c.error]
            )
    print(f"\n[ok] Wrote {len(cells)} cells → {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep profile × DPI × document for accuracy.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "Directory with PDF + matching .gt.txt pairs. Defaults to "
            "the bundled synthetic accuracy corpus."
        ),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=[
            "universal_accurate",
            "default",
            "quick_reliable",
            "low_quality_scan",
            "contracts_ru",
        ],
        help="Built-in profiles to sweep (space-separated).",
    )
    parser.add_argument(
        "--dpis",
        nargs="+",
        type=int,
        default=[300, 400, 500, 600],
        help="DPIs to sweep (space-separated integers).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write full results to CSV at this path.",
    )
    args = parser.parse_args()

    try:
        import jiwer  # noqa: F401  — hard dependency check at script entry
    except ImportError:
        print(
            "[error] jiwer is required for CER/WER computation. Install "
            "with `pip install jiwer`.",
            file=sys.stderr,
        )
        return 2

    if args.corpus is not None:
        if not args.corpus.is_dir():
            print(f"[error] not a directory: {args.corpus}", file=sys.stderr)
            return 2
        docs = _load_user_corpus(args.corpus)
        source = f"user corpus at {args.corpus}"
    else:
        docs = _load_synthetic_corpus()
        source = "bundled synthetic corpus"

    print(f"[info] Loaded {len(docs)} document(s) from {source}")
    print(
        f"[info] Sweeping {len(args.profiles)} profile(s) × "
        f"{len(args.dpis)} DPI(s) = "
        f"{len(docs) * len(args.profiles) * len(args.dpis)} cell(s)"
    )

    pipeline = _build_pipeline()
    profiles_dir = Path(tempfile.mkdtemp(prefix="benchmark-matrix-"))
    work_dir = Path(tempfile.mkdtemp(prefix="benchmark-matrix-out-"))

    cells: list[Cell] = []
    for doc in docs:
        doc_cells: list[Cell] = []
        for profile_name in args.profiles:
            try:
                profile = _load_profile(profile_name, profiles_dir)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[warn] could not load profile {profile_name!r}: {exc}",
                    file=sys.stderr,
                )
                continue
            for dpi in args.dpis:
                output_pdf = (
                    work_dir / f"{doc.name}__{profile_name}__{dpi}.pdf"
                )
                print(
                    f"  running {doc.name} × {profile_name} × {dpi} DPI ...",
                    file=sys.stderr,
                )
                cell = _run_cell(pipeline, profile, doc, dpi, output_pdf)
                doc_cells.append(cell)
                cells.append(cell)
        _print_document_block(doc.name, doc_cells)

    _print_best_summary(cells)

    if args.csv is not None:
        _write_csv(cells, args.csv)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
