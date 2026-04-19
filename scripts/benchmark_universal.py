"""OCR accuracy benchmark for a single real document.

The fastest feedback loop when tuning the preprocessing / post-
processing / confidence filter: point this script at a real PDF, read
the histogram, change one knob, rerun. Replaces "eyeball the results
panel" with numbers that answer:

  * What fraction of the words Tesseract emitted does the profile
    actually KEEP after the word-confidence filter?
  * Where does the per-word confidence distribution concentrate —
    a bimodal "clean body + noisy stamps" shape or a uniform mush?
  * What WORDS specifically fall below the threshold?  Looking at
    the top-20 dropped words tells you immediately whether the
    filter is eating legitimate rare terms ("ГЕКСАФОРМ") or just
    cutting real junk ("Taw", "нe", "||", "°").

Intended use::

    python scripts/benchmark_universal.py path/to/scan.pdf
    python scripts/benchmark_universal.py path/to/scan.pdf --profile quick_reliable
    python scripts/benchmark_universal.py path/to/scan.pdf --threshold 50

Output is plain-text and designed to be greppable — pipe it to a file
and diff two runs to see the delta from a preprocessing tweak.

Deliberate non-goals:

  * Does not mutate the profile or any on-disk config.
  * Does not compare against a ground-truth corpus (that's the
    job of ``tests/integration/test_accuracy_benchmark.py`` and
    its ``jiwer`` CER/WER pass).  This tool is for iteration on
    a user-specific document where no ground truth exists.
  * Does not update ``baseline.json`` — this is throwaway
    measurement for the dev loop.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Repo-root import path bootstrap so the script works via
# ``python scripts/benchmark_universal.py`` from a fresh checkout
# without ``pip install -e``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class PageStats:
    """Per-page measurement captured for the summary table."""

    page: int
    total_words: int
    kept_words: int
    mean_conf_all: float
    mean_conf_kept: float
    # Parallel lists so we can build aggregate histograms later.
    confs: list[float]
    words: list[str]


def _build_pipeline():
    """Construct OCRPipeline with compute_confidence=True so the filter
    path runs even when drop_low_conf_words is off (we still want the
    numbers)."""
    from src.application.pipeline import OCRPipeline
    from src.core.image_preprocessor import ImagePreprocessor
    from src.core.text_postprocessor import TextPostprocessor
    from src.infrastructure.tesseract_wrapper import TesseractWrapper

    # Register bundled Tesseract / Ghostscript on PATH, same as
    # ``src.cli.main`` does. Without this the script fails on a fresh
    # installed-app checkout where ``shutil.which('gswin64c')`` returns
    # None even though the binary is sitting under
    # ``resources/ghostscript/bin/``. Swallowed if the module isn't
    # importable so dev-only checkouts without a bundle still run.
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


def _load_profile(name: str):
    """Return the named built-in profile (creates profiles dir if missing)."""
    import tempfile

    from src.application.profile_manager import ProfileManager
    from src.infrastructure.config_storage import ProfileStorage

    # Use a tmp dir so we don't touch user's real profile directory.
    profiles_dir = Path(tempfile.mkdtemp(prefix="benchmark-profiles-"))
    storage = ProfileStorage(profiles_dir=profiles_dir)
    manager = ProfileManager(storage)
    manager.initialize_builtins()
    return storage.load(name)


def _collect_per_word_stats(
    pipeline, profile, input_pdf: Path, output_pdf: Path,
) -> tuple[list[PageStats], float]:
    """Run the pipeline and return per-page word statistics.

    Replicates the ``_compute_confidences`` inner loop without
    mutating the PageResult, so the same numbers appear here that
    the production pipeline would record.
    """
    from src.core.models import OCRJobConfig
    from src.shared.types import JobStatus

    started = time.time()
    result = pipeline.run(
        OCRJobConfig(
            input_path=str(input_pdf),
            output_path=str(output_pdf),
            profile=profile,
        )
    )
    elapsed = time.time() - started

    if result.status is not JobStatus.COMPLETED:
        print(f"[fail] Job status: {result.status}, error: {result.error}",
              file=sys.stderr)
        sys.exit(2)

    # Re-run image_to_data ourselves so we have per-word conf without
    # depending on whether the production pipeline kept the data.
    # This is cheap compared to the OCR itself.
    stats = _rescore_per_page(pipeline, profile, input_pdf, output_pdf)
    return stats, elapsed


def _rescore_per_page(pipeline, profile, input_pdf: Path, output_pdf: Path):
    """Gather per-word conf for every page of ``output_pdf`` by
    rasterising + running ``pytesseract.image_to_data`` ourselves.
    Uses the same config knobs (psm/oem/lang) the pipeline used."""
    import cv2
    import fitz
    import numpy as np
    import pytesseract

    lang = profile.ocr.tesseract_language_string
    tess_cfg = (
        f"--psm {int(profile.ocr.psm)} "
        f"--oem {int(profile.ocr.oem)}"
    )

    stats: list[PageStats] = []
    with fitz.open(str(input_pdf)) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=profile.ocr.dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n,
            )
            img = (
                cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                if pix.n == 3
                else arr.squeeze()
            )
            data = pytesseract.image_to_data(
                img, lang=lang, config=tess_cfg,
                output_type=pytesseract.Output.DICT,
            )
            confs: list[float] = []
            words: list[str] = []
            for w, c in zip(data.get("text", []), data.get("conf", []),
                            strict=False):
                if not isinstance(w, str) or not w.strip():
                    continue
                try:
                    cf = float(c)
                except (TypeError, ValueError):
                    continue
                if cf < 0:
                    continue
                confs.append(cf)
                words.append(w.strip())

            kept = [cf for cf in confs
                    if cf >= profile.ocr.confidence_threshold]
            stats.append(
                PageStats(
                    page=i + 1,
                    total_words=len(confs),
                    kept_words=len(kept),
                    mean_conf_all=statistics.fmean(confs) if confs else 0.0,
                    mean_conf_kept=statistics.fmean(kept) if kept else 0.0,
                    confs=confs,
                    words=words,
                )
            )
    return stats


def _histogram(confs: list[float], buckets: int = 10) -> list[int]:
    """Return ``buckets`` counts covering 0..100 conf range."""
    hist = [0] * buckets
    width = 100.0 / buckets
    for c in confs:
        idx = min(int(c / width), buckets - 1)
        hist[idx] += 1
    return hist


def _format_histogram(hist: list[int]) -> str:
    """Return an ASCII bar chart with bucket labels on the left."""
    total = sum(hist) or 1
    lines: list[str] = []
    bucket_w = 100 // len(hist)
    for i, count in enumerate(hist):
        lo, hi = i * bucket_w, (i + 1) * bucket_w
        pct = count / total
        bar = "█" * int(pct * 40)
        lines.append(f"  [{lo:3d}-{hi:3d}]  {count:5d}  {bar}")
    return "\n".join(lines)


def _top_words(
    words: list[str], confs: list[float], *, threshold: float,
    keep_side: str, n: int = 20,
) -> list[tuple[str, float, int]]:
    """Return the top-``n`` words on one side of the threshold,
    aggregated by (word, rounded-conf) so you see "this specific
    token Tesseract was THIS confident about, N times"."""
    if keep_side == "below":
        filtered = [
            (w, c) for w, c in zip(words, confs, strict=True)
            if c < threshold
        ]
    else:  # "above"
        filtered = [
            (w, c) for w, c in zip(words, confs, strict=True)
            if c >= threshold
        ]
    # Aggregate by word: sum counts, avg conf.
    agg: dict[str, list[float]] = {}
    for w, c in filtered:
        agg.setdefault(w, []).append(c)
    ranked = sorted(
        (
            (w, statistics.fmean(cs), len(cs))
            for w, cs in agg.items()
        ),
        # Primary sort: by mean conf (low first for dropped, high first
        # for kept) so the most-distinctive entries show up first.
        key=lambda t: (t[1] if keep_side == "below" else -t[1], -t[2]),
    )
    return ranked[:n]


def _print_report(
    stats: list[PageStats], threshold: float, elapsed: float,
    pdf_path: Path, profile_name: str,
) -> None:
    """Pretty-print the summary table + per-page rows + top words."""
    print("=" * 70)
    print(f"Benchmark: {pdf_path.name}  ({profile_name})")
    print(f"Threshold: conf >= {threshold:.1f}")
    print(f"Pipeline wall time: {elapsed:.1f}s")
    print("=" * 70)

    total_words = sum(s.total_words for s in stats)
    kept_words = sum(s.kept_words for s in stats)
    all_confs = [c for s in stats for c in s.confs]

    if total_words == 0:
        print("[warn] no words — pipeline produced empty OCR output")
        return

    kept_ratio = 100.0 * kept_words / total_words
    print()
    print(f"Pages ............... {len(stats)}")
    print(f"Total words ......... {total_words}")
    print(f"Kept (conf >= thr) .. {kept_words} ({kept_ratio:.1f}%)")
    print(f"Dropped ............. {total_words - kept_words} "
          f"({100 - kept_ratio:.1f}%)")
    print(f"Mean conf (all) ..... {statistics.fmean(all_confs):.1f}")
    kept_confs = [c for c in all_confs if c >= threshold]
    if kept_confs:
        print(f"Mean conf (kept) .... {statistics.fmean(kept_confs):.1f}")
    print()
    print("Per-page breakdown:")
    print(f"{'page':>5} {'words':>6} {'kept':>6} "
          f"{'% kept':>7} {'conf-all':>9} {'conf-kept':>10}")
    for s in stats:
        pct = 100.0 * s.kept_words / max(s.total_words, 1)
        print(f"{s.page:>5} {s.total_words:>6} {s.kept_words:>6} "
              f"{pct:>6.1f}% {s.mean_conf_all:>9.1f} "
              f"{s.mean_conf_kept:>10.1f}")
    print()

    print("Confidence histogram (all words, all pages):")
    print(_format_histogram(_histogram(all_confs)))
    print()

    all_words = [w for s in stats for w in s.words]
    dropped = _top_words(all_words, all_confs, threshold=threshold,
                         keep_side="below", n=20)
    kept_top = _top_words(all_words, all_confs, threshold=threshold,
                          keep_side="above", n=20)

    print(f"Top-20 DROPPED words (conf < {threshold:.0f}):")
    if not dropped:
        print("  (none — filter is a no-op on this document)")
    else:
        for word, mean_conf, count in dropped:
            print(f"  {mean_conf:5.1f}  ×{count:3d}  {word!r}")
    print()
    print(f"Top-20 KEPT words (conf >= {threshold:.0f}):")
    for word, mean_conf, count in kept_top:
        print(f"  {mean_conf:5.1f}  ×{count:3d}  {word!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark one PDF against a named profile.",
    )
    parser.add_argument("pdf", type=Path, help="Input PDF")
    parser.add_argument(
        "--profile", default="universal_accurate",
        help="Built-in profile name (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold for the DROP/KEEP split "
             "(default: profile's confidence_threshold)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the searchable PDF (default: <pdf>.ocr.pdf)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        print(f"[error] not a file: {args.pdf}", file=sys.stderr)
        return 2

    profile = _load_profile(args.profile)
    threshold = (
        args.threshold
        if args.threshold is not None
        else float(profile.ocr.confidence_threshold)
    )
    output_pdf = args.output or args.pdf.with_suffix(".ocr.pdf")

    pipeline = _build_pipeline()
    stats, elapsed = _collect_per_word_stats(
        pipeline, profile, args.pdf, output_pdf,
    )
    _print_report(stats, threshold, elapsed, args.pdf, args.profile)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
