"""Active learning: find word-crops where the OCR engine and our
post-correction disagree, so humans can prioritise hand-labeling them.

Rationale: the fine-tune dataset currently has 2929 crops covering
185 unique labels. Most are redundant ("4" ×740 times) — the model
already sees that pattern plenty. What helps most is adding labels
for cases the CURRENT model gets wrong.

This script:
    1. Runs the current EasyOCR model over ``inputs/*.pdf``.
    2. For each word-crop, computes a **disagreement score** from:
       - OCR confidence (low = uncertain)
       - Whether the raw OCR output matches any GT token (no match =
         likely wrong)
       - Whether the requisite validator rescues it (rescued = the
         model produced a recoverable string, only moderately useful)
       - Whether the domain LM has a close match (close match but
         different word = high-information sample)
    3. Emits a ranked list of the top-N most informative crops,
       each with its image, OCR prediction, best LM candidate, and
       disagreement score — ready for human review.

Output is a TSV you can open in a spreadsheet with image previews,
or point at Label Studio / doccano for batch annotation. Each
labeled crop goes back into ``inputs/*.txt`` (for the GT corpus)
and the next bootstrap run picks it up.

Usage::

    # Rank the top 200 disagreement candidates into a review file
    python scripts/finetune/active_learning.py --top 200

    # With a different conf threshold (less aggressive)
    python scripts/finetune/active_learning.py --top 500 --conf-threshold 0.7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"
EXPECTED_DIR = REPO_ROOT / "expected"
DEFAULT_OUT = REPO_ROOT / "datasets" / "active_learning"


@dataclass
class DisagreementSample:
    pdf_stem: str
    page: int
    box_idx: int
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in page coords
    raw_ocr: str
    ocr_conf: float
    lm_best_match: str | None
    lm_edit_distance: int
    score: float
    crop_path: Path | None = None


def _score(
    raw_ocr: str,
    ocr_conf: float,
    lm_best: str | None,
    lm_edit: int,
    gt_tokens: set[str],
) -> float:
    """Disagreement score — higher = more informative sample.

    Components (each in [0, 1], weighted and summed):
        uncertainty: 1 - ocr_conf
        no_gt_match: 1 if raw_ocr not in GT set else 0
        near_miss:   sigmoidal bonus when lm has a close match (edit
                     distance 1-2) — the sample is "wrong but fixable",
                     ideal for teaching.
        length_penalty: small penalty on very short raw strings (digits
                     like "4" dominate the low-conf pool but teach
                     little — we want words).
    """
    uncertainty = 1.0 - ocr_conf
    no_gt = 1.0 if raw_ocr.lower() not in gt_tokens else 0.0
    near_miss = 0.0
    if lm_best is not None and 1 <= lm_edit <= 2:
        # Exponentially decaying bonus: Lev=1 → 0.8, Lev=2 → 0.45
        near_miss = 0.8 * (0.56 ** (lm_edit - 1))
    length_penalty = max(0.0, (3 - len(raw_ocr)) / 6.0)
    return (
        0.45 * uncertainty
        + 0.35 * no_gt
        + 0.25 * near_miss
        - length_penalty
    )


def rank_disagreements(
    inputs_dir: Path = INPUTS_DIR,
    expected_dir: Path = EXPECTED_DIR,
    conf_threshold: float = 0.8,
    top_n: int = 200,
    dpi: int = 300,
) -> list[DisagreementSample]:
    """Run OCR over every PDF, return the top-N disagreement samples."""
    import easyocr

    from src.shared.domain_lm import DomainLM

    # Lazy import so this script works even if bootstrap hasn't been run.
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "finetune"))
    from bootstrap_dataset import _load_gt_tokens

    pdfs = sorted(inputs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[active_learning] No PDFs in {inputs_dir}", file=sys.stderr)
        return []

    print("[active_learning] Building domain LM …")
    lm = DomainLM.from_default_corpus(inputs_dir, expected_dir)
    print(f"  vocab: {len(lm)} unique tokens")

    print("[active_learning] Loading EasyOCR (ru+en, CPU) …")
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)

    candidates: list[DisagreementSample] = []
    for pdf in pdfs:
        gt_tokens = _load_gt_tokens(pdf.stem, inputs_dir, expected_dir)
        print(f"[active_learning] {pdf.stem}: {len(gt_tokens)} GT tokens, "
              f"OCR …")
        doc = fitz.open(str(pdf))
        try:
            for page_idx in range(doc.page_count):
                pix = doc[page_idx].get_pixmap(
                    dpi=dpi, colorspace=fitz.csGRAY,
                )
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width,
                )
                for box_idx, (bbox, text, conf) in enumerate(
                    reader.readtext(arr, detail=1, paragraph=False)
                ):
                    if not text.strip() or conf >= conf_threshold:
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
                    x1, y1 = int(max(xs)), int(max(ys))
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    # LM correction on the OCR output.
                    lm_result = lm.correct(text.strip(), ocr_conf=conf)
                    score = _score(
                        raw_ocr=text.strip(),
                        ocr_conf=conf,
                        lm_best=lm_result.word if lm_result.was_corrected else None,
                        lm_edit=lm_result.edit_distance,
                        gt_tokens=gt_tokens,
                    )
                    candidates.append(DisagreementSample(
                        pdf_stem=pdf.stem,
                        page=page_idx,
                        box_idx=box_idx,
                        bbox=(x0, y0, x1, y1),
                        raw_ocr=text.strip(),
                        ocr_conf=conf,
                        lm_best_match=(
                            lm_result.word if lm_result.was_corrected else None
                        ),
                        lm_edit_distance=lm_result.edit_distance,
                        score=score,
                    ))
        finally:
            doc.close()

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_n]


def dump_samples(
    samples: list[DisagreementSample],
    out_dir: Path,
    inputs_dir: Path,
    dpi: int = 300,
) -> None:
    """Save the TSV + crop images for human review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "disagreements.tsv"
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    # Open each source PDF once; crop in memory.
    pdf_cache: dict[str, fitz.Document] = {}
    try:
        with tsv.open("w", encoding="utf-8") as f:
            f.write(
                "crop_path\tpdf\tpage\traw_ocr\tocr_conf\tlm_best\tlm_edit\t"
                "score\n"
            )
            for i, s in enumerate(samples):
                pdf_path = inputs_dir / f"{s.pdf_stem}.pdf"
                if s.pdf_stem not in pdf_cache:
                    pdf_cache[s.pdf_stem] = fitz.open(str(pdf_path))
                doc = pdf_cache[s.pdf_stem]
                pix = doc[s.page].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width,
                )
                x0, y0, x1, y1 = s.bbox
                crop = arr[y0:y1, x0:x1]
                crop_path = crops_dir / f"{i:05d}_{s.pdf_stem}_{s.page}.jpg"
                Image.fromarray(crop).save(crop_path, "JPEG", quality=90)
                f.write(
                    f"{crop_path.name}\t{s.pdf_stem}\t{s.page}\t"
                    f"{s.raw_ocr}\t{s.ocr_conf:.3f}\t"
                    f"{s.lm_best_match or ''}\t{s.lm_edit_distance}\t"
                    f"{s.score:.3f}\n"
                )
    finally:
        for doc in pdf_cache.values():
            doc.close()
    print(f"[active_learning] Wrote {len(samples)} samples → {tsv}")
    print(f"[active_learning]   crops dir: {crops_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUTS_DIR)
    parser.add_argument("--expected", type=Path, default=EXPECTED_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top", type=int, default=200,
                        help="Number of top-disagreement samples (%(default)s)")
    parser.add_argument("--conf-threshold", type=float, default=0.8,
                        help="Only consider crops with OCR conf below this "
                             "(%(default)s)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    samples = rank_disagreements(
        args.inputs, args.expected,
        conf_threshold=args.conf_threshold,
        top_n=args.top,
        dpi=args.dpi,
    )
    if not samples:
        print("[active_learning] No candidates generated.", file=sys.stderr)
        return 1
    dump_samples(samples, args.out, args.inputs, dpi=args.dpi)
    print("\nNext steps:")
    print(f"  1. Open {args.out / 'disagreements.tsv'} + crops/ in a")
    print("     spreadsheet or Label Studio.")
    print("  2. For each crop, write the correct label into a column.")
    print("  3. Append labelled pairs to inputs/<pdf>.txt (they'll be")
    print("     picked up on the next bootstrap run).")
    print("  4. `python scripts/finetune/bootstrap_dataset.py` → "
          "re-fine-tune.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
