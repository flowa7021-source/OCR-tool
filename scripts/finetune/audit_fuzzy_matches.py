"""Audit of fuzzy-matched labels in the bootstrap dataset.

Runs the same logic as ``bootstrap_dataset.py`` but ONLY captures the
cases where the GT substitution was a Levenshtein-2 (or higher) edit
from the raw OCR output — i.e. the samples most at risk of being
wrongly auto-corrected. For each such pair, saves the crop image + a
TSV row with:

    filename  pdf_stem  page  raw_ocr  label_after_correction  edit_distance

Open the TSV in a spreadsheet alongside the crops/ folder: for each
row, look at the image and decide whether ``label_after_correction``
is what's actually written. A mismatch means the Lev-2 fuzzy match
hijacked the label to a similar-but-wrong word — that crop should be
dropped from the training set (or relabelled).

Usage::

    python scripts/finetune/audit_fuzzy_matches.py
    python scripts/finetune/audit_fuzzy_matches.py --min-edit 2 --max-samples 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"
EXPECTED_DIR = REPO_ROOT / "expected"
DEFAULT_OUT = REPO_ROOT / "colab_release" / "audit_fuzzy"


def audit(
    inputs_dir: Path,
    expected_dir: Path,
    out_dir: Path,
    min_edit: int = 2,
    max_samples: int = 40,
    dpi: int = 300,
    conf_min: float = 0.2,
) -> int:
    import easyocr

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "finetune"))
    from bootstrap_dataset import _best_match, _load_gt_tokens

    pdfs = sorted(inputs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[audit] No PDFs in {inputs_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)

    print("[audit] Loading EasyOCR (ru+en, CPU) …")
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)

    rows: list[tuple[str, str, int, str, str, int]] = []
    # (filename, pdf_stem, page, raw_ocr, label_after_corr, edit_distance)

    for pdf in pdfs:
        gt_tokens = _load_gt_tokens(pdf.stem, inputs_dir, expected_dir)
        if not gt_tokens:
            continue
        print(f"[audit] {pdf.stem}: {len(gt_tokens)} GT tokens")
        doc = fitz.open(str(pdf))
        try:
            for page_idx in range(doc.page_count):
                if len(rows) >= max_samples:
                    break
                pix = doc[page_idx].get_pixmap(
                    dpi=dpi, colorspace=fitz.csGRAY,
                )
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width,
                )
                for box_idx, (bbox, text, conf) in enumerate(
                    reader.readtext(arr, detail=1, paragraph=False)
                ):
                    if conf < conf_min or not text.strip():
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
                    x1, y1 = int(max(xs)), int(max(ys))
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue

                    raw = text.strip()
                    corrected = _best_match(raw, gt_tokens)
                    if corrected is None:
                        continue  # gt_only mode would drop this
                    # Only interested in fuzzy matches at >= min_edit
                    d = Levenshtein.distance(raw.lower(), corrected)
                    if d < min_edit:
                        continue
                    # Save crop
                    filename = (
                        f"{len(rows):03d}_{pdf.stem}_p{page_idx}_"
                        f"b{box_idx}.jpg"
                    )
                    crop = arr[y0:y1, x0:x1]
                    Image.fromarray(crop).save(
                        out_dir / "crops" / filename, quality=92,
                    )
                    rows.append((
                        filename, pdf.stem, page_idx, raw, corrected, d,
                    ))
                    if len(rows) >= max_samples:
                        break
        finally:
            doc.close()
        if len(rows) >= max_samples:
            break

    # Write the TSV with all rows, plus a friendly markdown summary.
    tsv_path = out_dir / "audit.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("filename\tpdf\tpage\traw_ocr\tlabel_after_correction\t"
                "edit_distance\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    # Markdown table — GitHub renders images inline from relative paths,
    # so viewing audit.md in the web UI gives an instant crop + label
    # side-by-side overview.
    md_path = out_dir / "README.md"
    md_lines = [
        "# Fuzzy-match audit",
        "",
        f"Sampled {len(rows)} crop(s) where the bootstrap's fuzzy match "
        f"replaced the raw EasyOCR output with a GT token at Levenshtein "
        f"distance ≥ {min_edit}. For each row, **look at the crop** and "
        f"decide whether `label_after_correction` is actually what's "
        f"printed on the crop:",
        "",
        "- If **yes** — the fuzzy match was correct (OCR misread a "
        "letter, the GT word rescued it). Training on this is fine.",
        "- If **no** — the match hijacked the label to a similar but "
        "wrong word. This crop should be dropped (or relabelled). The "
        "more of these you see, the more false-positive noise in the "
        "training set.",
        "",
        "| # | Crop | Raw OCR | Label | Lev | PDF |",
        "|---|------|---------|-------|-----|-----|",
    ]
    for i, (fn, stem, page, raw, lbl, d) in enumerate(rows):
        md_lines.append(
            f"| {i} | <img src=\"crops/{fn}\" height=\"40\"> | "
            f"`{raw}` | `{lbl}` | {d} | {stem} p{page} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"\n[audit] {len(rows)} Lev-{min_edit}+ fuzzy pair(s) collected")
    print(f"[audit]   TSV:   {tsv_path}")
    print(f"[audit]   MD:    {md_path}")
    print(f"[audit]   Crops: {out_dir / 'crops'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUTS_DIR)
    parser.add_argument("--expected", type=Path, default=EXPECTED_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--min-edit", type=int, default=2,
        help="Only capture pairs with edit distance ≥ this (default: 2)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=40,
        help="Stop after collecting this many pairs (default: 40)",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    return audit(
        args.inputs, args.expected, args.out,
        min_edit=args.min_edit,
        max_samples=args.max_samples,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    sys.exit(main())
