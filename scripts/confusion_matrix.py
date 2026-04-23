"""Character-level OCR confusion matrix analyzer.

Runs OCR over ``inputs/*.pdf``, aligns each output to the corresponding
``inputs/*.txt`` ground truth via Levenshtein alignment, and tallies
character-level substitutions. The top confusions reveal systematic
weaknesses — e.g. ``н↔и``, ``3↔З``, ``о↔0`` — that can be targeted with
synth-data augmentation or character-whitelist adjustments.

Output:
  reports/confusion_matrix.json — the top-K confusions per (source →
  predicted) pair, with counts and example contexts.
  reports/confusion_matrix.txt — human-readable summary.

Use case:
  1. Run this before/after fine-tune to see which confusions were
     reduced and which remain.
  2. Feed top confusions as hint to ``scripts/finetune/synth_data.py``
     (future extension) to generate targeted training pairs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz
import numpy as np
from rapidfuzz.distance import Levenshtein

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"
REPORTS_DIR = REPO_ROOT / "reports"


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace — for alignment."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _align_and_diff(ref: str, hyp: str) -> list[tuple[str, str]]:
    """Return list of (ref_char, hyp_char) substitution pairs from
    the Levenshtein edit path between ``ref`` and ``hyp``.

    Uses ``rapidfuzz.distance.Levenshtein.editops`` to get the actual
    edits. Only substitutions contribute to the confusion matrix;
    insertions and deletions are dropped (they indicate segmentation
    issues, not character confusion).
    """
    ops = Levenshtein.editops(ref, hyp)
    pairs: list[tuple[str, str]] = []
    for op in ops:
        # op.tag is "replace" / "insert" / "delete"
        if op.tag == "replace":
            pairs.append((ref[op.src_pos], hyp[op.dest_pos]))
    return pairs


def _ocr_pdf(pdf: Path, reader) -> str:
    doc = fitz.open(str(pdf))
    try:
        chunks: list[str] = []
        for i in range(doc.page_count):
            pix = doc[i].get_pixmap(dpi=300, colorspace=fitz.csGRAY)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width,
            )
            for _bbox, text, _conf in reader.readtext(
                arr, detail=1, paragraph=False,
            ):
                chunks.append(text)
        return "\n".join(chunks)
    finally:
        doc.close()


def analyze(inputs_dir: Path) -> tuple[Counter, list[tuple[str, int]]]:
    """Return (confusion_counter, per-document-error-counts)."""
    import easyocr
    print("[confusion] loading EasyOCR (ru + en, CPU)…")
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)

    confusion: Counter[tuple[str, str]] = Counter()
    per_doc: list[tuple[str, int]] = []
    for pdf in sorted(inputs_dir.glob("*.pdf")):
        txt = pdf.with_suffix(".txt")
        if not txt.exists():
            continue
        ref = _normalise(txt.read_text(encoding="utf-8"))
        print(f"[confusion] OCR {pdf.name}…")
        hyp = _normalise(_ocr_pdf(pdf, reader))
        pairs = _align_and_diff(ref, hyp)
        for pair in pairs:
            # Only keep character-printable substitutions.
            if any(c.isspace() for c in pair):
                continue
            confusion[pair] += 1
        per_doc.append((pdf.name, len(pairs)))
        print(f"  {len(pairs)} substitutions")
    return confusion, per_doc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=INPUTS_DIR)
    parser.add_argument(
        "--output-json", type=Path,
        default=REPORTS_DIR / "confusion_matrix.json",
    )
    parser.add_argument(
        "--output-txt", type=Path,
        default=REPORTS_DIR / "confusion_matrix.txt",
    )
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    confusion, per_doc = analyze(args.inputs)
    total = sum(confusion.values())
    top = confusion.most_common(args.top_k)

    report = {
        "total_substitutions": total,
        "per_document": [{"pdf": n, "substitutions": s} for n, s in per_doc],
        "top_confusions": [
            {"source": s, "predicted": p, "count": c,
             "share_pct": round(c * 100 / total, 2) if total else 0.0}
            for (s, p), c in top
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"Confusion matrix — total {total} substitutions across "
        f"{len(per_doc)} documents",
        "",
        "source → predicted  count  share",
        "-" * 40,
    ]
    for (s, p), c in top:
        share = c * 100 / total if total else 0.0
        lines.append(f"  {s!r:6s} → {p!r:6s}  {c:5d}  {share:5.1f}%")
    args.output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"[confusion] {total} total substitutions")
    print(f"[confusion] top {min(10, len(top))}:")
    for (s, p), c in top[:10]:
        print(f"  {s!r} → {p!r} : {c}")
    print(f"[confusion] report: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
