"""Build a EasyOCR recognizer training dataset from inputs/*.pdf + *.txt.

Workflow:
    1. Rasterize every page of every PDF in ``inputs/`` at 300 DPI.
    2. Run the current EasyOCR reader to get word-level bounding boxes.
    3. For each box, crop the image and save as ``<id>.jpg``.
    4. Auto-correct the label: find the nearest word in the human
       ground-truth ``inputs/<stem>.txt`` within Levenshtein distance 1
       (short) or 2 (long); use the GT token when it matches.
    5. Emit ``gt.txt`` in deep-text-recognition-benchmark format:
       one line per crop, ``<filename>\t<label>``.
    6. 80/20 train/val split by hash of (pdf_stem, page, idx) so
       reruns are deterministic.

Output layout under ``datasets/finetune_ru/``::

    training/
        img_00000001.jpg
        ...
        gt.txt
    validation/
        img_00001001.jpg
        ...
        gt.txt

Downstream: ``finetune_recognizer.py`` reads this layout directly.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"
DEFAULT_OUT = REPO_ROOT / "datasets" / "finetune_ru"
MIN_CROP_H = 8
MIN_CROP_W = 8


def _load_gt_tokens(stem: str, inputs_dir: Path = INPUTS_DIR) -> set[str]:
    """Return the set of word-like tokens from the ground-truth transcript."""
    txt = inputs_dir / f"{stem}.txt"
    if not txt.exists():
        return set()
    raw = txt.read_text(encoding="utf-8")
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-./,]{0,}", raw)
    return {t.strip(".,").lower() for t in tokens if len(t) >= 2}


def _best_match(word: str, gt_tokens: set[str]) -> str | None:
    """Return the nearest GT token within a length-adaptive edit budget."""
    lw = word.lower()
    if lw in gt_tokens:
        return lw
    budget = 1 if len(lw) < 6 else 2
    best: tuple[int, str] | None = None
    for t in gt_tokens:
        if abs(len(t) - len(lw)) > budget:
            continue
        d = Levenshtein.distance(lw, t, score_cutoff=budget)
        if d <= budget and (best is None or d < best[0]):
            best = (d, t)
    return best[1] if best else None


def _split_bucket(key: str) -> str:
    """Deterministic 80/20 split by SHA1 of the key."""
    h = hashlib.sha1(key.encode()).hexdigest()
    return "validation" if int(h[:4], 16) % 5 == 0 else "training"


def build_dataset(inputs_dir: Path, out_dir: Path, dpi: int = 300) -> int:
    import easyocr

    pdfs = sorted(inputs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[bootstrap] No PDFs in {inputs_dir}", file=sys.stderr)
        return 1

    (out_dir / "training").mkdir(parents=True, exist_ok=True)
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)
    train_gt = (out_dir / "training" / "gt.txt").open("w", encoding="utf-8")
    val_gt = (out_dir / "validation" / "gt.txt").open("w", encoding="utf-8")

    print("[bootstrap] Loading EasyOCR reader (ru + en, CPU)…")
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    total = auto_corrected = 0

    for pdf in pdfs:
        gt_tokens = _load_gt_tokens(pdf.stem, inputs_dir)
        doc = fitz.open(str(pdf))
        try:
            for page_idx in range(doc.page_count):
                pix = doc[page_idx].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width,
                )
                for box_idx, (bbox, text, conf) in enumerate(
                    reader.readtext(arr, detail=1, paragraph=False)
                ):
                    if conf < 0.2 or not text.strip():
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
                    x1, y1 = int(max(xs)), int(max(ys))
                    if x1 - x0 < MIN_CROP_W or y1 - y0 < MIN_CROP_H:
                        continue

                    label = text.strip()
                    corrected = _best_match(label, gt_tokens) if gt_tokens else None
                    if corrected and corrected != label:
                        label = corrected
                        auto_corrected += 1

                    key = f"{pdf.stem}:{page_idx}:{box_idx}"
                    bucket = _split_bucket(key)
                    total += 1
                    filename = f"img_{total:08d}.jpg"
                    crop = arr[y0:y1, x0:x1]
                    Image.fromarray(crop).save(
                        out_dir / bucket / filename, quality=92,
                    )
                    target = train_gt if bucket == "training" else val_gt
                    target.write(f"{filename}\t{label}\n")
        finally:
            doc.close()

    train_gt.close()
    val_gt.close()
    print(
        f"[bootstrap] Done: {total} crops written to {out_dir} "
        f"({auto_corrected} labels auto-corrected from ground truth)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    return build_dataset(args.inputs, args.out, args.dpi)


if __name__ == "__main__":
    sys.exit(main())
