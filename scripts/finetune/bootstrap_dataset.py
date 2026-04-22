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
import json
import re
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"
EXPECTED_DIR = REPO_ROOT / "expected"
DEFAULT_OUT = REPO_ROOT / "datasets" / "finetune_ru"
MIN_CROP_H = 8
MIN_CROP_W = 8

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-./,]{0,}")


def _tokenize(text: str) -> set[str]:
    """Extract lowercase word-like tokens (letters/digits, len ≥ 2)."""
    return {t.strip(".,").lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2}


def _walk_json_strings(obj) -> list[str]:
    """Flatten every string leaf in a (nested) JSON-like structure."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _walk_json_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _walk_json_strings(v)]
    return []


def _load_gt_tokens(
    stem: str,
    inputs_dir: Path = INPUTS_DIR,
    expected_dir: Path = EXPECTED_DIR,
) -> set[str]:
    """Combine tokens from ``inputs/<stem>.txt`` (plain transcript) and
    ``expected/<stem>.json`` (structured reference: INN, OGRN, party names,
    addresses, document numbers, phones, …). The JSON adds domain-specific
    vocabulary that the plain transcript may truncate or miss.
    """
    tokens: set[str] = set()
    txt = inputs_dir / f"{stem}.txt"
    if txt.exists():
        tokens |= _tokenize(txt.read_text(encoding="utf-8"))
    json_path = expected_dir / f"{stem}.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[bootstrap] WARN: {json_path.name} is not valid JSON: {e}",
                  file=sys.stderr)
        else:
            for s in _walk_json_strings(data):
                tokens |= _tokenize(s)
    return tokens


_NUMERIC_RE = re.compile(r"^[0-9][0-9.,\-/]*$")


def _best_match(word: str, gt_tokens: set[str]) -> str | None:
    """Return the nearest GT token.

    For TEXT tokens — length-adaptive Levenshtein budget (1 for short,
    2 for long). For NUMERIC tokens (ИНН, даты, суммы, номера) — strict
    exact-match only: fuzzy matching on numbers is the main source of
    false-positive label noise (e.g. ``22,391`` → ``2239``, ``95ЗЗ/Б``
    → ``9507/б`` from the audit) and the postprocess pipeline's
    checksum validators (:mod:`src.shared.requisite_validators`) will
    catch real requisite OCR errors at inference time anyway.
    """
    lw = word.lower()
    if lw in gt_tokens:
        return lw
    if _NUMERIC_RE.match(lw):
        return None
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


def build_dataset(
    inputs_dir: Path,
    out_dir: Path,
    dpi: int = 300,
    gt_only: bool = True,
    min_conf: float = 0.2,
) -> int:
    import easyocr

    pdfs = sorted(inputs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[bootstrap] No PDFs in {inputs_dir}", file=sys.stderr)
        return 1

    (out_dir / "training").mkdir(parents=True, exist_ok=True)
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)
    # EasyOCR trainer expects ``labels.csv`` with header ``filename,words``
    # (pandas.read_csv), so that's the primary format. ``gt.txt`` is also
    # written for human inspection and for tools that follow the
    # deep-text-recognition-benchmark convention.
    train_gt = (out_dir / "training" / "gt.txt").open("w", encoding="utf-8")
    val_gt = (out_dir / "validation" / "gt.txt").open("w", encoding="utf-8")
    train_csv = (out_dir / "training" / "labels.csv").open("w", encoding="utf-8")
    val_csv = (out_dir / "validation" / "labels.csv").open("w", encoding="utf-8")
    train_csv.write("filename,words\n")
    val_csv.write("filename,words\n")

    print("[bootstrap] Loading EasyOCR reader (ru + en, CPU)…")
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    n_total = n_kept = n_dropped_lowconf = n_dropped_nogt = 0
    n_exact = n_fuzzy = 0

    for pdf in pdfs:
        gt_tokens = _load_gt_tokens(pdf.stem, inputs_dir)
        if not gt_tokens:
            print(f"[bootstrap] WARN: no GT tokens for {pdf.stem} "
                  f"(check inputs/{pdf.stem}.txt and expected/{pdf.stem}.json)",
                  file=sys.stderr)
        print(f"[bootstrap] {pdf.stem}: {len(gt_tokens)} GT tokens")
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
                    n_total += 1
                    if conf < min_conf or not text.strip():
                        n_dropped_lowconf += 1
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
                    x1, y1 = int(max(xs)), int(max(ys))
                    if x1 - x0 < MIN_CROP_W or y1 - y0 < MIN_CROP_H:
                        continue

                    raw_label = text.strip()
                    corrected = (
                        _best_match(raw_label, gt_tokens) if gt_tokens else None
                    )
                    if gt_only and corrected is None:
                        # No GT token within Levenshtein budget → likely
                        # a misread; strict mode drops these.
                        n_dropped_nogt += 1
                        continue
                    if corrected is None:
                        label = raw_label
                    else:
                        label = corrected
                        if corrected.lower() == raw_label.lower():
                            n_exact += 1
                        else:
                            n_fuzzy += 1

                    key = f"{pdf.stem}:{page_idx}:{box_idx}"
                    bucket = _split_bucket(key)
                    n_kept += 1
                    filename = f"img_{n_kept:08d}.jpg"
                    crop = arr[y0:y1, x0:x1]
                    Image.fromarray(crop).save(
                        out_dir / bucket / filename, quality=92,
                    )
                    target = train_gt if bucket == "training" else val_gt
                    target.write(f"{filename}\t{label}\n")
                    csv_target = train_csv if bucket == "training" else val_csv
                    # Escape embedded commas / quotes for CSV round-trip.
                    safe_label = label.replace('"', '""')
                    csv_target.write(f'{filename},"{safe_label}"\n')
        finally:
            doc.close()

    train_gt.close()
    val_gt.close()
    train_csv.close()
    val_csv.close()
    print(
        f"[bootstrap] Done. "
        f"seen={n_total}  kept={n_kept}  "
        f"dropped_lowconf={n_dropped_lowconf}  "
        f"dropped_nogt={n_dropped_nogt}  "
        f"exact_match={n_exact}  fuzzy_match={n_fuzzy}  "
        f"(gt_only={gt_only}, min_conf={min_conf})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUTS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--keep-unverified", action="store_true",
        help="Keep crops whose label has no GT match (old, lenient behaviour). "
             "Default is strict: drop labels that aren't in inputs/*.txt or "
             "expected/*.json within a Levenshtein-2 budget.",
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.2,
        help="Drop EasyOCR boxes below this confidence (default: %(default)s).",
    )
    args = parser.parse_args()
    return build_dataset(
        args.inputs, args.out, args.dpi,
        gt_only=not args.keep_unverified,
        min_conf=args.min_conf,
    )


if __name__ == "__main__":
    sys.exit(main())
