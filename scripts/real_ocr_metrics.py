"""Real OCR accuracy metrics on the golden corpus.

Runs EasyOCR on every PDF in ``inputs/*.pdf``, computes CER / WER
against ``inputs/*.txt`` ground truth, and reports per-document +
aggregate confidence statistics. Writes a JSON report for CI artifact
upload. Exits non-zero when the average CER crosses ``--fail-above``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import fitz
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"
REPORTS_DIR = REPO_ROOT / "reports"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower().strip()


def _load_gt(stem: str) -> str:
    txt = (INPUTS_DIR / f"{stem}.txt").read_text(encoding="utf-8")
    lines = [
        line for line in txt.splitlines()
        if not re.fullmatch(r"[=\-]{10,}", line.strip())
        and not line.strip().startswith(("ФАЙЛ:", "СТРАНИЦ:"))
    ]
    return _normalize("\n".join(lines))


def _ocr_pdf(pdf: Path, reader) -> tuple[str, float]:
    doc = fitz.open(str(pdf))
    try:
        all_text: list[str] = []
        all_confs: list[float] = []
        for i in range(doc.page_count):
            pix = doc[i].get_pixmap(dpi=300, colorspace=fitz.csGRAY)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width,
            )
            for _bbox, text, conf in reader.readtext(
                arr, detail=1, paragraph=False,
            ):
                if conf < 0.1 or not text.strip():
                    continue
                all_text.append(text)
                all_confs.append(float(conf))
        mean = sum(all_confs) / len(all_confs) if all_confs else 0.0
        return _normalize("\n".join(all_text)), mean
    finally:
        doc.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fail-above", type=float, default=None,
        help="Exit 1 if mean CER (%%) exceeds this threshold",
    )
    args = parser.parse_args()

    pdfs = sorted(INPUTS_DIR.glob("*.pdf"))
    if not pdfs:
        print("[metrics] no PDFs in inputs/", file=sys.stderr)
        return 1

    from jiwer import cer, wer
    import easyocr
    print("[metrics] loading EasyOCR reader (ru + en, CPU)…")
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)

    rows: list[dict] = []
    for pdf in pdfs:
        txt = pdf.with_suffix(".txt")
        if not txt.exists():
            continue
        gt = _load_gt(pdf.stem)
        t0 = time.time()
        hyp, mean_conf = _ocr_pdf(pdf, reader)
        elapsed = time.time() - t0
        row = {
            "document": pdf.name,
            "pages": fitz.open(str(pdf)).page_count,
            "cer": round(cer(gt, hyp), 4),
            "wer": round(wer(gt, hyp), 4),
            "mean_confidence": round(mean_conf * 100.0, 2),
            "elapsed_sec": round(elapsed, 1),
            "chars_ocr": len(hyp),
            "chars_gt": len(gt),
        }
        rows.append(row)
        print(
            f"  {pdf.name:<40} CER={row['cer']*100:>6.1f}%  "
            f"WER={row['wer']*100:>6.1f}%  "
            f"conf={row['mean_confidence']:>5.1f}%  t={elapsed:>5.1f}s"
        )

    avg_cer = sum(r["cer"] for r in rows) / len(rows) if rows else 0.0
    avg_wer = sum(r["wer"] for r in rows) / len(rows) if rows else 0.0
    avg_conf = (
        sum(r["mean_confidence"] for r in rows) / len(rows) if rows else 0.0
    )
    print(
        f"  {'AVERAGE':<40} CER={avg_cer*100:>6.1f}%  "
        f"WER={avg_wer*100:>6.1f}%  conf={avg_conf:>5.1f}%"
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    report = {
        "engine": "easyocr",
        "average": {"cer": avg_cer, "wer": avg_wer, "mean_confidence": avg_conf},
        "documents": rows,
    }
    (REPORTS_DIR / "ocr_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[metrics] Report: {REPORTS_DIR / 'ocr_metrics.json'}")

    if args.fail_above is not None and avg_cer * 100.0 > args.fail_above:
        print(
            f"[metrics] FAIL: avg CER {avg_cer*100:.1f}% > "
            f"{args.fail_above}%", file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
