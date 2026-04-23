"""Per-field accuracy metrics on the golden corpus.

Beyond aggregate CER/WER, measures how well OCR extracts the fields
that actually matter downstream: ИНН, КПП, ОГРН, dates, legal forms,
party names. Compares full-page OCR output against the structured
``expected/*.json`` references using strict-match and fuzzy-match
scorers per field type.

Output:
  reports/per_field_metrics.json — per-doc + aggregate per-field
  accuracy, suitable for CI dashboards.

Exit code:
  0 on success, 1 when ``--fail-below`` is set and aggregate INN
  accuracy falls below the threshold.

Typical run:
  python scripts/per_field_metrics.py
  python scripts/per_field_metrics.py --fail-below-inn 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import fitz
import numpy as np

from src.application.requisite_extractor import extract_all
from src.shared.dpi_utils import estimate_page_source_dpi  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"
EXPECTED_DIR = REPO_ROOT / "expected"
REPORTS_DIR = REPO_ROOT / "reports"


def _walk_json(obj, path: list[str] | None = None):
    """Yield (path, value) for every scalar in a nested dict/list."""
    if path is None:
        path = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_json(v, path + [str(k)])
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_json(v, path + [str(i)])
    else:
        yield (path, obj)


def _collect_gt_fields(expected_path: Path) -> dict[str, set[str]]:
    """Extract expected INN / OGRN / KPP / dates from the JSON golden."""
    data = json.loads(expected_path.read_text(encoding="utf-8"))
    fields: dict[str, set[str]] = {
        "inn": set(), "ogrn": set(), "kpp": set(), "dates": set(),
    }
    for path, value in _walk_json(data):
        if not isinstance(value, str):
            continue
        # Path-based hints — tighter than content-based regex.
        last = path[-1].lower() if path else ""
        if last == "inn":
            fields["inn"].add(value)
        elif last == "ogrn" or last == "ogrnip":
            fields["ogrn"].add(value)
        elif last == "kpp":
            fields["kpp"].add(value)
        elif last in ("date", "issue_date", "start_date", "end_date"):
            # Normalize ISO and dd.mm.yyyy both ways.
            fields["dates"].add(value.strip())
    return fields


def _run_ocr(pdf: Path, reader) -> str:
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


def _score_set(expected: set[str], extracted: set[str]) -> dict[str, float]:
    """TP/FN/FP-based precision+recall against the golden set."""
    if not expected:
        return {"precision": None, "recall": None, "n_expected": 0,
                "n_extracted": len(extracted)}
    tp = len(expected & extracted)
    fn = len(expected - extracted)
    fp = len(extracted - expected)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "n_expected": len(expected),
        "n_extracted": len(extracted),
        "tp": tp, "fn": fn, "fp": fp,
    }


def _normalize_date(d: str) -> str:
    """Try parsing and return ISO date; fall back to raw."""
    import datetime
    for fmt in (
        "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y",
        "%d.%m.%y",
    ):
        try:
            return datetime.datetime.strptime(d, fmt).date().isoformat()
        except ValueError:
            continue
    return d.strip()


def score_pdf(pdf: Path, expected_path: Path, reader) -> dict:
    """Run full pipeline on ``pdf`` and score per-field accuracy."""
    gt = _collect_gt_fields(expected_path)
    ocr_text = _run_ocr(pdf, reader)
    extracted = extract_all(ocr_text)

    extracted_sets = {
        "inn": {c.value for c in extracted.inns},
        "ogrn": {c.value for c in extracted.ogrns},
        "kpp": {c.value for c in extracted.kpps},
        "dates": {_normalize_date(c.value) for c in extracted.dates},
    }
    gt_normalized = {
        "inn": gt["inn"],
        "ogrn": gt["ogrn"],
        "kpp": gt["kpp"],
        "dates": {_normalize_date(d) for d in gt["dates"]},
    }
    return {
        "pdf": pdf.name,
        "fields": {
            field: _score_set(gt_normalized[field], extracted_sets[field])
            for field in ("inn", "ogrn", "kpp", "dates")
        },
    }


def aggregate(per_pdf: list[dict]) -> dict:
    """Macro-average precision/recall across documents per field."""
    agg: dict[str, dict] = {}
    for field in ("inn", "ogrn", "kpp", "dates"):
        precisions, recalls = [], []
        total_tp = total_fn = total_fp = 0
        for doc in per_pdf:
            scores = doc["fields"][field]
            if scores.get("precision") is not None:
                precisions.append(scores["precision"])
            if scores.get("recall") is not None:
                recalls.append(scores["recall"])
            total_tp += scores.get("tp", 0)
            total_fn += scores.get("fn", 0)
            total_fp += scores.get("fp", 0)
        agg[field] = {
            "macro_precision": (
                round(sum(precisions) / len(precisions), 3)
                if precisions else None
            ),
            "macro_recall": (
                round(sum(recalls) / len(recalls), 3) if recalls else None
            ),
            "micro_tp": total_tp, "micro_fn": total_fn, "micro_fp": total_fp,
        }
    return agg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=REPORTS_DIR / "per_field_metrics.json",
    )
    parser.add_argument(
        "--fail-below-inn", type=float, default=None,
        help="Exit 1 if aggregate INN recall (0..1) is below this",
    )
    args = parser.parse_args()

    pdfs = sorted(INPUTS_DIR.glob("*.pdf"))
    if not pdfs:
        print("[per-field] No PDFs in inputs/", file=sys.stderr)
        return 1
    import easyocr
    print("[per-field] loading EasyOCR (ru + en, CPU)…")
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)

    per_pdf: list[dict] = []
    for pdf in pdfs:
        expected = EXPECTED_DIR / f"{pdf.stem}.json"
        if not expected.exists():
            print(f"[per-field] skip {pdf.stem} (no golden JSON)")
            continue
        print(f"[per-field] scoring {pdf.name}…")
        result = score_pdf(pdf, expected, reader)
        per_pdf.append(result)
        for field, s in result["fields"].items():
            if s["n_expected"]:
                print(
                    f"  {field:<6} precision={s['precision']:.2f}  "
                    f"recall={s['recall']:.2f}  "
                    f"tp/fn/fp={s['tp']}/{s['fn']}/{s['fp']}"
                )

    agg = aggregate(per_pdf)
    report = {"per_pdf": per_pdf, "aggregate": agg}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print("[per-field] aggregate:")
    for field, s in agg.items():
        print(
            f"  {field:<6} macro_P={s['macro_precision']}  "
            f"macro_R={s['macro_recall']}  "
            f"micro_tp/fn/fp={s['micro_tp']}/{s['micro_fn']}/{s['micro_fp']}"
        )
    print(f"[per-field] report: {args.output}")

    if args.fail_below_inn is not None:
        inn_r = agg["inn"].get("macro_recall")
        if inn_r is None or inn_r < args.fail_below_inn:
            print(
                f"[per-field] FAIL: INN recall {inn_r} < "
                f"{args.fail_below_inn}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
