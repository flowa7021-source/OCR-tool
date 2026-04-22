"""Convert the bootstrap's ``gt.txt`` (filename TAB label) to EasyOCR
trainer's ``labels.csv`` (filename,"label") in place.

Newer runs of ``bootstrap_dataset.py`` already write both formats. Use
this script to back-fill ``labels.csv`` for datasets produced by older
bootstrap versions — without re-running EasyOCR on every PDF page.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def convert(gt_path: Path) -> int:
    """Emit ``labels.csv`` next to ``gt_path``. Returns line count."""
    out_path = gt_path.parent / "labels.csv"
    lines = gt_path.read_text(encoding="utf-8").splitlines()
    rows: list[str] = ["filename,words"]
    for line in lines:
        if not line.strip():
            continue
        try:
            filename, label = line.split("\t", 1)
        except ValueError:
            continue
        safe_label = label.replace('"', '""')
        rows.append(f'{filename},"{safe_label}"')
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows) - 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent
        / "datasets" / "finetune_ru",
    )
    args = parser.parse_args()
    for subset in ("training", "validation"):
        gt = args.dataset / subset / "gt.txt"
        if not gt.exists():
            print(f"skip {subset}: {gt} not found")
            continue
        n = convert(gt)
        print(f"{subset}: {n} rows -> {gt.parent / 'labels.csv'}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
