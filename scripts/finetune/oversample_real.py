"""Duplicate real-crop rows in labels.csv so they weigh more in the
training loss against the large synth-crop pool.

Problem:
    Our dataset mixes ``bootstrap_dataset.py`` output (real scan
    crops, ~2500) with ``synth_data.py`` output (rendered crops,
    20 000). If every sample gets equal loss weight, the model
    spends 89% of its updates on synthetic pixels and may fail to
    generalise to real scans — the "synth-biased" failure mode.

Fix (this script):
    Append extra rows of each real-crop (filename, label) tuple to
    ``labels.csv`` so that each real jpg gets seen ``--multiplier``
    times per epoch. Crop files are not duplicated on disk — the
    DataLoader's ``OCRDataset.__getitem__`` resolves each row's
    filename independently, so ``N`` rows all pointing at
    ``img_00000042.jpg`` makes the model see that image N times
    per epoch.

Usage::

    # Triple-weight real crops (real becomes 27% → 43% of the mix)
    python scripts/finetune/oversample_real.py --multiplier 3

    # Go harder
    python scripts/finetune/oversample_real.py --multiplier 5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "finetune_ru"

# Real-crop filenames follow ``img_{1..999999}.jpg`` (bootstrap indices
# start at 1). Synth crops start at ``img_01000000.jpg`` (per
# ``synth_data.py`` ``start_index``). Anything below 1_000_000 is real.
REAL_INDEX_CUTOFF = 1_000_000


def _is_real_filename(fn: str) -> bool:
    """``img_{NNNNNNNN}.jpg`` → True if N < 1e6."""
    stem = Path(fn).stem
    if not stem.startswith("img_"):
        return False
    try:
        idx = int(stem[4:])
    except ValueError:
        return False
    return idx < REAL_INDEX_CUTOFF


def oversample_split(csv_path: Path, multiplier: int) -> tuple[int, int]:
    """Append ``multiplier - 1`` extra copies of each real row to
    ``csv_path`` in place. Returns (rows_before, rows_after).
    """
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        rows = list(reader)
    before = len(rows)
    real_rows = [r for r in rows if _is_real_filename(r[0])]
    # Append multiplier-1 extra copies (the original row already counts once).
    with csv_path.open("a", encoding="utf-8") as f:
        # Preserve the quoting pattern bootstrap_dataset uses — it writes
        # ``filename,"label"`` so we emulate that.
        for _ in range(multiplier - 1):
            for fn, label in real_rows:
                safe = label.replace('"', '""')
                f.write(f'{fn},"{safe}"\n')
    after = before + len(real_rows) * (multiplier - 1)
    return before, after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help="Dataset root — expects training/ + validation/ subdirs "
             "with labels.csv in each (default: %(default)s)",
    )
    parser.add_argument(
        "--multiplier", type=int, default=3,
        help="How many total copies of each real row (including the "
             "original). 3 = original + 2 duplicates (default).",
    )
    args = parser.parse_args()
    if args.multiplier < 2:
        print(f"[oversample] multiplier={args.multiplier} is a no-op; "
              "use ≥ 2 to actually duplicate rows.")
        return 1

    for split in ("training", "validation"):
        csv_path = args.dataset / split / "labels.csv"
        if not csv_path.exists():
            print(f"[oversample] skip {split}: {csv_path} not found")
            continue
        before, after = oversample_split(csv_path, args.multiplier)
        print(
            f"[oversample] {split}: {before} → {after} rows "
            f"(+{after - before} duplicates)",
        )
    print(f"[oversample] multiplier={args.multiplier} applied in-place.")
    print(
        "[oversample] Note: this only modifies labels.csv. jpgs are not "
        "physically duplicated. To revert, re-run bootstrap_dataset.py."
    )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
