"""One-command local fine-tune runbook.

Runs the full pipeline: preflight checks → baseline metrics on stock
weights → fine-tune → post-train metrics → comparison report →
optional auto-swap.

This is the entrypoint for the "result on first try" workflow:
single command, single wait, objective before/after numbers.

Usage::

    # Full run on RTX 2060 S / 3060 / 4060 (8 GB)
    python scripts/finetune/run_local.py

    # Skip the baseline step if you just measured stock accuracy
    python scripts/finetune/run_local.py --skip-baseline

    # Auto-keep new weights if average CER improves by ≥ 5 pp
    python scripts/finetune/run_local.py --auto-apply-if-better 5.0

    # Dry run — check preconditions only
    python scripts/finetune/run_local.py --preflight-only

Expected wall-clock on RTX 2060 Super:
  preflight               few seconds
  baseline metrics        ~2-5 min (OCR on 5 PDFs at 300 DPI)
  fine-tune (30 epochs)   ~30-45 min
  post-train metrics      ~2-5 min
  -----------------------
  total                   ~35-55 min
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"
DATASET_DIR = REPO_ROOT / "datasets" / "finetune_ru"
MODEL_DIR = REPO_ROOT / "resources" / "easyocr_models"
BASE_WEIGHTS = MODEL_DIR / "cyrillic_g2.pth"
STOCK_BACKUP = MODEL_DIR / "cyrillic_g2.pth.stock"
REPORTS_DIR = REPO_ROOT / "reports" / "finetune_run"


# ── Preflight ──────────────────────────────────────────────────────────────

class PreflightError(RuntimeError):
    """Raised when a required precondition isn't met."""


# Backward-compat alias — the older name was shipped in an earlier draft.
PreflightFailure = PreflightError


def _check_torch_cuda() -> dict:
    import torch
    avail = torch.cuda.is_available()
    if not avail:
        raise PreflightError(
            "CUDA not available. Install PyTorch with CUDA:\n"
            "  pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu121"
        )
    dev = torch.cuda.get_device_properties(0)
    vram_gb = dev.total_memory / 1024**3
    return {
        "cuda_available": True,
        "device_name": dev.name,
        "vram_gb": round(vram_gb, 1),
        "compute_capability": f"{dev.major}.{dev.minor}",
        "torch_version": torch.__version__,
    }


def _check_deps() -> dict:
    """Verify trainer deps present; give actionable install hint on missing."""
    missing = []
    for pkg, import_name in [
        ("pandas", "pandas"),
        ("natsort", "natsort"),
        ("lmdb", "lmdb"),
        ("fire", "fire"),
        ("nltk", "nltk"),
        ("Pillow", "PIL"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise PreflightError(
            f"Missing Python packages: {missing}\n"
            f"Install with: pip install {' '.join(missing)}"
        )
    return {"trainer_deps": "ok"}


def _check_dataset() -> dict:
    """Expect ``datasets/finetune_ru/{training,validation}/labels.csv``."""
    missing = []
    for split in ("training", "validation"):
        csv = DATASET_DIR / split / "labels.csv"
        if not csv.exists():
            missing.append(str(csv))
    if missing:
        zip_path = REPO_ROOT / "colab_release" / "finetune_ru_colab.zip"
        hint = "Run: python scripts/finetune/bootstrap_dataset.py"
        if zip_path.exists():
            hint = (
                "Run: python -c \"import zipfile; "
                f"zipfile.ZipFile('{zip_path}').extractall('datasets')\""
                "\n        (faster — uses pre-built ZIP from the repo)"
            )
        raise PreflightError(
            "Dataset missing:\n  "
            + "\n  ".join(missing)
            + f"\n{hint}"
        )
    stats = {}
    for split in ("training", "validation"):
        csv = DATASET_DIR / split / "labels.csv"
        n_rows = sum(1 for _ in csv.open(encoding="utf-8")) - 1
        n_jpg = len(list((DATASET_DIR / split).glob("*.jpg")))
        stats[f"{split}_rows"] = n_rows
        stats[f"{split}_jpgs"] = n_jpg
    return stats


def _check_weights() -> dict:
    if not BASE_WEIGHTS.exists():
        raise PreflightError(
            f"Stock weights missing at {BASE_WEIGHTS}.\n"
            "Run: python scripts/prefetch_easyocr_models.py"
        )
    return {
        "weights_size_mb": round(BASE_WEIGHTS.stat().st_size / 1e6, 1),
    }


def _check_disk_space() -> dict:
    """Training needs ~500 MB for trainer clone + checkpoints."""
    total, used, free = shutil.disk_usage(REPO_ROOT)
    free_gb = free / 1024**3
    if free_gb < 2.0:
        raise PreflightError(
            f"Low disk space: {free_gb:.1f} GB free, need ≥ 2 GB."
        )
    return {"disk_free_gb": round(free_gb, 1)}


def preflight() -> dict:
    """Run all checks; raise :class:`PreflightFailure` on the first miss."""
    report = {}
    print("[preflight] Checking CUDA …")
    report.update(_check_torch_cuda())
    print(f"  GPU: {report['device_name']} "
          f"({report['vram_gb']} GB VRAM, CC {report['compute_capability']})")
    print(f"  PyTorch: {report['torch_version']}")

    print("[preflight] Checking Python deps …")
    _check_deps()

    print("[preflight] Checking dataset …")
    report.update(_check_dataset())
    print(f"  training: {report['training_rows']} rows, "
          f"{report['training_jpgs']} jpgs")
    print(f"  validation: {report['validation_rows']} rows, "
          f"{report['validation_jpgs']} jpgs")

    print("[preflight] Checking base weights …")
    report.update(_check_weights())

    print("[preflight] Checking disk space …")
    report.update(_check_disk_space())
    print(f"  free: {report['disk_free_gb']} GB")

    print("[preflight] ✓ all checks passed")
    return report


# ── Metrics ────────────────────────────────────────────────────────────────

def run_metrics(label: str, out_json: Path) -> dict:
    """Invoke ``real_ocr_metrics.py`` and capture the report.

    The script uses whatever ``cyrillic_g2.pth`` is currently in the
    EasyOCR cache directory. Before-and-after comparison depends on the
    caller swapping weights into the cache between calls — see
    :func:`sync_weights_to_cache`.
    """
    print(f"\n[metrics-{label}] Running real_ocr_metrics on {INPUTS_DIR} …")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(
        [sys.executable, str(REPO_ROOT / "scripts" / "real_ocr_metrics.py"),
         "--output", str(out_json)],
        cwd=str(REPO_ROOT),
    )
    if rc != 0:
        print(f"[metrics-{label}] metrics script exited with {rc}",
              file=sys.stderr)
        return {}
    if not out_json.exists():
        return {}
    return json.loads(out_json.read_text(encoding="utf-8"))


def sync_weights_to_cache(weights_path: Path) -> Path | None:
    """Copy ``weights_path`` into the EasyOCR model cache so that
    ``easyocr.Reader(['ru','en'])`` loads our chosen file. Returns the
    cache path for logging (or None if cache dir can't be located).
    """
    try:
        import easyocr
    except ImportError:
        return None
    # Lightweight init just to resolve the cache directory.
    reader = easyocr.Reader(
        ["ru", "en"], gpu=False, verbose=False, download_enabled=False,
    )
    cache_dir = Path(reader.model_storage_directory)
    target = cache_dir / "cyrillic_g2.pth"
    shutil.copy2(weights_path, target)
    return target


# ── Report ─────────────────────────────────────────────────────────────────

def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    return f"{v * 100:5.1f}%"


def summarise(before: dict, after: dict) -> str:
    """Render a side-by-side comparison of before/after metrics."""
    if not before or not after:
        return "[report] missing metrics — cannot compare"
    b_cer = before.get("avg_cer")
    a_cer = after.get("avg_cer")
    b_wer = before.get("avg_wer")
    a_wer = after.get("avg_wer")
    b_conf = before.get("avg_conf")
    a_conf = after.get("avg_conf")
    delta_cer = (b_cer - a_cer) if (b_cer is not None and a_cer is not None) else None
    delta_wer = (b_wer - a_wer) if (b_wer is not None and a_wer is not None) else None
    delta_conf = (a_conf - b_conf) if (b_conf is not None and a_conf is not None) else None
    lines = [
        "",
        "═══════════════════════════════════════════════════════════════════",
        "  Baseline (stock) vs. Fine-tuned — average over golden corpus",
        "═══════════════════════════════════════════════════════════════════",
        f"  {'metric':<10} {'before':>10} {'after':>10} {'Δ':>10}",
        f"  {'─' * 44}",
        f"  {'CER':<10} {_fmt_pct(b_cer):>10} {_fmt_pct(a_cer):>10} "
        f"{(f'{delta_cer * 100:+5.1f}pp' if delta_cer is not None else '—'):>10}",
        f"  {'WER':<10} {_fmt_pct(b_wer):>10} {_fmt_pct(a_wer):>10} "
        f"{(f'{delta_wer * 100:+5.1f}pp' if delta_wer is not None else '—'):>10}",
        f"  {'mean_conf':<10} {_fmt_pct(b_conf):>10} {_fmt_pct(a_conf):>10} "
        f"{(f'{delta_conf * 100:+5.1f}pp' if delta_conf is not None else '—'):>10}",
        "═══════════════════════════════════════════════════════════════════",
    ]
    if delta_cer is not None:
        if delta_cer > 0.03:
            lines.append("  ✓ Fine-tune IMPROVED accuracy — keep the new weights.")
        elif delta_cer < -0.03:
            lines.append("  ✗ Fine-tune HURT accuracy — consider rolling back.")
        else:
            lines.append("  ~ No significant change — train longer / clean data.")
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Run all checks and exit without training."
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the baseline metrics pass. Use if you ran it separately.",
    )
    parser.add_argument(
        "--skip-metrics", action="store_true",
        help="Skip all metrics — train only. Fastest option if you'll "
             "validate manually later.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Passed through to finetune_recognizer.py (default: %(default)s).",
    )
    parser.add_argument(
        "--auto-apply-if-better", type=float, default=None, metavar="PP",
        help="Automatically keep new weights if avg CER dropped by at "
             "least this many percentage points (e.g. 3.0 = -3pp or better).",
    )
    args = parser.parse_args()

    t0 = time.time()
    try:
        report = preflight()
    except PreflightError as e:
        print(f"\n[preflight] FAILED:\n  {e}", file=sys.stderr)
        return 1
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if args.preflight_only:
        print("[preflight] --preflight-only set, exiting.")
        return 0

    # Ensure we have a stock backup before doing anything destructive.
    if not STOCK_BACKUP.exists() and BASE_WEIGHTS.exists():
        shutil.copy2(BASE_WEIGHTS, STOCK_BACKUP)
        print(f"[runner] Backed up stock weights → {STOCK_BACKUP.name}")

    # Step 1: baseline metrics (skip if requested).
    before = {}
    if not (args.skip_metrics or args.skip_baseline):
        sync_weights_to_cache(STOCK_BACKUP if STOCK_BACKUP.exists() else BASE_WEIGHTS)
        before = run_metrics("baseline", REPORTS_DIR / "metrics_baseline.json")
        if before:
            print(f"[metrics-baseline] CER: {before.get('avg_cer', 0) * 100:.1f}%  "
                  f"WER: {before.get('avg_wer', 0) * 100:.1f}%  "
                  f"conf: {before.get('avg_conf', 0) * 100:.1f}%")

    # Step 2: fine-tune.
    print(f"\n[runner] Starting fine-tune ({args.epochs} × 1000 iterations) …")
    rc = subprocess.call(
        [sys.executable, str(REPO_ROOT / "scripts" / "finetune" / "finetune_recognizer.py"),
         "--epochs", str(args.epochs)],
        cwd=str(REPO_ROOT),
    )
    if rc != 0:
        print(f"[runner] Fine-tune failed with exit code {rc}",
              file=sys.stderr)
        return rc

    # Step 3: post-train metrics.
    after = {}
    if not args.skip_metrics:
        sync_weights_to_cache(BASE_WEIGHTS)  # now points at fine-tuned
        after = run_metrics("trained", REPORTS_DIR / "metrics_trained.json")
        if after:
            print(f"[metrics-trained] CER: {after.get('avg_cer', 0) * 100:.1f}%  "
                  f"WER: {after.get('avg_wer', 0) * 100:.1f}%  "
                  f"conf: {after.get('avg_conf', 0) * 100:.1f}%")

    # Step 4: summary + auto-apply decision.
    if before and after:
        report_text = summarise(before, after)
        print(report_text)
        (REPORTS_DIR / "report.txt").write_text(report_text + "\n",
                                                encoding="utf-8")
        if args.auto_apply_if_better is not None:
            delta_pp = (before.get("avg_cer", 0) - after.get("avg_cer", 0)) * 100
            if delta_pp >= args.auto_apply_if_better:
                print(f"\n[runner] ✓ Δ CER = -{delta_pp:.1f}pp ≥ "
                      f"threshold {args.auto_apply_if_better}pp — "
                      "keeping new weights.")
            else:
                print(f"\n[runner] ✗ Δ CER = -{delta_pp:.1f}pp < "
                      f"threshold {args.auto_apply_if_better}pp — "
                      "ROLLING BACK to stock.")
                shutil.copy2(STOCK_BACKUP, BASE_WEIGHTS)
                print(f"[runner] Rolled back: {BASE_WEIGHTS}")

    elapsed = time.time() - t0
    print(f"\n[runner] Total wall-clock: {elapsed / 60:.1f} min")
    print(f"[runner] Reports in: {REPORTS_DIR}")
    print("[runner] Next steps:")
    print("  1. Review reports/finetune_run/report.txt for the delta.")
    print("  2. If numbers are good: python build.py → fresh installer.")
    print("  3. If numbers are bad: cp resources/easyocr_models/"
          "cyrillic_g2.pth.stock resources/easyocr_models/cyrillic_g2.pth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
