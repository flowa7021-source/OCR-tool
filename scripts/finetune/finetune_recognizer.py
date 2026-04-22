"""Fine-tune EasyOCR's cyrillic_g2 recognizer on our bootstrapped dataset.

The official EasyOCR trainer is a subfolder of the upstream repo
(JaidedAI/EasyOCR `trainer/`) that ships its own config format and
``train.py`` entrypoint. This script:

    1. Clones the trainer repo into ``.finetune_trainer/`` (idempotent).
    2. Writes a YAML config pointing at ``datasets/finetune_ru/``.
    3. Seeds training from the shipped ``resources/easyocr_models/
       cyrillic_g2.pth`` (continues, doesn't train from scratch).
    4. Invokes ``python train.py --config ...`` inside the trainer.
    5. When training finishes, copies the best ``.pth`` back into
       ``resources/easyocr_models/cyrillic_g2.pth`` so ``build.py``
       bundles the fine-tuned weights into the next installer.

Usage:

    python scripts/finetune/bootstrap_dataset.py      # prepare dataset
    python scripts/finetune/finetune_recognizer.py    # run training

Defaults to 100 epochs on GPU; falls back to CPU when CUDA is missing
(painfully slow — set ``--epochs 5`` for a smoke-run).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINER_DIR = REPO_ROOT / ".finetune_trainer"
TRAINER_REPO = "https://github.com/JaidedAI/EasyOCR.git"
DEFAULT_DATA = REPO_ROOT / "datasets" / "finetune_ru"
MODEL_DIR = REPO_ROOT / "resources" / "easyocr_models"
BASE_WEIGHTS = MODEL_DIR / "cyrillic_g2.pth"


def _clone_trainer() -> Path:
    trainer_root = TRAINER_DIR / "trainer"
    if trainer_root.exists():
        return trainer_root
    TRAINER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[finetune] Cloning EasyOCR trainer into {TRAINER_DIR} …")
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--filter=blob:none",
        "--sparse", TRAINER_REPO, str(TRAINER_DIR),
    ])
    subprocess.check_call(
        ["git", "sparse-checkout", "set", "trainer"], cwd=str(TRAINER_DIR),
    )
    return trainer_root


def _write_config(trainer_root: Path, data_dir: Path, epochs: int) -> Path:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = f"""number: '0123456789'
symbol: " !\\"#$%&'()*+,-./:;<=>?@[\\\\]^_`{{|}}~ €₽"
lang_char: 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
experiment_name: 'ru_finetune'
train_data: '{data_dir.as_posix()}'
valid_data: '{(data_dir / 'validation').as_posix()}'
manualSeed: 1111
workers: 2
batch_size: 32
num_iter: {epochs * 1000}
valInterval: 500
saved_model: '{BASE_WEIGHTS.as_posix()}'
FT: True
optim: 'adam'
lr: 0.0005
beta1: 0.9
rho: 0.95
eps: 0.00000001
grad_clip: 5
select_data: 'training'
batch_ratio: '1.0'
total_data_usage_ratio: '1.0'
batch_max_length: 34
imgH: 64
imgW: 600
rgb: False
contrast_adjust: 0.0
data_filtering_off: False
PAD: True
Transformation: 'None'
FeatureExtraction: 'ResNet'
SequenceModeling: 'BiLSTM'
Prediction: 'CTC'
num_fiducial: 20
input_channel: 1
output_channel: 256
hidden_size: 256
decode: 'greedy'
new_prediction: False
freeze_FeatureFxtraction: False
freeze_SequenceModeling: False
device: '{device}'
"""
    cfg_path = trainer_root / "saved_models" / "ru_finetune.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(cfg, encoding="utf-8")
    print(f"[finetune] Wrote trainer config: {cfg_path} (device={device})")
    return cfg_path


def _run_training(trainer_root: Path, config_path: Path) -> int:
    return subprocess.call(
        [sys.executable, "train.py", "--config_name",
         config_path.stem],
        cwd=str(trainer_root),
    )


def _swap_weights(trainer_root: Path) -> bool:
    best = trainer_root / "saved_models" / "ru_finetune" / "best_accuracy.pth"
    if not best.exists():
        print(f"[finetune] No trained weights at {best}", file=sys.stderr)
        return False
    backup = BASE_WEIGHTS.with_suffix(".pth.stock")
    if not backup.exists() and BASE_WEIGHTS.exists():
        shutil.copy2(BASE_WEIGHTS, backup)
        print(f"[finetune] Backed up stock weights to {backup.name}")
    shutil.copy2(best, BASE_WEIGHTS)
    print(f"[finetune] Installed fine-tuned weights → {BASE_WEIGHTS}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--skip-swap", action="store_true",
                        help="Train but don't overwrite bundled weights")
    args = parser.parse_args()

    if not (args.data / "training" / "gt.txt").exists():
        print(
            f"[finetune] Missing dataset at {args.data}. "
            "Run: python scripts/finetune/bootstrap_dataset.py",
            file=sys.stderr,
        )
        return 1
    if not BASE_WEIGHTS.exists():
        print(
            f"[finetune] Missing base weights at {BASE_WEIGHTS}. "
            "Run: python scripts/prefetch_easyocr_models.py",
            file=sys.stderr,
        )
        return 1

    trainer_root = _clone_trainer()
    config_path = _write_config(trainer_root, args.data, args.epochs)
    rc = _run_training(trainer_root, config_path)
    if rc != 0:
        print(f"[finetune] Training failed with exit code {rc}",
              file=sys.stderr)
        return rc
    if not args.skip_swap and not _swap_weights(trainer_root):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
