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
    if not trainer_root.exists():
        TRAINER_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[finetune] Cloning EasyOCR trainer into {TRAINER_DIR} …")
        subprocess.check_call([
            "git", "clone", "--depth", "1", "--filter=blob:none",
            "--sparse", TRAINER_REPO, str(TRAINER_DIR),
        ])
        subprocess.check_call(
            ["git", "sparse-checkout", "set", "trainer"], cwd=str(TRAINER_DIR),
        )
    _patch_trainer(trainer_root)
    return trainer_root


def _patch_trainer(trainer_root: Path) -> None:
    """Apply compatibility patches for PyTorch ≥ 2.0 and Python 3. Idempotent."""
    _patch_dataset(trainer_root / "dataset.py")
    _patch_train(trainer_root / "train.py")


def _patch_dataset(ds: Path) -> None:
    src = ds.read_text(encoding="utf-8")
    if "from itertools import accumulate as _accumulate" in src:
        return  # already patched
    patched = src.replace(
        "from torch._utils import _accumulate",
        "try:\n    from torch._utils import _accumulate\n"
        "except ImportError:\n    from itertools import accumulate as _accumulate",
    ).replace(
        "data_loader_iter.next()", "next(data_loader_iter)"
    ).replace(
        "self.dataloader_iter_list[i].next()",
        "next(self.dataloader_iter_list[i])",
    )
    if patched != src:
        ds.write_text(patched, encoding="utf-8")
        print("[finetune] Patched trainer/dataset.py for PyTorch 2.x / Python 3")


def _patch_train(tr: Path) -> None:
    src = tr.read_text(encoding="utf-8")
    if "map_location=" in src:
        return  # already patched
    # map_location='cpu' → CPU machines can load CUDA-serialised weights.
    # weights_only=False → needed on PyTorch ≥ 2.6 (default flipped to True and
    # blocks arbitrary pickled state_dicts).
    patched = src.replace(
        "pretrained_dict = torch.load(opt.saved_model)",
        "pretrained_dict = torch.load(opt.saved_model, "
        "map_location='cpu', weights_only=False)",
    )
    if patched != src:
        tr.write_text(patched, encoding="utf-8")
        print("[finetune] Patched trainer/train.py: torch.load map_location=cpu")


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
sensitive: True
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
new_prediction: True
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
    # The EasyOCR trainer has no argparse main block — it's designed to be
    # called from a notebook (trainer.ipynb). We drive it via a small runner.
    runner = (
        "import os, sys, yaml, torch.backends.cudnn as cudnn\n"
        "sys.path.insert(0, os.getcwd())\n"
        "from train import train\n"
        "from utils import AttrDict\n"
        "with open(sys.argv[1], encoding='utf-8') as f:\n"
        "    opt = AttrDict(yaml.safe_load(f))\n"
        "opt.character = opt.number + opt.symbol + opt.lang_char\n"
        "os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)\n"
        "cudnn.benchmark = True\n"
        "cudnn.deterministic = False\n"
        "train(opt, amp=False)\n"
    )
    runner_path = trainer_root / "_runner.py"
    runner_path.write_text(runner, encoding="utf-8")
    return subprocess.call(
        [sys.executable, str(runner_path), str(config_path)],
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
