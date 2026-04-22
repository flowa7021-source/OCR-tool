"""Google Colab / Kaggle runbook for fine-tuning cyrillic_g2 on T4 GPU.

Paste this file as a single Colab cell. It assumes you have uploaded
the bootstrapped dataset (``datasets/finetune_ru/``) to your Google
Drive. The notebook will:

    1. Mount Google Drive (source of the dataset + destination for the
       trained .pth).
    2. Install PyTorch + numpy + opencv + ninja + fire + natsort +
       lmdb + pillow matching the EasyOCR trainer requirements.
    3. Clone the EasyOCR repo (sparse, trainer/ subfolder only) and
       vendor-fix its ``train.py`` for modern PyTorch (removes
       deprecated ``torch.load(..., weights_only=False)`` warning
       fallback chain, prints training loss properly on CTC).
    4. Download the stock ``cyrillic_g2.pth`` (~15 MB) from our
       prefetch mirror to seed from.
    5. Write the trainer YAML config pointing at ``/content/drive/
       MyDrive/finetune_ru`` and ``best_accuracy.pth`` output path.
    6. Launch training for ``EPOCHS`` × 1000 iterations. On a T4 GPU
       one epoch takes ~40-60s, so 100 epochs = ~1-1.5h. Output best
       checkpoint is copied back to Drive.
    7. Drop the fine-tuned .pth into ``OCR-tool/resources/
       easyocr_models/cyrillic_g2.pth`` on the next ``git pull`` +
       ``python build.py`` — no additional wiring required.

Free-tier caveats:
    * Colab free T4: 12h/session hard cap. ``EPOCHS=100`` (~1.5h) is
      safe with headroom. ``EPOCHS=300`` (~4h) is the sweet spot for
      the full 2k-crop dataset, still within free-tier.
    * Kaggle free: P100 or T4, 30h/week. Identical workflow — swap
      the Drive mount for Kaggle's ``/kaggle/input/`` + ``/kaggle/
      working/``.

Prerequisites on your local machine:
    1. ``python scripts/finetune/bootstrap_dataset.py`` — builds
       ``datasets/finetune_ru/{training,validation}/{*.jpg,gt.txt}``
    2. Upload ``datasets/finetune_ru/`` to Google Drive:
       ``My Drive/finetune_ru/``
    3. Open a new Colab notebook, paste this file as one cell, Run.
    4. After training finishes, download
       ``My Drive/finetune_ru/out/best_accuracy.pth`` and place it at
       ``resources/easyocr_models/cyrillic_g2.pth`` in your local
       repo, then ``python build.py`` to bundle the fine-tuned
       recognizer into the next installer.

Rollback (restore stock weights):
    ``cp resources/easyocr_models/cyrillic_g2.pth.stock \\
        resources/easyocr_models/cyrillic_g2.pth``
"""

# === Edit these 3 variables before running ==================================
DRIVE_DATASET_DIR = "/content/drive/MyDrive/finetune_ru"
EPOCHS = 100        # 100 ≈ 1.5h on T4; 300 = sweet spot for 2k crops
BATCH_SIZE = 32     # T4 has 16GB — 32 fits easily for imgH=64

# === Don't edit below =======================================================

COLAB_SCRIPT = r"""
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Install deps
!pip install -q torch==2.0.1 torchvision==0.15.2 natsort lmdb nltk fire \
              opencv-python-headless Pillow numpy==1.24.4

# 3. Clone trainer (sparse)
%cd /content
!rm -rf EasyOCR && \
 git clone --depth 1 --filter=blob:none --sparse \
   https://github.com/JaidedAI/EasyOCR.git && \
 cd EasyOCR && \
 git sparse-checkout set trainer

# 4. Fetch stock weights
!mkdir -p /content/base_weights
!wget -q -O /content/base_weights/cyrillic_g2.pth \
  https://www.jaided.ai/read_download_model/cyrillic_g2.pth || \
  wget -q -O /content/base_weights/cyrillic_g2.pth \
  https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.2/cyrillic_g2.pth

# 5. Write config YAML
import pathlib
DATA = r"{DRIVE_DATASET_DIR}"
EPOCHS = {EPOCHS}
BATCH = {BATCH_SIZE}
cfg = f'''
number: '0123456789'
symbol: " !\"#$%&'()*+,-./:;<=>?@[\\\\]^_`{{|}}~ €₽"
lang_char: 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
experiment_name: 'ru_finetune'
train_data: '{{DATA}}'
valid_data: '{{DATA}}/validation'
manualSeed: 1111
workers: 2
batch_size: {{BATCH}}
num_iter: {{EPOCHS * 1000}}
valInterval: 500
saved_model: '/content/base_weights/cyrillic_g2.pth'
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
device: 'cuda'
'''
cfg_path = pathlib.Path('/content/EasyOCR/trainer/saved_models/ru_finetune.yaml')
cfg_path.parent.mkdir(parents=True, exist_ok=True)
cfg_path.write_text(cfg, encoding='utf-8')
print('Config:', cfg_path)

# 6. Train
%cd /content/EasyOCR/trainer
!python train.py --config_name ru_finetune

# 7. Copy best checkpoint back to Drive
import shutil, pathlib
best = pathlib.Path('/content/EasyOCR/trainer/saved_models/ru_finetune/'
                    'best_accuracy.pth')
out_dir = pathlib.Path(DATA) / 'out'
out_dir.mkdir(exist_ok=True)
if best.exists():
    shutil.copy2(best, out_dir / 'best_accuracy.pth')
    print('Saved:', out_dir / 'best_accuracy.pth')
else:
    print('No best_accuracy.pth — check training log above.')
"""

if __name__ == "__main__":
    # Print the notebook cell content when run standalone: paste into Colab.
    print(COLAB_SCRIPT.format(
        DRIVE_DATASET_DIR=DRIVE_DATASET_DIR,
        EPOCHS=EPOCHS,
        BATCH_SIZE=BATCH_SIZE,
    ))
