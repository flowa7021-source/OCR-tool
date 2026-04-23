"""Render synthetic word-crops for EasyOCR recognizer fine-tune.

Rationale: our bootstrapped TN/UPD dataset has ~2900 GT-verified word
crops with 161 unique labels. For a CRNN recogniser that's thin —
rare labels appear 1-5 times, CTC has trouble generalising.

This generator expands the training set by rendering each GT token
(plus synthetic noise variants) in fonts similar to those used on
Russian tax forms. We don't need studio-quality synth — just enough
variety that the network sees each label across different shapes and
light degradation.

Output format matches bootstrap_dataset.py: labels.csv with header
``filename,words``. Crops are appended to an existing dataset dir
(use ``--out`` to write to a separate dir if you want to keep real
and synthetic crops apart for ablation runs).

Usage::

    # 10k synthetic crops sampling from the default corpus vocab
    python scripts/finetune/synth_data.py --n 10000

    # Bigger, more aggressive degradation
    python scripts/finetune/synth_data.py --n 30000 --max-degradation 0.7

    # Write to a fresh dir for comparison
    python scripts/finetune/synth_data.py --out datasets/synthetic_only --n 20000
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT = REPO_ROOT / "datasets" / "finetune_ru"

# Fonts to sample from — aim for variety in stroke weight and width.
# Paths are Linux-typical; on Windows the script skips missing ones.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    str(REPO_ROOT / "resources" / "DejaVuSans.ttf"),
)


def _find_fonts() -> list[Path]:
    fonts = [Path(p) for p in _FONT_CANDIDATES if Path(p).exists()]
    # Deduplicate by content hash (DejaVuSans.ttf in two places → same file).
    seen: dict[str, Path] = {}
    for f in fonts:
        h = hashlib.sha1(f.read_bytes()[:4096]).hexdigest()
        seen.setdefault(h, f)
    return list(seen.values())


def _load_vocab(
    inputs_dir: Path,
    expected_dir: Path,
) -> Counter[str]:
    """Reuse the bootstrap's GT-token extraction — same tokens the
    real crops are labelled against, so synthetic crops share the
    vocab.
    """
    # Import lazily — avoids pulling easyocr just for tokenisation.
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "finetune"))
    from bootstrap_dataset import _load_gt_tokens, _walk_json_strings  # noqa: F401

    vocab: Counter[str] = Counter()
    for pdf in inputs_dir.glob("*.pdf"):
        for tok in _load_gt_tokens(pdf.stem, inputs_dir, expected_dir):
            vocab[tok] += 1
    return vocab


def _render_word(
    word: str,
    font_path: Path,
    font_size: int,
    padding: int = 6,
    bg: int = 255,
    fg: int = 0,
    char_spacing_jitter: float = 0.0,
) -> Image.Image:
    """Render ``word`` centered on a white-ish canvas using ``font_path``.

    When ``char_spacing_jitter > 0``, characters are drawn individually
    with randomised horizontal offsets in [−j × W, +j × W] where W is
    the glyph advance. Simulates variable kerning seen on aged printed
    forms where the scanner's sub-pixel sampling bleeds character
    boundaries unevenly. Matches one of the hardest-to-learn artefacts
    on real ТН/УПД scans.
    """
    font = ImageFont.truetype(str(font_path), font_size)
    tmp = Image.new("L", (1, 1), bg)
    draw_tmp = ImageDraw.Draw(tmp)
    bbox = draw_tmp.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Add a small extra margin when per-char jitter is active so the
    # accumulated offset doesn't clip.
    jitter_pad = int(font_size * char_spacing_jitter * len(word)) if char_spacing_jitter else 0
    canvas_w = tw + padding * 2 + jitter_pad
    canvas_h = th + padding * 2
    img = Image.new("L", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(img)

    if char_spacing_jitter <= 0.0:
        # Fast path — no per-char jitter, draw whole string at once.
        draw.text((padding - bbox[0], padding - bbox[1]),
                  word, fill=fg, font=font)
        return img

    # Per-character draw with random horizontal offset. Y stays the same
    # so baseline alignment is preserved (critical for OCR).
    rng = random.Random(hash(word) & 0xFFFFFFFF)
    x = padding - bbox[0]
    y = padding - bbox[1]
    for ch in word:
        ch_bbox = draw_tmp.textbbox((0, 0), ch, font=font)
        ch_w = ch_bbox[2] - ch_bbox[0]
        draw.text((x, y), ch, fill=fg, font=font)
        jitter = rng.uniform(-char_spacing_jitter, char_spacing_jitter) * ch_w
        x += ch_w + jitter
    return img


def _augment(img: Image.Image, intensity: float, rng: random.Random) -> Image.Image:
    """Apply randomised degradation proportional to ``intensity`` in [0, 1].

    Simulates what scanning does to printed text: blur, JPEG blockiness,
    salt-and-pepper speckle, mild rotation, stroke erosion, ink bleed,
    horizontal stretch/squish (simulates non-uniform scanner sampling).
    """
    # 1. Rotation ±2° × intensity
    max_deg = 2.0 * intensity
    if max_deg > 0.1:
        angle = rng.uniform(-max_deg, max_deg)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255,
                         expand=True)

    # 1b. Horizontal stretch/squish (simulates scanner sub-pixel sampling
    #     errors): scale W by factor in [1 − 0.15×intensity, 1 + 0.15×intensity].
    stretch = 1.0 + rng.uniform(-0.15, 0.15) * intensity
    if abs(stretch - 1.0) > 0.02:
        w, h = img.size
        new_w = max(4, int(round(w * stretch)))
        img = img.resize((new_w, h), Image.BICUBIC)

    # 1c. Ink bleed / stroke thickening — dilate dark pixels so adjacent
    #     glyphs merge (common on overinked forms or heavy scan pressure).
    #     PIL's MinFilter applied to grayscale extends dark regions
    #     (equivalent to dilating the ink).
    if intensity > 0.4 and rng.random() < 0.25:
        img = img.filter(ImageFilter.MinFilter(3))

    # 2. Gaussian blur σ in [0, 1.5 × intensity]
    sigma = rng.uniform(0, 1.5 * intensity)
    if sigma > 0.1:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # 3. Contrast jitter (multiplicative scale on pixel values)
    arr = np.asarray(img, dtype=np.float32)
    contrast = 1.0 + rng.uniform(-0.3 * intensity, 0.3 * intensity)
    brightness = rng.uniform(-20 * intensity, 20 * intensity)
    arr = np.clip(arr * contrast + brightness, 0, 255)

    # 4. Salt-and-pepper noise
    noise_frac = 0.02 * intensity
    if noise_frac > 0.001:
        mask = rng.random()  # noqa: F841 — placeholder, unused
        n_pixels = int(arr.size * noise_frac)
        if n_pixels > 0:
            ys = np.random.randint(0, arr.shape[0], n_pixels)
            xs = np.random.randint(0, arr.shape[1], n_pixels)
            vals = np.random.choice([0, 255], n_pixels)
            arr[ys, xs] = vals

    # 5. JPEG-style compression artefacts — round-trip through a JPEG
    # buffer at a randomly degraded quality level.
    img = Image.fromarray(arr.astype(np.uint8))
    if intensity > 0.2:
        q = int(rng.uniform(40, 95 - 40 * intensity))
        import io
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("L").copy()

    return img


def _split_bucket(key: str) -> str:
    h = hashlib.sha1(key.encode()).hexdigest()
    return "validation" if int(h[:4], 16) % 5 == 0 else "training"


def build(
    vocab: Counter[str],
    fonts: list[Path],
    n_samples: int,
    out_dir: Path,
    max_degradation: float = 0.6,
    min_font_size: int = 14,
    max_font_size: int = 48,
    multi_dpi: bool = True,
    char_jitter_prob: float = 0.25,
    char_jitter_max: float = 0.08,
    seed: int = 1337,
    start_index: int = 1_000_000,
) -> int:
    """Render ``n_samples`` synthetic crops and append to ``out_dir``.

    Starts labelling from ``img_01000000.jpg`` to stay clear of real
    crop indices (bootstrap uses 8 digits from 1). Appends rows to
    the existing labels.csv / gt.txt in-place.
    """
    if not fonts:
        print("[synth] No fonts available — install ttf-dejavu / fontconfig",
              file=sys.stderr)
        return 1

    rng = random.Random(seed)
    np.random.seed(seed)

    (out_dir / "training").mkdir(parents=True, exist_ok=True)
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)

    vocab_words = list(vocab.keys())
    # Weight sampling by frequency × mild flattening — very rare words
    # still get at least a handful of samples.
    weights = [max(1, vocab[w]) ** 0.6 for w in vocab_words]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    train_gt = (out_dir / "training" / "gt.txt").open("a", encoding="utf-8")
    val_gt = (out_dir / "validation" / "gt.txt").open("a", encoding="utf-8")
    train_csv = (out_dir / "training" / "labels.csv")
    val_csv = (out_dir / "validation" / "labels.csv")
    # Preserve existing header / contents — only create if absent.
    for p in (train_csv, val_csv):
        if not p.exists():
            p.write_text("filename,words\n", encoding="utf-8")
    train_csv_f = train_csv.open("a", encoding="utf-8")
    val_csv_f = val_csv.open("a", encoding="utf-8")

    generated = 0
    # Multi-DPI size buckets — biased distribution so model sees mostly
    # mid-range with enough tails to handle 150-dpi and 600-dpi extremes.
    size_pool: list[int] = []
    if multi_dpi:
        size_pool = (
            list(range(min_font_size, 22)) * 1       # tiny (low-DPI scans)
            + list(range(22, 32)) * 3                # mid (default 300-DPI)
            + list(range(32, 40)) * 2                # large (high-DPI)
            + list(range(40, max_font_size + 1)) * 1 # extra-large (zoomed)
        )
    try:
        for i in range(n_samples):
            word = rng.choices(vocab_words, weights=probs, k=1)[0]
            font = rng.choice(fonts)
            size = (
                rng.choice(size_pool) if size_pool
                else rng.randint(min_font_size, max_font_size)
            )
            # Per-char horizontal jitter — applied stochastically so only
            # some crops show the artefact, not all. Prevents training
            # from locking onto a consistent "shifted" representation.
            jitter = (rng.uniform(0.02, char_jitter_max)
                      if rng.random() < char_jitter_prob else 0.0)
            try:
                crop = _render_word(
                    word, font, size, char_spacing_jitter=jitter,
                )
            except OSError as e:
                print(f"[synth] skip {word!r} on {font.name}: {e}",
                      file=sys.stderr)
                continue
            crop = _augment(
                crop, rng.uniform(0.1, max_degradation), rng,
            )

            idx = start_index + i
            key = f"synth:{idx}"
            bucket = _split_bucket(key)
            filename = f"img_{idx:08d}.jpg"
            path = out_dir / bucket / filename
            crop.convert("L").save(path, "JPEG", quality=92)
            tgt_gt = train_gt if bucket == "training" else val_gt
            tgt_csv = train_csv_f if bucket == "training" else val_csv_f
            tgt_gt.write(f"{filename}\t{word}\n")
            safe = word.replace('"', '""')
            tgt_csv.write(f'{filename},"{safe}"\n')
            generated += 1
            if generated % 1000 == 0:
                print(f"[synth] rendered {generated}/{n_samples}",
                      file=sys.stderr)
    finally:
        train_gt.close()
        val_gt.close()
        train_csv_f.close()
        val_csv_f.close()

    print(f"[synth] Done: {generated} synthetic crops written to {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs", type=Path, default=REPO_ROOT / "inputs",
        help="Dir with *.pdf (to collocate .txt transcripts)",
    )
    parser.add_argument(
        "--expected", type=Path, default=REPO_ROOT / "expected",
        help="Dir with *.json structured references",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=10000,
                        help="Number of synthetic crops (default: %(default)s)")
    parser.add_argument("--max-degradation", type=float, default=0.6,
                        help="Upper bound on augmentation intensity in [0, 1]")
    parser.add_argument("--min-font-size", type=int, default=14,
                        help="Smallest rendered font (px). Default 14 — "
                             "exercises model on low-DPI scan equivalents.")
    parser.add_argument("--max-font-size", type=int, default=48,
                        help="Largest rendered font (px). Default 48 — "
                             "covers zoomed-in / high-DPI crops.")
    parser.add_argument("--no-multi-dpi", action="store_true",
                        help="Disable DPI-weighted size bucketing "
                             "(use uniform random in [min, max] instead).")
    parser.add_argument("--char-jitter-prob", type=float, default=0.25,
                        help="Fraction of crops that get per-character "
                             "horizontal jitter applied (default: 0.25).")
    parser.add_argument("--char-jitter-max", type=float, default=0.08,
                        help="Max per-character horizontal offset as a "
                             "fraction of glyph width (default: 0.08).")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    fonts = _find_fonts()
    if not fonts:
        print("[synth] ERROR: no fonts found. Install "
              "ttf-dejavu / fonts-liberation.", file=sys.stderr)
        return 1
    print(f"[synth] Using {len(fonts)} font(s):")
    for f in fonts:
        print(f"  {f}")

    vocab = _load_vocab(args.inputs, args.expected)
    if not vocab:
        print(f"[synth] ERROR: no GT tokens found in {args.inputs} / "
              f"{args.expected}", file=sys.stderr)
        return 1
    print(f"[synth] Vocab: {len(vocab)} unique tokens "
          f"(total frequency sum: {sum(vocab.values())})")

    return build(
        vocab, fonts, args.n, args.out,
        max_degradation=args.max_degradation,
        min_font_size=args.min_font_size,
        max_font_size=args.max_font_size,
        multi_dpi=not args.no_multi_dpi,
        char_jitter_prob=args.char_jitter_prob,
        char_jitter_max=args.char_jitter_max,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
