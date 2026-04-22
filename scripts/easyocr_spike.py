"""Etap 0 — Go/No-Go spike: EasyOCR vs Tesseract accuracy on golden corpus.

Usage:
    python scripts/easyocr_spike.py

What it does:
    1. For each PDF in inputs/, run Tesseract 5 (via ocrmypdf + pytesseract)
       and get the full-document plaintext.
    2. For each PDF in inputs/, run EasyOCR (rus+en) and get plaintext.
    3. Load the human ground-truth from inputs/*.txt.
    4. Compute CER (Character Error Rate) for each engine against the GT.
    5. Print a comparison table.

Decision threshold:
    EasyOCR CER worse by > 5 pp absolute → STOP (don't migrate).
    EasyOCR CER within 2 pp → GREEN LIGHT.
    2-5 pp → re-evaluate after checking specific error categories.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"

# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(txt_path: Path) -> str:
    """Load human-verified transcript; strip structural headers/separators."""
    text = txt_path.read_text(encoding="utf-8")
    # Remove decorative lines and section headers (=====, ----, [Раздел])
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"[=\-]{10,}", stripped):
            continue
        if stripped.startswith("ФАЙЛ:") or stripped.startswith("СТРАНИЦ:"):
            continue
        lines.append(line)
    return "\n".join(lines)


def normalize_for_cer(text: str) -> str:
    """Minimal normalisation for fair CER comparison.

    - collapse whitespace to single spaces
    - lowercase
    - keep Cyrillic, Latin, digits, basic punctuation
    """
    text = re.sub(r"\s+", " ", text)
    text = text.lower().strip()
    return text

# ---------------------------------------------------------------------------
# CER computation (via jiwer)
# ---------------------------------------------------------------------------

def compute_cer(hypothesis: str, reference: str) -> float:
    try:
        from jiwer import cer
        return cer(reference, hypothesis)
    except Exception as e:
        print(f"  [CER error: {e}]")
        return float("nan")

# ---------------------------------------------------------------------------
# Tesseract baseline
# ---------------------------------------------------------------------------

def tesseract_ocr_pdf(pdf_path: Path) -> str:
    """Rasterise each PDF page and run pytesseract directly (no ocrmypdf)."""
    import pytesseract
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    all_text: list[str] = []
    for i in range(doc.page_count):
        page = doc[i]
        pix = page.get_pixmap(dpi=300, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        text = pytesseract.image_to_string(
            img,
            lang="rus+eng",
            config="--oem 1 --psm 3",
        )
        all_text.append(text)
    doc.close()
    return "\n".join(all_text)


# ---------------------------------------------------------------------------
# EasyOCR baseline
# ---------------------------------------------------------------------------

def easyocr_ocr_pdf(pdf_path: Path, reader) -> str:
    """Rasterise each page of a PDF and run EasyOCR on it."""
    doc = fitz.open(str(pdf_path))
    all_text: list[str] = []

    for i in range(doc.page_count):
        page = doc[i]
        # 300 DPI for a good balance of speed and quality
        pix = page.get_pixmap(dpi=300, colorspace=fitz.csGRAY)
        img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width
        )
        # EasyOCR returns [(bbox, text, conf), ...]
        results = reader.readtext(img_arr, detail=1, paragraph=False)
        page_lines = [text for (_, text, conf) in results if conf > 0.1]
        all_text.append("\n".join(page_lines))

    doc.close()
    return "\n".join(all_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pdfs = sorted(INPUTS_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in inputs/")
        sys.exit(1)

    print("=" * 70)
    print("ETAP 0 — EasyOCR vs Tesseract accuracy spike")
    print("=" * 70)

    # --- Ground truths
    ground_truths: dict[str, str] = {}
    for pdf in pdfs:
        txt = pdf.with_suffix(".txt")
        if txt.exists():
            ground_truths[pdf.stem] = normalize_for_cer(load_ground_truth(txt))
        else:
            print(f"[WARN] No ground truth for {pdf.name}")

    # --- Tesseract pass
    print("\n[1/2] Running Tesseract 5 (ocrmypdf, rus+eng, PSM=3, OEM=1)…")
    tess_results: dict[str, str] = {}
    tess_times: dict[str, float] = {}
    for pdf in pdfs:
        if pdf.stem not in ground_truths:
            continue
        print(f"  {pdf.name}…", end=" ", flush=True)
        t0 = time.time()
        try:
            text = tesseract_ocr_pdf(pdf)
            tess_results[pdf.stem] = normalize_for_cer(text)
            tess_times[pdf.stem] = time.time() - t0
            print(f"OK ({tess_times[pdf.stem]:.1f}s, {len(text)} chars)")
        except Exception as e:
            print(f"FAILED: {e}")
            tess_results[pdf.stem] = ""
            tess_times[pdf.stem] = 0.0

    # --- EasyOCR pass
    print("\n[2/2] Loading EasyOCR model (rus, en)…", flush=True)
    t_load = time.time()
    try:
        import easyocr
        reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
        print(f"  Model loaded in {time.time() - t_load:.1f}s")
    except Exception as e:
        print(f"  EasyOCR load FAILED: {e}")
        sys.exit(1)

    easy_results: dict[str, str] = {}
    easy_times: dict[str, float] = {}
    for pdf in pdfs:
        if pdf.stem not in ground_truths:
            continue
        print(f"  {pdf.name}…", end=" ", flush=True)
        t0 = time.time()
        try:
            text = easyocr_ocr_pdf(pdf, reader)
            easy_results[pdf.stem] = normalize_for_cer(text)
            easy_times[pdf.stem] = time.time() - t0
            print(f"OK ({easy_times[pdf.stem]:.1f}s, {len(text)} chars)")
        except Exception as e:
            print(f"FAILED: {e}")
            easy_results[pdf.stem] = ""
            easy_times[pdf.stem] = 0.0

    # --- CER comparison
    print("\n" + "=" * 70)
    print(f"{'Document':<40} {'Tess CER':>10} {'Easy CER':>10} {'Delta':>10} {'Tess s':>8} {'Easy s':>8}")
    print("-" * 70)

    tess_cers: list[float] = []
    easy_cers: list[float] = []

    for pdf in pdfs:
        stem = pdf.stem
        if stem not in ground_truths:
            continue
        gt = ground_truths[stem]
        tcer = compute_cer(tess_results.get(stem, ""), gt)
        ecer = compute_cer(easy_results.get(stem, ""), gt)
        delta = ecer - tcer
        tess_cers.append(tcer)
        easy_cers.append(ecer)
        flag = "✅" if delta <= 0.02 else ("⚠️" if delta <= 0.05 else "❌")
        print(
            f"{pdf.name:<40} {tcer*100:>9.1f}% {ecer*100:>9.1f}% {delta*100:>+9.1f}pp {flag}  "
            f"{tess_times.get(stem,0):>7.1f}s {easy_times.get(stem,0):>7.1f}s"
        )

    if tess_cers and easy_cers:
        import statistics
        avg_tess = statistics.mean(tess_cers)
        avg_easy = statistics.mean(easy_cers)
        avg_delta = avg_easy - avg_tess
        print("-" * 70)
        print(
            f"{'AVERAGE':<40} {avg_tess*100:>9.1f}% {avg_easy*100:>9.1f}% {avg_delta*100:>+9.1f}pp   "
            f"{sum(tess_times.values()):>7.1f}s {sum(easy_times.values()):>7.1f}s"
        )
        print()
        print("VERDICT:")
        if avg_delta <= 0.0:
            print("  ✅ GREEN LIGHT — EasyOCR equal or BETTER than Tesseract.")
            print("  Proceed with full migration plan.")
        elif avg_delta <= 0.02:
            print("  ✅ GREEN LIGHT — EasyOCR within 2pp of Tesseract.")
            print("  Delta is recoverable via postprocess; proceed with migration.")
        elif avg_delta <= 0.05:
            print("  ⚠️  AMBER — EasyOCR 2–5pp worse. Examine error categories.")
            print("  Check if errors are on entity fields (ИНН/КПП) or prose.")
            print("  Consider dоbuchenie (fine-tuning) before committing.")
        else:
            print("  ❌ STOP — EasyOCR >5pp worse on Russian business docs.")
            print("  Migration would regress accuracy unacceptably.")
            print("  Consider PaddleOCR or fine-tuning EasyOCR model before retrying.")

    # --- Quick qualitative sample
    print()
    print("=" * 70)
    print("QUALITATIVE SAMPLE (TN_k_UPD_36, first 600 chars of each engine):")
    print("=" * 70)
    sample_stem = "TN_k_UPD_36_ot_02.09.2022"
    if sample_stem in tess_results:
        print("\n[Tesseract]")
        print(tess_results[sample_stem][:600])
    if sample_stem in easy_results:
        print("\n[EasyOCR]")
        print(easy_results[sample_stem][:600])
    if sample_stem in ground_truths:
        print("\n[Ground Truth]")
        print(ground_truths[sample_stem][:600])


if __name__ == "__main__":
    main()
