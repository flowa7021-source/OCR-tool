"""Детальный разбор результатов spike: что именно читают движки vs ground truth."""

from __future__ import annotations
import re
import sys
import time
import tempfile
from pathlib import Path
from collections import Counter

import fitz
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"

DOCS = [
    "TN_k_UPD_36_ot_02.09.2022",
    "TN_k_UPD_41_ot_06.09.2022",
    "TN_k_UPD_47_ot_09.01.2023",
    "TN_k_UPD_48_ot_09.01.2023",
    "UPD_662_ot_22.10.2022",
]

# ──────────────────────────────────────────────────────────────────────────────
# OCR helpers (те же, что в spike.py)
# ──────────────────────────────────────────────────────────────────────────────

def tess_ocr(pdf_path: Path) -> str:
    import pytesseract
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(doc.page_count):
        pix = doc[i].get_pixmap(dpi=300, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        pages.append(pytesseract.image_to_string(img, lang="rus+eng", config="--oem 1 --psm 3"))
    doc.close()
    return "\n".join(pages)


def easy_ocr(pdf_path: Path, reader) -> str:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(doc.page_count):
        pix = doc[i].get_pixmap(dpi=300, colorspace=fitz.csGRAY)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        results = reader.readtext(arr, detail=1, paragraph=False)
        pages.append("\n".join(t for _, t, c in results if c > 0.1))
    doc.close()
    return "\n".join(pages)


def load_gt(stem: str) -> str:
    txt = (INPUTS_DIR / f"{stem}.txt").read_text(encoding="utf-8")
    lines = [l for l in txt.splitlines()
             if not re.fullmatch(r"[=\-]{10,}", l.strip())
             and not l.strip().startswith(("ФАЙЛ:", "СТРАНИЦ:"))]
    return "\n".join(lines)


def norm(text: str) -> str:
    text = re.sub(r"\s+", " ", text).lower().strip()
    return text

# ──────────────────────────────────────────────────────────────────────────────
# Метрики
# ──────────────────────────────────────────────────────────────────────────────

def cer(hyp: str, ref: str) -> float:
    from jiwer import cer as _cer
    return _cer(ref, hyp)

def wer(hyp: str, ref: str) -> float:
    from jiwer import wer as _wer
    return _wer(ref, hyp)

def cyrillic_ratio(text: str) -> float:
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return 0.0
    cyr = sum(1 for c in chars if 'Ѐ' <= c <= 'ӿ')
    return cyr / len(chars)

def noise_ratio(text: str) -> float:
    """Доля символов, не являющихся буквами/цифрами/пробелами/punct."""
    if not text:
        return 0.0
    noise = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in '.,;:!?-–—«»""\'()[]{}@#%&*+=/<>|\\~^'))
    return noise / len(text)

KEY_ENTITIES = [
    # ИНН
    ("7813266190", "ИНН ГЕКСАФОРМ"),
    ("7707820890", "ИНН Моспроект"),
    ("7811757210", "ИНН ДЕЛОВЫЕ ПЕРЕВОЗКИ"),
    # КПП
    ("770701001", "КПП Моспроект"),
    ("781101001", "КПП ДЕЛОВЫЕ ПЕРЕВОЗКИ"),
    # ОГРН
    ("5137746157490", "ОГРН Моспроект"),
    # Организации
    ("гексаформ", "Название ГЕКСАФОРМ"),
    ("моспроект", "Название Моспроект"),
    ("деловые перевозки", "Название ДЕЛОВЫЕ ПЕРЕВОЗКИ"),
    # Водитель
    ("беляев", "Фамилия водителя"),
    # Номер ТН
    ("2908-23а", "Номер ТН"),
    # Даты
    ("29.08.2022", "Дата ТН"),
    # Вес
    ("7309", "Масса груза"),
    # Адреса
    ("блохина", "Ул. Блохина (грузоотправитель)"),
    ("петергоф", "Адрес погрузки"),
    ("щедровка", "Адрес разгрузки"),
]

def check_entities(text_norm: str, entities: list) -> list[tuple[str, str, bool]]:
    return [(name, token, token in text_norm) for token, name in entities]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Загружаем EasyOCR модель…")
    import easyocr
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    print("OK\n")

    for stem in DOCS:
        pdf = INPUTS_DIR / f"{stem}.pdf"
        gt_raw = load_gt(stem)
        gt = norm(gt_raw)

        print("=" * 72)
        print(f"ДОКУМЕНТ: {stem}")
        print("=" * 72)

        # ── Tesseract
        t0 = time.time()
        tess_raw = tess_ocr(pdf)
        t_tess = time.time() - t0
        tess = norm(tess_raw)

        # ── EasyOCR
        t0 = time.time()
        easy_raw = easy_ocr(pdf, reader)
        t_easy = time.time() - t0
        easy = norm(easy_raw)

        # ── Метрики
        t_cer = cer(tess, gt)
        e_cer = cer(easy, gt)
        t_wer = wer(tess, gt)
        e_wer = wer(easy, gt)

        print(f"  Chars (GT / Tess / Easy): {len(gt)} / {len(tess)} / {len(easy)}")
        print(f"  CER:  Tesseract={t_cer*100:.1f}%   EasyOCR={e_cer*100:.1f}%   Δ={( e_cer - t_cer)*100:+.1f}pp")
        print(f"  WER:  Tesseract={t_wer*100:.1f}%   EasyOCR={e_wer*100:.1f}%   Δ={(e_wer - t_wer)*100:+.1f}pp")
        print(f"  Скорость: Tess={t_tess:.0f}s  Easy={t_easy:.0f}s")

        print(f"\n  Кириллица (Tess/Easy): {cyrillic_ratio(tess)*100:.0f}% / {cyrillic_ratio(easy)*100:.0f}%")
        print(f"  Мусор-символы (Tess/Easy): {noise_ratio(tess)*100:.1f}% / {noise_ratio(easy)*100:.1f}%")

        # ── Ключевые сущности (TN_k_UPD_36 entities)
        entities_to_check = KEY_ENTITIES if "36" in stem else [
            ("ооо", "Слово ООО"),
            ("инн", "Слово ИНН"),
            ("кпп", "Слово КПП"),
        ]
        print("\n  Ключевые сущности:")
        tess_hit = easy_hit = 0
        for name, token, _ in check_entities(gt, entities_to_check):
            if token not in gt:
                continue
            t_found = token in tess
            e_found = token in easy
            tess_hit += t_found
            easy_hit += e_found
            t_mark = "✅" if t_found else "❌"
            e_mark = "✅" if e_found else "❌"
            print(f"    {name:<35} Tess:{t_mark}  Easy:{e_mark}")
        print(f"  Итого сущностей найдено: Tess={tess_hit}/{len(entities_to_check)}  Easy={easy_hit}/{len(entities_to_check)}")

        # ── Фрагмент первой страницы
        print("\n  --- Tesseract (первые 400 chars) ---")
        print(" ", tess[:400])
        print("\n  --- EasyOCR (первые 400 chars) ---")
        print(" ", easy[:400])
        print("\n  --- Ground Truth (первые 400 chars) ---")
        print(" ", gt[:400])
        print()


if __name__ == "__main__":
    main()
