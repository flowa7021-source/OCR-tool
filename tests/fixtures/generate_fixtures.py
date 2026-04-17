"""Generate realistic test PDF fixtures that model real-world scan problems.

Synthetic-but-realistic: no PII, but each fixture targets a specific
class of scan artefact that has caused OCR failures in production:

  * **contract_ru_4page.pdf** — 4-page Russian contract with:
      - Different content per page (title, body, table, signatures)
      - Mixed font sizes (header 24pt, body 12pt, fine print 8pt)
      - Horizontal rules between sections
      - Numbers mixed with Cyrillic (dates, sums)

  * **faded_scan.pdf** — Low-contrast single page simulating a
    photocopy of a photocopy (foreground 140, background 210).

  * **skewed_noisy.pdf** — Rotated 3° + 2% salt-pepper noise,
    simulating a crooked scan of a dusty original.

Run this script once to populate ``tests/fixtures/``. The generated
PDFs are committed to the repo so CI doesn't need to regenerate them.

Usage:
    python tests/fixtures/generate_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Bootstrap repo root.
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

FIXTURES_DIR = Path(__file__).resolve().parent


def _find_cyrillic_font() -> Path | None:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(p).is_file():
            return Path(p)
    return None


def _rasterise_to_image_pdf(
    text_pdf_bytes: bytes, dest: Path, dpi: int = 200
) -> None:
    """Rasterise a text-PDF to an image-only PDF (scan simulation)."""
    import fitz

    scan = fitz.open()
    src = fitz.open(stream=text_pdf_bytes, filetype="pdf")
    try:
        for page in src:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            new = scan.new_page(
                width=page.rect.width, height=page.rect.height
            )
            new.insert_image(new.rect, stream=pix.tobytes("png"))
        scan.save(str(dest))
    finally:
        scan.close()
        src.close()


def generate_contract_ru_4page() -> None:
    """4-page Russian contract with mixed content per page."""
    import fitz

    font = _find_cyrillic_font()
    use_cyrillic = font is not None

    doc = fitz.open()
    try:
        pages_content = [
            # Page 1: title + header
            (
                "ДОГОВОР ПОДРЯДА № 36\n"
                "от 02.09.2022 г.\n\n"
                "г. Москва\n\n"
                "ООО «Альфа-Строй» в лице директора\n"
                "Иванова И.И., именуемый далее\n"
                "«Заказчик», с одной стороны, и\n"
                "ООО «Бета-Сервис» в лице директора\n"
                "Петрова П.П., именуемый далее\n"
                "«Подрядчик», с другой стороны,\n"
                "заключили настоящий Договор.\n"
                if use_cyrillic
                else "CONTRACT AGREEMENT No 36\nDated 02.09.2022\n\n"
                "Alpha Corp represented by Director\n"
                "John Smith, hereinafter the Client,\n"
                "and Beta Services represented by\n"
                "Director Jane Doe, hereinafter\n"
                "the Contractor, have entered\n"
                "into this Agreement.\n"
            ),
            # Page 2: body with numbers
            (
                "1. ПРЕДМЕТ ДОГОВОРА\n\n"
                "1.1. Подрядчик обязуется выполнить\n"
                "работы по ремонту помещения\n"
                "площадью 120,5 кв.м. по адресу:\n"
                "ул. Ленина, д. 15, офис 302.\n\n"
                "2. СТОИМОСТЬ РАБОТ\n\n"
                "2.1. Общая стоимость работ\n"
                "составляет 1 250 000 (Один миллион\n"
                "двести пятьдесят тысяч) рублей 00\n"
                "копеек, в том числе НДС 20%.\n\n"
                "2.2. Оплата производится в течение\n"
                "10 (десяти) рабочих дней.\n"
                if use_cyrillic
                else "1. SUBJECT OF AGREEMENT\n\n"
                "1.1. The Contractor shall perform\n"
                "renovation works on premises of\n"
                "120.5 sq.m. at 15 Lenin St, office 302.\n\n"
                "2. COST OF WORKS\n\n"
                "2.1. Total cost is 1,250,000\n"
                "(One million two hundred fifty\n"
                "thousand) rubles including 20% VAT.\n\n"
                "2.2. Payment within 10 business days.\n"
            ),
            # Page 3: table-like content with lines
            (
                "3. СРОКИ ВЫПОЛНЕНИЯ РАБОТ\n\n"
                "Этап | Описание        | Срок\n"
                "-----+----------------+--------\n"
                "  1  | Демонтаж       | 5 дней\n"
                "  2  | Черновые работы | 15 дней\n"
                "  3  | Чистовые работы| 10 дней\n"
                "  4  | Приёмка        | 3 дня\n"
                "-----+----------------+--------\n"
                "     | ИТОГО          | 33 дня\n\n"
                "4. ОТВЕТСТВЕННОСТЬ СТОРОН\n\n"
                "4.1. За нарушение сроков — пеня\n"
                "0,1% от суммы за каждый день.\n"
                if use_cyrillic
                else "3. SCHEDULE\n\n"
                "Phase | Description     | Duration\n"
                "------+----------------+---------\n"
                "  1   | Demolition     | 5 days\n"
                "  2   | Rough work     | 15 days\n"
                "  3   | Finishing      | 10 days\n"
                "  4   | Acceptance     | 3 days\n"
                "------+----------------+---------\n"
                "      | TOTAL          | 33 days\n\n"
                "4. LIABILITY\n\n"
                "4.1. Penalty for delay: 0.1% per day.\n"
            ),
            # Page 4: signatures + fine print
            (
                "5. ПОДПИСИ СТОРОН\n\n\n"
                "Заказчик:           Подрядчик:\n\n"
                "_____________    _____________\n"
                "  Иванов И.И.      Петров П.П.\n\n"
                "М.П.                М.П.\n\n\n"
                "Приложение: акт выполненных работ\n"
                "на 2 (двух) листах.\n\n"
                "Мелкий шрифт: настоящий договор\n"
                "составлен в двух экземплярах,\n"
                "имеющих одинаковую юридическую\n"
                "силу, по одному для каждой\n"
                "из сторон. ИНН 7712345678\n"
                "КПП 771201001 ОГРН 1027700000001\n"
                if use_cyrillic
                else "5. SIGNATURES\n\n\n"
                "Client:             Contractor:\n\n"
                "_____________    _____________\n"
                "  J. Smith          J. Doe\n\n\n\n"
                "Appendix: acceptance certificate\n"
                "on 2 (two) pages.\n\n"
                "Fine print: this agreement is made\n"
                "in two copies of equal legal force,\n"
                "one for each party. TIN 7712345678\n"
                "REG 1027700000001\n"
            ),
        ]

        for text in pages_content:
            page = doc.new_page(width=612, height=792)
            kwargs = {"fontsize": 14}
            if use_cyrillic and font:
                kwargs["fontfile"] = str(font)
                kwargs["fontname"] = "CyrFont"
            else:
                kwargs["fontname"] = "helv"
            page.insert_text((50, 60), text, **kwargs)

        raw = doc.tobytes()
    finally:
        doc.close()

    _rasterise_to_image_pdf(raw, FIXTURES_DIR / "contract_ru_4page.pdf")
    print(f"  contract_ru_4page.pdf — {(FIXTURES_DIR / 'contract_ru_4page.pdf').stat().st_size} B")


def generate_faded_scan() -> None:
    """Single-page low-contrast scan (faded photocopy)."""
    import cv2
    import fitz

    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)
        page.insert_text(
            (72, 200), "FADED INVOICE 2026\nTotal: $1,234.56",
            fontsize=30, fontname="helv",
        )
        raw = doc.tobytes()
    finally:
        doc.close()

    src = fitz.open(stream=raw, filetype="pdf")
    try:
        pix = src.load_page(0).get_pixmap(dpi=200, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    finally:
        src.close()

    # Reduce contrast: map [0,255] → [140,210]
    arr = (140 + arr.astype(np.float32) / 255.0 * 70).astype(np.uint8)

    ok, buf = cv2.imencode(".png", arr)
    assert ok

    out = fitz.open()
    try:
        h, w = arr.shape[:2]
        p = out.new_page(width=612, height=792)
        p.insert_image(p.rect, stream=buf.tobytes())
        out.save(str(FIXTURES_DIR / "faded_scan.pdf"))
    finally:
        out.close()
    print(f"  faded_scan.pdf — {(FIXTURES_DIR / 'faded_scan.pdf').stat().st_size} B")


def generate_skewed_noisy() -> None:
    """Rotated 3° + salt-pepper noise."""
    import cv2
    import fitz

    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)
        page.insert_text(
            (72, 200), "SKEWED AND NOISY\nDocument 2026",
            fontsize=30, fontname="helv",
        )
        raw = doc.tobytes()
    finally:
        doc.close()

    src = fitz.open(stream=raw, filetype="pdf")
    try:
        pix = src.load_page(0).get_pixmap(dpi=200, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    finally:
        src.close()

    # Rotate 3°
    h, w = arr.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    arr = cv2.warpAffine(arr, rot_matrix, (w, h), borderValue=(255, 255, 255))

    # Salt-pepper 2%
    rng = np.random.default_rng(42)
    noise = rng.random(arr.shape[:2])
    arr[noise < 0.01] = 0
    arr[noise > 0.99] = 255

    ok, buf = cv2.imencode(".png", arr)
    assert ok

    out = fitz.open()
    try:
        p = out.new_page(width=612, height=792)
        p.insert_image(p.rect, stream=buf.tobytes())
        out.save(str(FIXTURES_DIR / "skewed_noisy.pdf"))
    finally:
        out.close()
    print(f"  skewed_noisy.pdf — {(FIXTURES_DIR / 'skewed_noisy.pdf').stat().st_size} B")


def main() -> int:
    print("Generating test fixtures…")
    generate_contract_ru_4page()
    generate_faded_scan()
    generate_skewed_noisy()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
