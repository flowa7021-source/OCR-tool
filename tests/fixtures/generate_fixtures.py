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

  * **invoice_typical_form.pdf** — Typical Russian накладная (waybill)
    form with a dense multi-cell grid, headers, a signatures block and
    small 8-9pt text. Matches the shape of the user's real failing
    document without using their data.

  * **dense_small_text.pdf** — Full page of 7pt Russian body text with
    no layout structure. Stresses Tesseract's layout analyser; on
    real scans this shape is the #1 cause of timeout-style crashes.

  * **stamped_page.pdf** — Page with a circular rotated stamp overlay
    over the body text plus a signature line. Stamps historically
    crash ``PSM.AUTO`` layout analysis — this fixture is the canary.

  * **rotated_table.pdf** — A bordered 5-column table rotated ~8°
    past the typical auto-deskew threshold. The retry-tier at
    ``PSM.SINGLE_BLOCK`` / ``PSM.SPARSE_TEXT`` should still recover it.

Run this script once to populate ``tests/fixtures/``. The generated
PDFs are committed to the repo so CI doesn't need to regenerate them.

Usage:
    python tests/fixtures/generate_fixtures.py                # baseline set
    python tests/fixtures/generate_fixtures.py --nightly      # + 20 corpus docs
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


def generate_invoice_typical_form() -> None:
    """Typical Russian накладная (waybill) form — dense grid + signatures.

    Models the user's production failure case without using their PII.
    Layout characteristics that historically crash Tesseract:
      * Many narrow cells in a bordered grid (layout analyser
        mis-segments)
      * 8-9pt text inside cells (sub-pixel anti-aliasing on 300 DPI
        scans confuses the LSTM)
      * Mixed numeric + Cyrillic in adjacent cells
      * Signature block with horizontal rules and small text below
    """
    import fitz

    font = _find_cyrillic_font()
    # English fallback still exercises the grid-layout-crash signal;
    # only the Cyrillic glyph signal is lost when no system font is
    # available.
    use_cyrillic = font is not None

    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)
        kwargs: dict = {"fontsize": 10}
        if use_cyrillic and font is not None:
            kwargs["fontfile"] = str(font)
            kwargs["fontname"] = "CyrFont"
        else:
            kwargs["fontname"] = "helv"

        # Header
        header = (
            "ТОВАРНАЯ НАКЛАДНАЯ № 1247/2024\nот 15.03.2024 г."
            if use_cyrillic
            else "WAYBILL No 1247/2024\nDated 15.03.2024"
        )
        page.insert_text((50, 50), header, fontsize=14, **{
            k: v for k, v in kwargs.items() if k != "fontsize"
        })

        # Parties block
        parties = (
            "Грузоотправитель: ООО «Склад-Логистик», ИНН 7712345678\n"
            "Грузополучатель: ООО «Розница-Опт», ИНН 7798765432\n"
            "Плательщик: тот же\nОснование: Договор № 15 от 01.03.2024"
            if use_cyrillic
            else "Shipper: Warehouse-Logistics LLC, TIN 7712345678\n"
            "Consignee: Retail-Whsale LLC, TIN 7798765432\n"
            "Payer: same\nBasis: Contract No 15 of 01.03.2024"
        )
        page.insert_text((50, 110), parties, **kwargs)

        # Grid table: 5 rows × 6 columns of small cells
        col_x = [50, 110, 230, 330, 400, 470, 560]  # 6 columns
        row_y = [210, 240, 270, 300, 330, 360]       # 5 data rows + header
        # Draw grid lines
        for x in col_x:
            page.draw_line(fitz.Point(x, row_y[0]), fitz.Point(x, row_y[-1]))
        for y in row_y:
            page.draw_line(fitz.Point(col_x[0], y), fitz.Point(col_x[-1], y))

        # Header row
        headers_ru = ["№", "Наименование товара", "Ед.", "Кол-во", "Цена", "Сумма"]
        headers_en = ["#", "Product name", "Unit", "Qty", "Price", "Total"]
        headers = headers_ru if use_cyrillic else headers_en
        for x, h in zip(col_x[:-1], headers, strict=False):
            page.insert_text((x + 2, row_y[0] + 12), h, fontsize=8, **{
                k: v for k, v in kwargs.items() if k != "fontsize"
            })

        # 4 data rows with narrow-cell contents
        rows_ru = [
            ["1", "Кабель ВВГнг 3×2,5", "м", "150,0", "85,50", "12 825,00"],
            ["2", "Розетка с з/к белая", "шт", "24", "145,00", "3 480,00"],
            ["3", "Автомат АВВ 16А 1Р", "шт", "12", "320,50", "3 846,00"],
            ["4", "Светильник LED 36Вт", "шт", "8", "1 250,00", "10 000,00"],
        ]
        rows_en = [
            ["1", "Cable VVGng 3x2.5", "m", "150.0", "85.50", "12,825.00"],
            ["2", "Socket w/ground wht", "pc", "24", "145.00", "3,480.00"],
            ["3", "Circuit breaker 16A", "pc", "12", "320.50", "3,846.00"],
            ["4", "LED lamp 36W", "pc", "8", "1,250.00", "10,000.00"],
        ]
        rows = rows_ru if use_cyrillic else rows_en
        for row_idx, row in enumerate(rows, start=1):
            y = row_y[row_idx] + 12
            for col_idx, cell in enumerate(row):
                x = col_x[col_idx] + 2
                page.insert_text((x, y), cell, fontsize=8, **{
                    k: v for k, v in kwargs.items() if k != "fontsize"
                })

        # Totals block
        total_txt = (
            "ИТОГО: 30 151,00 руб.\nВ т.ч. НДС 20%: 5 025,17 руб.\n"
            "Всего мест: 4. Масса брутто: 145,8 кг."
            if use_cyrillic
            else "TOTAL: 30,151.00 RUB\nincl. VAT 20%: 5,025.17 RUB\n"
            "Total packages: 4. Gross weight: 145.8 kg."
        )
        page.insert_text((50, 400), total_txt, **kwargs)

        # Signatures block with horizontal rules
        sig_txt = (
            "Отпустил: ____________ /Сидоров С.С./   М.П.\n\n"
            "Принял:   ____________ /Козлов К.К./    М.П.\n\n"
            "Накладная составлена в двух экземплярах."
            if use_cyrillic
            else "Released by: ____________ /Sidorov/   seal\n\n"
            "Received:   ____________ /Kozlov/      seal\n\n"
            "Waybill in two copies of equal force."
        )
        page.insert_text((50, 480), sig_txt, **kwargs)

        raw = doc.tobytes()
    finally:
        doc.close()

    # Rasterise at 200 DPI to simulate a typical scan
    _rasterise_to_image_pdf(raw, FIXTURES_DIR / "invoice_typical_form.pdf")
    out = FIXTURES_DIR / "invoice_typical_form.pdf"
    print(f"  invoice_typical_form.pdf — {out.stat().st_size} B")


def generate_dense_small_text() -> None:
    """Full page of 7pt Russian body text — no structure, just dense lines."""
    import fitz

    font = _find_cyrillic_font()
    use_cyrillic = font is not None

    body_ru = (
        "Настоящим подтверждается, что товарно-материальные ценности, "
        "указанные в товарной накладной, приняты грузополучателем в "
        "полном объёме, претензий по количеству и качеству не имеется. "
        "Упаковка не нарушена, пломбы целы, сопроводительные документы "
        "оформлены надлежащим образом в соответствии с требованиями "
        "действующего законодательства Российской Федерации. "
    ) * 15
    body_en = (
        "This confirms that the goods listed in the waybill have been "
        "accepted by the consignee in full, with no claims regarding "
        "quantity or quality. Packaging is intact, seals are undamaged, "
        "shipping documents are executed in accordance with applicable "
        "law. "
    ) * 15

    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)
        kwargs: dict = {"fontsize": 7}
        if use_cyrillic and font is not None:
            kwargs["fontfile"] = str(font)
            kwargs["fontname"] = "CyrFont"
        else:
            kwargs["fontname"] = "helv"
        # Insert textbox so PyMuPDF wraps naturally
        rect = fitz.Rect(50, 50, 562, 742)
        page.insert_textbox(
            rect, body_ru if use_cyrillic else body_en, **kwargs
        )
        raw = doc.tobytes()
    finally:
        doc.close()

    _rasterise_to_image_pdf(raw, FIXTURES_DIR / "dense_small_text.pdf")
    out = FIXTURES_DIR / "dense_small_text.pdf"
    print(f"  dense_small_text.pdf — {out.stat().st_size} B")


def generate_stamped_page() -> None:
    """Page with body text plus a circular rotated stamp overlay.

    Stamps crash ``PSM.AUTO`` layout analysis because the circular
    shape confuses Tesseract's text-line detector. Our retry tier
    using ``PSM.SPARSE_TEXT`` / ``PSM.SINGLE_BLOCK`` should still
    recover at least the non-stamp body text.
    """
    import cv2
    import fitz

    font = _find_cyrillic_font()
    use_cyrillic = font is not None

    # Base document
    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)
        body = (
            "АКТ ПРИЁМА-ПЕРЕДАЧИ\nот 18.04.2024\n\n"
            "Стороны подтверждают передачу оборудования\n"
            "согласно спецификации № 42 от 10.04.2024.\n\n"
            "Передал: Иванов И.И.  ____________\n"
            "Принял:  Петров П.П.  ____________"
            if use_cyrillic
            else "ACCEPTANCE ACT\nDated 18.04.2024\n\n"
            "Parties confirm equipment transfer per\n"
            "specification No 42 dated 10.04.2024.\n\n"
            "Released: Smith J.  ____________\n"
            "Accepted: Doe J.    ____________"
        )
        kwargs: dict = {"fontsize": 14}
        if use_cyrillic and font is not None:
            kwargs["fontfile"] = str(font)
            kwargs["fontname"] = "CyrFont"
        else:
            kwargs["fontname"] = "helv"
        page.insert_text((50, 100), body, **kwargs)
        raw = doc.tobytes()
    finally:
        doc.close()

    # Rasterise and overlay a simulated stamp
    import numpy as np

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

    h, w = arr.shape[:2]
    # Draw a circular stamp — two concentric circles + diagonal text
    cx, cy, radius = int(w * 0.75), int(h * 0.7), 110
    cv2.circle(arr, (cx, cy), radius, (0, 0, 150), 3)
    cv2.circle(arr, (cx, cy), radius - 15, (0, 0, 150), 2)

    # Rotated text inside the stamp (simulated by warping text overlay)
    overlay = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(
        overlay, "OOO ALFA-STROY", (cx - 95, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 150), 2,
    )
    cv2.putText(
        overlay, "INN 7712345678", (cx - 85, cy + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1,
    )
    # Rotate the stamp overlay ~15° for realism
    rot = cv2.getRotationMatrix2D((cx, cy), 15, 1.0)
    overlay = cv2.warpAffine(
        overlay, rot, (w, h), borderValue=(255, 255, 255)
    )
    # Multiply: stamp ink "darkens" underlying pixels only where overlay
    # has ink (non-white).
    mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) < 240
    arr[mask] = overlay[mask]

    ok, buf = cv2.imencode(".png", arr)
    assert ok

    out = fitz.open()
    try:
        p = out.new_page(width=612, height=792)
        p.insert_image(p.rect, stream=buf.tobytes())
        out.save(str(FIXTURES_DIR / "stamped_page.pdf"))
    finally:
        out.close()
    print(
        f"  stamped_page.pdf — "
        f"{(FIXTURES_DIR / 'stamped_page.pdf').stat().st_size} B"
    )


def generate_rotated_table() -> None:
    """Bordered 5-column table rotated ~8° past auto-deskew threshold.

    Our universal_accurate profile caps auto-deskew at ±5°; an 8°
    rotation survives preprocessing and reaches the OCR engine still
    tilted. Retry-tier PSMs (``SINGLE_BLOCK``, ``SPARSE_TEXT``) must
    still recover at least some text from the table cells.
    """
    import cv2
    import fitz
    import numpy as np

    font = _find_cyrillic_font()
    use_cyrillic = font is not None

    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)
        # 5-column table
        col_x = [50, 150, 270, 370, 470, 560]
        row_y = list(range(150, 450, 30))
        for x in col_x:
            page.draw_line(fitz.Point(x, row_y[0]), fitz.Point(x, row_y[-1]))
        for y in row_y:
            page.draw_line(fitz.Point(col_x[0], y), fitz.Point(col_x[-1], y))

        kwargs: dict = {"fontsize": 10}
        if use_cyrillic and font is not None:
            kwargs["fontfile"] = str(font)
            kwargs["fontname"] = "CyrFont"
        else:
            kwargs["fontname"] = "helv"

        headers = (
            ["Дата", "Операция", "Приход", "Расход", "Остаток"]
            if use_cyrillic
            else ["Date", "Operation", "In", "Out", "Balance"]
        )
        for i, h in enumerate(headers):
            page.insert_text((col_x[i] + 4, row_y[0] + 18), h, **kwargs)

        rows_ru = [
            ["01.03", "Нач. остаток", "", "", "125 000"],
            ["05.03", "Поступление", "45 000", "", "170 000"],
            ["12.03", "Списание", "", "12 500", "157 500"],
            ["18.03", "Списание", "", "35 200", "122 300"],
            ["25.03", "Поступление", "80 000", "", "202 300"],
            ["28.03", "Закрытие", "", "", "202 300"],
            ["31.03", "Сверка", "", "", "202 300"],
            ["01.04", "Перенос", "", "", "202 300"],
        ]
        rows_en = [
            ["01.03", "Opening", "", "", "125,000"],
            ["05.03", "Inflow", "45,000", "", "170,000"],
            ["12.03", "Write-off", "", "12,500", "157,500"],
            ["18.03", "Write-off", "", "35,200", "122,300"],
            ["25.03", "Inflow", "80,000", "", "202,300"],
            ["28.03", "Closing", "", "", "202,300"],
            ["31.03", "Reconc", "", "", "202,300"],
            ["01.04", "Carry", "", "", "202,300"],
        ]
        rows = rows_ru if use_cyrillic else rows_en
        for r_idx, row in enumerate(rows, start=1):
            if r_idx >= len(row_y):
                break
            for c_idx, cell in enumerate(row):
                page.insert_text(
                    (col_x[c_idx] + 4, row_y[r_idx] + 18), cell, **kwargs
                )
        raw = doc.tobytes()
    finally:
        doc.close()

    # Rasterise and rotate 8°
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

    h, w = arr.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 8.0, 1.0)
    arr = cv2.warpAffine(arr, rot, (w, h), borderValue=(255, 255, 255))
    ok, buf = cv2.imencode(".png", arr)
    assert ok

    out = fitz.open()
    try:
        p = out.new_page(width=612, height=792)
        p.insert_image(p.rect, stream=buf.tobytes())
        out.save(str(FIXTURES_DIR / "rotated_table.pdf"))
    finally:
        out.close()
    print(
        f"  rotated_table.pdf — "
        f"{(FIXTURES_DIR / 'rotated_table.pdf').stat().st_size} B"
    )


def generate_nightly_corpus() -> None:
    """Generate 20 additional adversarial-combination fixtures for nightly CI.

    Each fixture is seeded deterministically so runs are reproducible.
    Combinations cover the cross-product of (base-shape × damage):

      * 5 invoice/waybill variations with different totals/parties
      * 5 stamped pages at different stamp positions + rotations
      * 5 rotated tables at skew angles 3°-12° + light noise
      * 5 faded + noisy combined (low contrast + salt-pepper)

    Stored under ``tests/fixtures/nightly/*.pdf``. Nightly CI runs all
    of them through every Tesseract profile and asserts no crashes.
    """
    nightly_dir = FIXTURES_DIR / "nightly"
    nightly_dir.mkdir(exist_ok=True)

    _generate_invoice_variations(nightly_dir, count=5)
    _generate_stamp_variations(nightly_dir, count=5)
    _generate_rotated_table_variations(nightly_dir, count=5)
    _generate_faded_noisy_variations(nightly_dir, count=5)

    generated = sorted(nightly_dir.glob("*.pdf"))
    total_bytes = sum(p.stat().st_size for p in generated)
    print(
        f"  nightly/: {len(generated)} docs, "
        f"{total_bytes / 1024 / 1024:.1f} MB total"
    )


def _generate_invoice_variations(out_dir: Path, *, count: int) -> None:
    """N invoice forms with shuffled totals / party names."""
    import fitz

    font = _find_cyrillic_font()
    use_cyrillic = font is not None

    for idx in range(count):
        rng = _seeded_rng(f"invoice-{idx}")
        doc_num = rng.randint(1000, 9999)
        total = rng.randint(5000, 500000)
        vat = int(total * 0.2 / 1.2)

        doc = fitz.open()
        try:
            page = doc.new_page(width=612, height=792)
            kw = _text_kwargs(font, use_cyrillic, 10)
            hdr = (
                f"НАКЛАДНАЯ № {doc_num}/2024"
                if use_cyrillic
                else f"WAYBILL No {doc_num}/2024"
            )
            page.insert_text((50, 50), hdr, fontsize=14, **{
                k: v for k, v in kw.items() if k != "fontsize"
            })
            body = (
                f"Сумма: {total:,} руб., НДС: {vat:,} руб."
                if use_cyrillic
                else f"Total: {total:,} RUB, VAT: {vat:,} RUB"
            ).replace(",", " ")
            page.insert_text((50, 120), body, **kw)
            raw = doc.tobytes()
        finally:
            doc.close()
        _rasterise_to_image_pdf(raw, out_dir / f"invoice_{idx:02d}.pdf")


def _generate_stamp_variations(out_dir: Path, *, count: int) -> None:
    """N pages with stamps at different positions + rotations."""
    import cv2
    import fitz
    import numpy as np

    font = _find_cyrillic_font()
    use_cyrillic = font is not None

    for idx in range(count):
        rng = _seeded_rng(f"stamp-{idx}")
        doc = fitz.open()
        try:
            page = doc.new_page(width=612, height=792)
            kw = _text_kwargs(font, use_cyrillic, 12)
            body = (
                f"Документ № {rng.randint(100, 999)}\nТиповая форма"
                if use_cyrillic
                else f"Document No {rng.randint(100, 999)}\nStandard form"
            )
            page.insert_text((50, 100), body, **kw)
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

        h, w = arr.shape[:2]
        cx = rng.randint(int(w * 0.3), int(w * 0.8))
        cy = rng.randint(int(h * 0.4), int(h * 0.8))
        r = rng.randint(80, 130)
        cv2.circle(arr, (cx, cy), r, (0, 0, 150), 3)
        cv2.circle(arr, (cx, cy), r - 15, (0, 0, 150), 2)

        ok, buf = cv2.imencode(".png", arr)
        assert ok
        out = fitz.open()
        try:
            p = out.new_page(width=612, height=792)
            p.insert_image(p.rect, stream=buf.tobytes())
            out.save(str(out_dir / f"stamp_{idx:02d}.pdf"))
        finally:
            out.close()


def _generate_rotated_table_variations(out_dir: Path, *, count: int) -> None:
    """N small tables rotated 3°-12° + mild noise."""
    import cv2
    import fitz
    import numpy as np

    font = _find_cyrillic_font()
    use_cyrillic = font is not None

    for idx in range(count):
        rng = _seeded_rng(f"rot-table-{idx}")
        angle = 3.0 + idx * 2.0  # 3°, 5°, 7°, 9°, 11°
        doc = fitz.open()
        try:
            page = doc.new_page(width=612, height=792)
            kw = _text_kwargs(font, use_cyrillic, 10)
            col_x = [50, 180, 310, 440, 560]
            row_y = list(range(150, 400, 30))
            for x in col_x:
                page.draw_line(
                    fitz.Point(x, row_y[0]), fitz.Point(x, row_y[-1])
                )
            for y in row_y:
                page.draw_line(
                    fitz.Point(col_x[0], y), fitz.Point(col_x[-1], y)
                )
            for r_idx in range(1, min(len(row_y), 6)):
                for c_idx in range(4):
                    page.insert_text(
                        (col_x[c_idx] + 4, row_y[r_idx] + 18),
                        f"{rng.randint(100, 9999)}", **kw,
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
        h, w = arr.shape[:2]
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        arr = cv2.warpAffine(arr, rot, (w, h), borderValue=(255, 255, 255))
        # Light noise
        noise = np.random.default_rng(idx).random(arr.shape[:2])
        arr[noise < 0.005] = 0
        arr[noise > 0.995] = 255

        ok, buf = cv2.imencode(".png", arr)
        assert ok
        out = fitz.open()
        try:
            p = out.new_page(width=612, height=792)
            p.insert_image(p.rect, stream=buf.tobytes())
            out.save(str(out_dir / f"rot_table_{idx:02d}.pdf"))
        finally:
            out.close()


def _generate_faded_noisy_variations(out_dir: Path, *, count: int) -> None:
    """N pages combining low contrast + salt-pepper noise at varying levels."""
    import cv2
    import fitz
    import numpy as np

    font = _find_cyrillic_font()
    use_cyrillic = font is not None

    for idx in range(count):
        rng = _seeded_rng(f"faded-noisy-{idx}")
        low = 120 + idx * 5   # 120..140 foreground
        high = 200 + idx * 5  # 200..220 background
        noise_pct = 0.01 + idx * 0.005

        doc = fitz.open()
        try:
            page = doc.new_page(width=612, height=792)
            kw = _text_kwargs(font, use_cyrillic, 16)
            body = (
                f"Акт сверки № {rng.randint(100, 999)}\n"
                "Стороны подтверждают сумму расчётов."
                if use_cyrillic
                else f"Reconciliation Act No {rng.randint(100, 999)}\n"
                "Parties confirm settlement amount."
            )
            page.insert_text((60, 200), body, **kw)
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

        arr = (low + arr.astype(np.float32) / 255.0 * (high - low)).astype(
            np.uint8
        )
        rnd = np.random.default_rng(idx + 100).random(arr.shape[:2])
        arr[rnd < noise_pct / 2] = 0
        arr[rnd > 1 - noise_pct / 2] = 255

        ok, buf = cv2.imencode(".png", arr)
        assert ok
        out = fitz.open()
        try:
            p = out.new_page(width=612, height=792)
            p.insert_image(p.rect, stream=buf.tobytes())
            out.save(str(out_dir / f"faded_noisy_{idx:02d}.pdf"))
        finally:
            out.close()


def _seeded_rng(seed: str):
    """Deterministic ``random.Random`` instance for corpus reproducibility."""
    import random

    return random.Random(seed)


def _text_kwargs(font: Path | None, cyrillic: bool, fontsize: int) -> dict:
    """Build ``insert_text`` kwargs that select a Cyrillic-capable font
    when available and fall back to PyMuPDF's built-in ``helv`` otherwise."""
    if cyrillic and font is not None:
        return {
            "fontsize": fontsize,
            "fontfile": str(font),
            "fontname": "CyrFont",
        }
    return {"fontsize": fontsize, "fontname": "helv"}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Also generate the 20-doc nightly corpus under tests/fixtures/nightly/",
    )
    args = parser.parse_args()

    print("Generating baseline test fixtures…")
    generate_contract_ru_4page()
    generate_faded_scan()
    generate_skewed_noisy()
    generate_invoice_typical_form()
    generate_dense_small_text()
    generate_stamped_page()
    generate_rotated_table()
    if args.nightly:
        print("Generating nightly corpus (20 documents)…")
        generate_nightly_corpus()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
