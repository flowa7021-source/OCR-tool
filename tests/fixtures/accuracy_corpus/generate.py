"""Ground-truth accuracy corpus generator.

Produces PDF + matching ``.txt`` (the "truth") pairs under
``tests/fixtures/accuracy_corpus/``. Unlike
``tests/fixtures/generate_fixtures.py`` — which deliberately renders
*adversarial* pages to stress the retry tiers — this corpus targets
**clean realistic scans** where Tesseract should achieve high
accuracy. It's the yardstick we measure preprocessing and
post-processing changes against.

Design notes:

* Each document is a synthetic image-only PDF so OCRmyPDF actually
  has to OCR it (a text-selectable PDF short-circuits the pipeline).
* Rasterisation DPI (300) is deliberately moderate — high enough that
  Tesseract can read 10-12pt body, low enough that a realistic scan
  quality is simulated.
* Text is rendered via PyMuPDF's built-in ``helv`` for Latin and
  DejaVu Sans for Cyrillic, both of which Tesseract's LSTM handles
  cleanly — so baseline CER should be ≤ 3-5% on a well-tuned
  profile.
* Every generator function returns the exact ground-truth string
  that was rendered, so the benchmark can diff recognised text
  against it without having to hand-maintain ``.txt`` files.

Regenerate on demand:
    python -m tests.fixtures.accuracy_corpus.generate
    # or
    pytest tests/integration/test_accuracy_benchmark.py --regenerate-corpus
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CorpusDocument:
    """A single corpus entry: PDF + its ground-truth text."""

    name: str
    pdf_path: Path
    ground_truth: str
    description: str


def _find_cyrillic_font() -> Path | None:
    """Locate a system TTF that renders Cyrillic.

    PyMuPDF's built-in ``helv`` is Latin-only — rendering ``Привет``
    through it produces glyphs Tesseract can't read. Callers that
    need Cyrillic pick up a Unicode-capable font via this helper and
    skip gracefully when none is available.
    """
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        p = Path(candidate)
        if p.is_file():
            return p
    return None


def _rasterise_to_image_pdf(
    text_pdf_bytes: bytes, dest: Path, dpi: int = 300,
) -> None:
    """Convert a selectable-text PDF to an image-only PDF.

    Pipeline tests need image-only inputs so OCRmyPDF's
    ``skip_text=True`` short-circuit doesn't pass the user's own
    text through unchanged.
    """
    import fitz

    scan = fitz.open()
    src = fitz.open(stream=text_pdf_bytes, filetype="pdf")
    try:
        for page in src:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            new = scan.new_page(
                width=page.rect.width, height=page.rect.height,
            )
            new.insert_image(new.rect, stream=pix.tobytes("png"))
        scan.save(str(dest))
    finally:
        scan.close()
        src.close()


def _render_text_pdf(
    text: str,
    dest: Path,
    *,
    cyrillic: bool,
    fontsize: int = 14,
    dpi: int = 300,
) -> None:
    """Render ``text`` to a rasterised image-only PDF."""
    import fitz

    font_file: Path | None = None
    if cyrillic:
        font_file = _find_cyrillic_font()
        if font_file is None:
            raise RuntimeError(
                "Cyrillic corpus document requested but no system "
                "font with Cyrillic coverage is installed."
            )

    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)  # US Letter
        kwargs: dict = {"fontsize": fontsize}
        if font_file is not None:
            kwargs["fontfile"] = str(font_file)
            kwargs["fontname"] = "CyrFont"
        else:
            kwargs["fontname"] = "helv"
        rect = fitz.Rect(72, 72, 540, 720)  # 1 inch margins
        page.insert_textbox(rect, text, **kwargs)
        raw = doc.tobytes()
    finally:
        doc.close()

    _rasterise_to_image_pdf(raw, dest, dpi=dpi)


# ---------------------------------------------------------------------------
# Corpus documents — each returns a :class:`CorpusDocument`
# ---------------------------------------------------------------------------


def _english_business_letter() -> CorpusDocument:
    """Clean English business letter — the easiest baseline."""
    truth = (
        "Dear Mr Johnson\n\n"
        "Thank you for your order number 12345 dated "
        "March 15 2024. We confirm receipt of payment in the "
        "amount of 1250 dollars and 00 cents including VAT.\n\n"
        "Your shipment will be dispatched within 3 business days "
        "and should arrive no later than March 22 2024. A tracking "
        "number will be sent to your registered email address once "
        "the parcel leaves our warehouse.\n\n"
        "Please do not hesitate to contact our customer service "
        "team if you have any questions regarding this order or "
        "our products and services.\n\n"
        "Sincerely,\nAlpha Corporation Limited"
    )
    dest = CORPUS_DIR / "en_business_letter.pdf"
    _render_text_pdf(truth, dest, cyrillic=False, fontsize=14)
    return CorpusDocument(
        name="en_business_letter",
        pdf_path=dest,
        ground_truth=truth,
        description="Clean English prose — baseline for English accuracy",
    )


def _russian_business_letter() -> CorpusDocument:
    """Clean Russian business letter — baseline for Russian accuracy."""
    truth = (
        "Уважаемый Иван Иванович\n\n"
        "Благодарим за заказ номер 12345 от "
        "15 марта 2024 года. Подтверждаем получение оплаты в "
        "размере 1250 рублей 00 копеек с учётом налога на "
        "добавленную стоимость.\n\n"
        "Отгрузка товара будет произведена в течение трёх "
        "рабочих дней и должна быть получена не позднее "
        "22 марта 2024 года. Номер для отслеживания посылки "
        "будет отправлен на указанный адрес электронной почты.\n\n"
        "Если у вас возникнут вопросы по данному заказу или "
        "нашей продукции просьба обращаться в отдел поддержки "
        "клиентов.\n\n"
        "С уважением\nОбщество с ограниченной ответственностью Альфа"
    )
    dest = CORPUS_DIR / "ru_business_letter.pdf"
    _render_text_pdf(truth, dest, cyrillic=True, fontsize=14)
    return CorpusDocument(
        name="ru_business_letter",
        pdf_path=dest,
        ground_truth=truth,
        description="Clean Russian prose — baseline for Cyrillic accuracy",
    )


def _mixed_language_memo() -> CorpusDocument:
    """Russian memo with embedded English technical terms.

    This is the document shape where the old Tesseract setup confused
    Latin/Cyrillic look-alikes. Baseline captures how the word-level
    fixup handles the boundary cases.
    """
    truth = (
        "Служебная записка\n\n"
        "Отдел разработки использует следующие инструменты "
        "OpenCV, PyMuPDF, Tesseract и Ghostscript для построения "
        "пайплайна распознавания текста. Все компоненты открыты "
        "и распространяются под лицензиями Apache 2.0 и GPL.\n\n"
        "По результатам тестирования точность OCR на русском "
        "языке составляет около 97 процентов а на английском "
        "около 99 процентов. Обработка одной страницы формата A4 "
        "занимает от 30 до 90 секунд в зависимости от разрешения "
        "и сложности макета.\n\n"
        "Прошу согласовать выделение дополнительных серверных "
        "мощностей для ночной обработки корпуса из 500 документов."
    )
    dest = CORPUS_DIR / "ru_en_mixed_memo.pdf"
    _render_text_pdf(truth, dest, cyrillic=True, fontsize=13)
    return CorpusDocument(
        name="ru_en_mixed_memo",
        pdf_path=dest,
        ground_truth=truth,
        description="Russian body with English tech terms — tests the "
        "Latin/Cyrillic disambiguation pass",
    )


def _russian_invoice_like() -> CorpusDocument:
    """Russian invoice body — numbers + units + abbreviations.

    Targets user-words enrichment: ``руб``, ``шт``, ``кг``, ``ИНН``,
    ``КПП``, ``ОГРН`` should boost recognition confidence.
    """
    truth = (
        "Счёт на оплату номер 1247 от 15 марта 2024 года\n\n"
        "Поставщик ООО Альфа ИНН 7712345678 "
        "КПП 771201001 ОГРН 1027700000001\n\n"
        "Плательщик ООО Бета ИНН 7798765432 "
        "КПП 779801002 ОГРН 1037700000002\n\n"
        "Основание Договор номер 15 от 01 марта 2024 года\n\n"
        "Позиция 1 Кабель силовой ВВГнг три на два запятая пять "
        "мм квадрат 150 м по 85 рублей 50 копеек за метр итого "
        "12 825 рублей 00 копеек.\n\n"
        "Позиция 2 Розетка с заземлением белая 24 шт по "
        "145 рублей 00 копеек за штуку итого 3 480 рублей.\n\n"
        "Итого к оплате 16 305 рублей включая налог на "
        "добавленную стоимость 20 процентов в размере "
        "2 717 рублей 50 копеек."
    )
    dest = CORPUS_DIR / "ru_invoice_body.pdf"
    _render_text_pdf(truth, dest, cyrillic=True, fontsize=13)
    return CorpusDocument(
        name="ru_invoice_body",
        pdf_path=dest,
        ground_truth=truth,
        description="Russian invoice body — tests unit/abbreviation accuracy",
    )


def _dense_small_russian() -> CorpusDocument:
    """Dense 9pt Russian text — stress test for small-font recognition."""
    truth = (
        "Настоящим подтверждается что товарно-материальные "
        "ценности указанные в товарной накладной приняты "
        "грузополучателем в полном объёме претензий по "
        "количеству и качеству не имеется упаковка не "
        "нарушена пломбы целы сопроводительные документы "
        "оформлены надлежащим образом в соответствии с "
        "требованиями действующего законодательства "
        "Российской Федерации и условий заключённого "
        "договора поставки. Товар передан в технически "
        "исправном состоянии пригодном для целевого "
        "использования по назначению. Стороны подтверждают "
        "отсутствие взаимных претензий по исполнению "
        "обязательств на дату подписания настоящего акта."
    )
    dest = CORPUS_DIR / "ru_dense_small.pdf"
    _render_text_pdf(truth, dest, cyrillic=True, fontsize=10)
    return CorpusDocument(
        name="ru_dense_small",
        pdf_path=dest,
        ground_truth=truth,
        description="9pt Russian body — small-font stress",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


#: Ordered tuple of (name, generator_callable). The benchmark iterates
#: this list in order to produce reproducible per-document results.
CORPUS_GENERATORS: tuple[tuple[str, callable], ...] = (
    ("en_business_letter", _english_business_letter),
    ("ru_business_letter", _russian_business_letter),
    ("ru_en_mixed_memo", _mixed_language_memo),
    ("ru_invoice_body", _russian_invoice_like),
    ("ru_dense_small", _dense_small_russian),
)


def generate_corpus(force: bool = False) -> list[CorpusDocument]:
    """Build all corpus entries. Idempotent — regenerates only on ``force``.

    Returns:
        List of :class:`CorpusDocument` in deterministic order.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    docs: list[CorpusDocument] = []
    for name, gen in CORPUS_GENERATORS:
        pdf_path = CORPUS_DIR / f"{name}.pdf"
        if pdf_path.exists() and not force:
            # Load the ground truth from the cached ``.txt`` sibling.
            truth_path = CORPUS_DIR / f"{name}.txt"
            if truth_path.exists():
                docs.append(
                    CorpusDocument(
                        name=name,
                        pdf_path=pdf_path,
                        ground_truth=truth_path.read_text(encoding="utf-8"),
                        description=f"cached — {name}",
                    )
                )
                continue
        doc = gen()
        (CORPUS_DIR / f"{doc.name}.txt").write_text(
            doc.ground_truth, encoding="utf-8",
        )
        docs.append(doc)
    return docs


def main() -> int:
    print(f"Generating accuracy corpus at {CORPUS_DIR}…")
    docs = generate_corpus(force=True)
    for d in docs:
        size_kb = d.pdf_path.stat().st_size / 1024
        truth_chars = len(d.ground_truth)
        print(f"  {d.name}.pdf ({size_kb:.0f} KB, {truth_chars} chars truth)")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
