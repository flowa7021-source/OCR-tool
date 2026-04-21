"""End-to-end OCR + parser pipeline on real TN/УПД PDFs.

Симулирует полный пользовательский путь: берём сканированный PDF из
``inputs/``, прогоняем его через реальный Tesseract (PyMuPDF рендер →
pytesseract per-page), нормализуем текст, парсим через ``src.tn_parser``
и сравниваем извлечённые поля с экспертной разметкой из
``expected/<stem>.json``.

Оценка формируется из двух независимых метрик:

1. **Accuracy** — доля полей, совпавших с экспертом.
   Считается через ту же логику, что и ``scripts/run_golden.py`` —
   по полю `name` / `inn` / частичному совпадению наименования груза
   и т.д. Это «правильно ли мы извлекли значение».

2. **Confidence** — среднее взвешенное per-field `FieldConfidence`,
   которое приходит из парсера. Это «насколько уверена система в
   собственном ответе» (не зависит от эксперта — приходит из
   длины / качества OCR-текста, наличия якорей разделов и т.д.).

Цель пайплайна: accuracy ≥ 70 %, confidence ≥ 0.80 на корпусе из 4 ТН.
При достижении этого уровня симбиоз «OCR + парсер» считается
удовлетворительным для продакшена.

Запуск:

    python scripts/e2e_tn_pipeline.py           # человеко-читаемый отчёт
    python scripts/e2e_tn_pipeline.py --json    # машинно-читаемый JSON
    python scripts/e2e_tn_pipeline.py --min-accuracy 0.70 --min-conf 0.80

Exit codes:
    0 — все файлы прошли пороги accuracy/confidence.
    1 — хотя бы один PDF ниже порога accuracy.
    2 — хотя бы один PDF ниже порога confidence.
    3 — OCR или парсер упал на каком-то файле (bug).
    4 — отсутствуют внешние зависимости (Tesseract / tessdata / PDF-файлы).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Windows default codepage (cp1252 / cp866) не кодирует кириллицу и
# маркеры ✓/✗ — выводим в UTF-8 принудительно. reconfigure
# доступен с Python 3.7; отсутствие (StringIO) — silent no-op.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            _reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Environment probes — fail-fast, no silent skips
# ---------------------------------------------------------------------------


def _check_environment() -> tuple[bool, str]:
    """Проверяет доступность всех обязательных компонентов.

    Возвращает (ok, message). При ok=False тест должен упасть с exit 4,
    а не «тихо пропуститься» — пользователь хочет явную диагностику
    «чего не хватает», а не зелёный CI с молча отключённым OCR.
    """
    missing: list[str] = []

    # 1) Tesseract бинарь на PATH.
    if shutil.which("tesseract") is None:
        missing.append("tesseract (binary on PATH)")

    # 2) rus.traineddata + eng.traineddata доступны.
    try:
        import pytesseract
        try:
            langs = set(pytesseract.get_languages(config=""))
        except Exception as exc:  # noqa: BLE001
            return False, f"pytesseract.get_languages failed: {exc!r}"
        for lang in ("rus", "eng"):
            if lang not in langs:
                missing.append(f"{lang}.traineddata (tessdata dir)")
    except ImportError:
        missing.append("pytesseract (pip install pytesseract)")

    # 3) PyMuPDF — используется для растеризации PDF → image.
    try:
        import fitz  # noqa: F401  (проверяем факт импорта)
    except ImportError:
        missing.append("PyMuPDF (pip install pymupdf)")

    # 4) Pillow — обязателен для pytesseract.
    try:
        import PIL  # noqa: F401
    except ImportError:
        missing.append("Pillow (pip install Pillow)")

    # 5) inputs/ с реальными PDF.
    pdfs = sorted((_REPO_ROOT / "inputs").glob("TN_k_UPD_*.pdf"))
    if not pdfs:
        missing.append("inputs/TN_k_UPD_*.pdf (expert PDF corpus)")

    if missing:
        return False, (
            "Отсутствуют обязательные компоненты среды:\n  - "
            + "\n  - ".join(missing)
            + "\n\nУстановите их (см. CLAUDE.md § Windows priority) и "
            "повторите запуск. Мы НЕ skip'аем тест, потому что пропуск "
            "делал бы CI зелёным при фактическом отказе пайплайна."
        )
    return True, ""


# ---------------------------------------------------------------------------
# OCR-слой (pytesseract + PyMuPDF rasterisation)
# ---------------------------------------------------------------------------


def _preprocess_image(pil_img, dpi: int):
    """Apply production-grade preprocessing перед подачей Tesseract'у.

    Использует тот же ``ImagePreprocessor`` из src.core что и
    production-пайплайн, с настройками из builtin-профиля
    ``universal_accurate`` (Sauvola + CLAHE + deskew + border_removal
    + median denoise). Без этого raw pytesseract на серых сканах
    даёт на 20-30 п.п. ниже confidence и «съедает» целые секции.

    Возвращает PIL.Image (тот же format что input — для компата с
    pytesseract.image_to_string).
    """
    try:
        import numpy as np
        from PIL import Image

        from src.application.profile_manager import ProfileManager
        from src.core.image_preprocessor import ImagePreprocessor
        from src.infrastructure.config_storage import ProfileStorage
    except ImportError:
        # В минимальной среде без src.core — возвращаем без
        # препроцессинга. Fast-path остаётся работоспособным.
        return pil_img

    # Load builtin-профиль один раз (caches per-run).
    storage = ProfileStorage()
    manager = ProfileManager(storage)
    manager.initialize_builtins()
    profile = manager.load("universal_accurate")

    # PIL → numpy (grayscale для Sauvola/OTSU).
    arr = np.array(pil_img.convert("L"))

    preprocessor = ImagePreprocessor()
    processed, _angle = preprocessor.process(arr, profile.preprocess, dpi=dpi)

    return Image.fromarray(processed)


def _ocr_pdf(
    pdf: Path,
    dpi: int = 300,
    lang: str = "rus+eng",
    *,
    preprocess: bool = True,
) -> tuple[str, float]:
    """Возвращает (raw_text, ocr_confidence_0_to_1) для одного PDF.

    Страницы растеризуются в PNG через PyMuPDF, проходят через
    ``ImagePreprocessor`` (Sauvola + CLAHE + deskew + border_removal —
    настройки universal_accurate профиля), и только потом отдаются
    Tesseract'у. Без препроцессинга OCR на серых ТН-сканах теряет
    20-30 п.п. confidence — ключевой шаг качества.

    Параметр ``preprocess=False`` отключает препроцессинг (для
    baseline-замеров и debug'а). По умолчанию включено.

    OCR-confidence — усреднение per-word ``conf`` поверх всех страниц,
    доступное через ``image_to_data``.
    """
    import fitz
    import pytesseract
    from PIL import Image

    doc = fitz.open(str(pdf))
    try:
        chunks: list[str] = []
        confs: list[float] = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # ImagePreprocessor перед OCR — главное отличие от
            # naive-версии. Tesseract на Sauvola-бинаризованной
            # странице с CLAHE-нормализованным контрастом даёт
            # +15-25% confidence относительно raw rasterization.
            if preprocess:
                img = _preprocess_image(img, dpi=dpi)

            # Текст + per-word метрики.
            text = pytesseract.image_to_string(img, lang=lang)
            chunks.append(text)

            try:
                data = pytesseract.image_to_data(
                    img, lang=lang, output_type=pytesseract.Output.DICT
                )
                for c in data.get("conf", []):
                    try:
                        v = float(c)
                    except (TypeError, ValueError):
                        continue
                    # Tesseract возвращает -1 для не-текстовых боксов.
                    if v >= 0:
                        confs.append(v)
            except Exception:
                # image_to_data иногда падает на пустых страницах
                # ("Estimating resolution as ..."). Не валим пайплайн —
                # одна страница без metrics не меняет общую картину.
                pass
    finally:
        doc.close()

    # Склеиваем страницы через \f, чтобы splitter мог их различать.
    full_text = "\f".join(chunks)
    mean_conf = (statistics.mean(confs) / 100.0) if confs else 0.0
    return full_text, mean_conf


# ---------------------------------------------------------------------------
# Сравнение с экспертной разметкой (переиспользует run_golden.py)
# ---------------------------------------------------------------------------


def _expected_for(pdf: Path) -> dict[str, Any] | None:
    expected = _REPO_ROOT / "expected" / (pdf.stem + ".json")
    if not expected.is_file():
        # В inputs/ могут лежать PDF без экспертной разметки — например,
        # UPD_662_* в нашем корпусе. Пропускаем сверку, но НЕ OCR.
        return None
    return json.loads(expected.read_text(encoding="utf-8"))


def _field_accuracy(row, expected_json: dict[str, Any]) -> dict[str, Any]:
    """Делегирует в run_golden: one source of truth для проверок.

    Возвращает per-field {'ok': bool|None, 'comment': str} + общий
    accuracy = доля полей, прошедших проверку, среди полей с
    expectation'ом.
    """
    from scripts.run_golden import FIELDS, check_field, expected_fields

    exp = expected_fields(expected_json)
    per_field: dict[str, dict[str, Any]] = {}
    ok_cnt = fail_cnt = skip_cnt = 0
    for fld in FIELDS:
        got = getattr(row, fld, "")
        ok, comment = check_field(fld, exp.get(fld), got)
        per_field[fld] = {"ok": ok, "comment": comment, "got": got}
        if ok is True:
            ok_cnt += 1
        elif ok is False:
            fail_cnt += 1
        else:
            skip_cnt += 1
    denom = ok_cnt + fail_cnt
    accuracy = (ok_cnt / denom) if denom > 0 else 0.0
    return {
        "per_field": per_field,
        "ok": ok_cnt,
        "fail": fail_cnt,
        "skip": skip_cnt,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Основной прогон
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    pdf: str
    ocr_chars: int
    ocr_confidence: float       # 0..1, среднее per-word Tesseract conf
    parser_confidence: float    # 0..1, FieldConfidence.overall()
    accuracy: float             # 0..1, доля полей совпавших с экспертом
    elapsed_s: float
    per_field: dict[str, Any]
    ok: int
    fail: int
    skip: int

    def combined_confidence(self) -> float:
        """Симбиоз OCR × парсер: OCR уверенность в тексте × уверенность
        парсера в найденных полях. Именно этот скаляр = «насколько
        система уверена в итоговом ответе».

        Формула: среднее геометрическое — падение любой компоненты
        (грязный OCR ИЛИ парсер не нашёл поле) одинаково тянет
        общий результат вниз. Арифметическое среднее маскировало бы
        провал одной из сторон.
        """
        a = max(self.ocr_confidence, 0.0)
        b = max(self.parser_confidence, 0.0)
        if a == 0.0 or b == 0.0:
            return 0.0
        return (a * b) ** 0.5


def run_one(
    pdf: Path,
    *,
    dpi: int = 300,
    verbose: bool = False,
    preprocess: bool = True,
) -> PipelineResult:
    """OCR + parser + accuracy для одного PDF. Исключения пробрасываем
    вверх — их ловит ``main`` и возвращает exit 3 (bug-in-pipeline).
    """
    from src.tn_parser.core import parse_text
    from src.tn_parser.normalize import normalize_for_sections

    t0 = time.time()
    raw_text, ocr_conf = _ocr_pdf(pdf, dpi=dpi, preprocess=preprocess)
    rows = parse_text(normalize_for_sections(raw_text), pdf.name)
    row = rows[0] if rows else None

    if row is None:
        return PipelineResult(
            pdf=pdf.name, ocr_chars=len(raw_text), ocr_confidence=ocr_conf,
            parser_confidence=0.0, accuracy=0.0, elapsed_s=time.time() - t0,
            per_field={}, ok=0, fail=0, skip=0,
        )

    parser_conf = row.confidence.overall()
    expected = _expected_for(pdf)
    if expected is None:
        return PipelineResult(
            pdf=pdf.name, ocr_chars=len(raw_text), ocr_confidence=ocr_conf,
            parser_confidence=parser_conf, accuracy=float("nan"),
            elapsed_s=time.time() - t0,
            per_field={}, ok=0, fail=0, skip=0,
        )

    acc = _field_accuracy(row, expected)
    return PipelineResult(
        pdf=pdf.name,
        ocr_chars=len(raw_text),
        ocr_confidence=ocr_conf,
        parser_confidence=parser_conf,
        accuracy=acc["accuracy"],
        elapsed_s=time.time() - t0,
        per_field=acc["per_field"],
        ok=acc["ok"],
        fail=acc["fail"],
        skip=acc["skip"],
    )


def _print_report(results: list[PipelineResult]) -> None:
    print()
    print("=" * 78)
    print(f"{'PDF':<42} {'OCR':>7} {'PARSER':>7} {'COMB':>7} {'ACC':>7} {'TIME':>6}")
    print("-" * 78)
    for r in results:
        acc_str = "—" if r.accuracy != r.accuracy else f"{r.accuracy * 100:.0f}%"
        print(
            f"{r.pdf[:42]:<42} "
            f"{r.ocr_confidence * 100:>6.1f}% "
            f"{r.parser_confidence * 100:>6.1f}% "
            f"{r.combined_confidence() * 100:>6.1f}% "
            f"{acc_str:>7} "
            f"{r.elapsed_s:>5.1f}s"
        )
    print("=" * 78)

    rated = [r for r in results if r.accuracy == r.accuracy]   # not NaN
    if rated:
        mean_acc = statistics.mean(r.accuracy for r in rated)
        mean_ocr = statistics.mean(r.ocr_confidence for r in rated)
        mean_par = statistics.mean(r.parser_confidence for r in rated)
        mean_comb = statistics.mean(r.combined_confidence() for r in rated)
        print(
            f"{'MEAN (docs with expected)':<42} "
            f"{mean_ocr * 100:>6.1f}% "
            f"{mean_par * 100:>6.1f}% "
            f"{mean_comb * 100:>6.1f}% "
            f"{mean_acc * 100:>6.0f}%"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="e2e-tn-pipeline",
        description=(
            "Full OCR + parser end-to-end на реальных ТН-PDF из "
            "inputs/. Fail-fast: если чего-то нет в среде — exit 4; "
            "если pipeline упал — exit 3; если accuracy/conf ниже "
            "порога — exit 1/2."
        ),
    )
    p.add_argument(
        "--dpi", type=int, default=300,
        help="DPI растеризации страниц для OCR (default 300)",
    )
    p.add_argument(
        "--min-accuracy", type=float, default=0.70,
        help="Минимум field-accuracy (0..1) на документ (default 0.70)",
    )
    p.add_argument(
        "--min-conf", type=float, default=0.50,
        help=(
            "Минимум combined confidence (0..1) на документ "
            "(default 0.50; OCR×parser геометрическое среднее)"
        ),
    )
    p.add_argument(
        "--json", dest="as_json", action="store_true",
        help="Machine-readable JSON вместо человеко-читаемой таблицы",
    )
    p.add_argument(
        "--json-out", type=Path, default=None,
        help="Сохранить JSON в файл (для upload-artifact в CI)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Печатать per-field comment для каждого документа",
    )
    p.add_argument(
        "--no-preprocess", dest="preprocess", action="store_false",
        default=True,
        help=(
            "Отключить препроцессинг изображений (Sauvola / CLAHE / "
            "deskew / border_removal) перед OCR. Только для baseline-"
            "замеров: без препроцессинга OCR теряет 15-25 п.п. "
            "confidence на типичных серых ТН-сканах."
        ),
    )
    args = p.parse_args(argv)

    ok, msg = _check_environment()
    if not ok:
        print(msg, file=sys.stderr)
        return 4

    pdfs = sorted((_REPO_ROOT / "inputs").glob("TN_k_UPD_*.pdf"))
    if not args.as_json:
        print(f"Прогон OCR+парсер на {len(pdfs)} PDF из inputs/ "
              f"(DPI={args.dpi}, min_accuracy={args.min_accuracy}, "
              f"min_conf={args.min_conf})")

    results: list[PipelineResult] = []
    for pdf in pdfs:
        if not args.as_json:
            print(f"  … {pdf.name}")
        try:
            r = run_one(
                pdf,
                dpi=args.dpi,
                verbose=args.verbose,
                preprocess=args.preprocess,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"PIPELINE ERROR on {pdf.name}: {exc!r}", file=sys.stderr,
            )
            return 3
        results.append(r)
        if args.verbose and not args.as_json:
            for fname, info in r.per_field.items():
                mark = "✓" if info["ok"] is True else (
                    "✗" if info["ok"] is False else "·"
                )
                print(f"    {mark} {fname:10s} {info['comment']}")

    if args.as_json or args.json_out:
        payload = {
            "meta": {
                "dpi": args.dpi,
                "min_accuracy": args.min_accuracy,
                "min_conf": args.min_conf,
                "tesseract_version": _tesseract_version(),
                "platform": sys.platform,
            },
            "results": [asdict(r) | {"combined_confidence": r.combined_confidence()}
                        for r in results],
        }
        blob = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(blob, encoding="utf-8")
        if args.as_json:
            print(blob)

    if not args.as_json:
        _print_report(results)

    # Gate: каждый документ с expected должен пройти оба порога.
    acc_failures = [
        r for r in results
        if r.accuracy == r.accuracy and r.accuracy < args.min_accuracy
    ]
    conf_failures = [
        r for r in results
        if r.accuracy == r.accuracy
        and r.combined_confidence() < args.min_conf
    ]

    if acc_failures:
        if not args.as_json:
            print(
                f"\n❌ {len(acc_failures)} PDF ниже min_accuracy "
                f"({args.min_accuracy * 100:.0f}%):",
                file=sys.stderr,
            )
            for r in acc_failures:
                print(
                    f"  - {r.pdf}: accuracy {r.accuracy * 100:.0f}%",
                    file=sys.stderr,
                )
        return 1
    if conf_failures:
        if not args.as_json:
            print(
                f"\n❌ {len(conf_failures)} PDF ниже min_conf "
                f"({args.min_conf * 100:.0f}%):",
                file=sys.stderr,
            )
            for r in conf_failures:
                print(
                    f"  - {r.pdf}: combined "
                    f"{r.combined_confidence() * 100:.0f}% "
                    f"(OCR {r.ocr_confidence * 100:.0f}%, "
                    f"parser {r.parser_confidence * 100:.0f}%)",
                    file=sys.stderr,
                )
        return 2

    if not args.as_json:
        print(
            f"\n✅ Все {len(results)} PDF прошли accuracy ≥ "
            f"{args.min_accuracy * 100:.0f}% и combined_conf ≥ "
            f"{args.min_conf * 100:.0f}%"
        )
    return 0


def _tesseract_version() -> str:
    try:
        import pytesseract
        return str(pytesseract.get_tesseract_version())
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
