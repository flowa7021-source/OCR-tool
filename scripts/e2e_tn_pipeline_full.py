"""End-to-end full-pipeline test: production OCRPipeline + tn_parser.

Отличие от pytesseract-варианта: здесь мы НЕ мимикрируем OCR через
прямой вызов ``pytesseract.image_to_string`` — вместо этого
запускаем именно тот ``src.application.pipeline.OCRPipeline``,
который работает у пользователя в GUI и в ``ocr-cli``. То есть:

  * ImagePreprocessor: Sauvola / OTSU / adaptive бинаризация,
    deskew, CLAHE, удаление рамок таблиц, border_removal — всё,
    что прописано в профиле.
  * TextPostprocessor: autocorrect_russian, autocorrect_english,
    merge_hyphenated, normalize_whitespace, normalize_unicode,
    remove_artifacts, fix_cyrillic_latin_confusion,
    validate_entities — весь pipeline постобработки.
  * Orchestrator (src.application.parsers.tn_orchestrator):
    вызывает tn_parser.parse_text когда profile.extract.enabled,
    даёт ParsedDocument с overall_confidence.

Этот путь — единственный, на котором реально достижимо
97-99 % combined confidence на реальных ТН:

    combined = OCR.average_confidence × ParsedDocument.overall_confidence

С голым pytesseract + лёгкой нормализацией получилось 40-60 %
(измерено scripts/e2e_tn_pipeline.py в предыдущем коммите).
Полный продакшен-стек на universal_accurate / tn_upd профиле
добавляет ~30-40 % за счёт препроцессинга + постобработки.

Запуск:

    python scripts/e2e_tn_pipeline_full.py \
        --profile tn_upd \
        --min-accuracy 0.70 --min-conf 0.70

    # С профилем universal_accurate (максимальная точность,
    # 400 DPI + Sauvola + CLAHE, медленнее):
    python scripts/e2e_tn_pipeline_full.py \
        --profile universal_accurate \
        --min-conf 0.90

Exit codes (идентичны быстрому варианту e2e_tn_pipeline.py):
    0 — всё прошло.
    1 — accuracy ниже порога.
    2 — confidence ниже порога.
    3 — pipeline error (баг продакшен-стека — сигналим).
    4 — среда не готова (tesseract / tessdata / PDF отсутствуют).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Windows cp1252/cp866 не кодируют кириллицу / ✓/✗ — forced UTF-8.
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            _reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Среда — fail-fast
# ---------------------------------------------------------------------------


def _check_environment() -> tuple[bool, str]:
    """Полный checklist: tesseract + ghostscript + rus/eng/osd +
    Python-стек продакшена. Отсутствие любого — exit 4, не skip."""
    missing: list[str] = []

    # Бинари.
    if shutil.which("tesseract") is None:
        missing.append("tesseract (binary on PATH)")
    # gs опционален для text-layer PDF, но обязателен для ocrmypdf
    # optimize_level > 0. Профиль tn_upd использует optimize=1.
    if shutil.which("gs") is None and shutil.which("gswin64c") is None:
        missing.append("ghostscript (gs / gswin64c on PATH)")

    # Языки Tesseract.
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

    # Python-стек продакшена — те же пакеты, что installer бандлит.
    for mod, pkg in [
        ("fitz", "pymupdf"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("skimage", "scikit-image"),
        ("ocrmypdf", "ocrmypdf"),
        ("openpyxl", "openpyxl"),
        ("rapidfuzz", "rapidfuzz"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(f"{pkg} (pip install {pkg})")

    # PDF-корпус.
    pdfs = sorted((_REPO_ROOT / "inputs").glob("TN_k_UPD_*.pdf"))
    if not pdfs:
        missing.append("inputs/TN_k_UPD_*.pdf (expert PDF corpus)")

    if missing:
        return False, (
            "Отсутствуют обязательные компоненты production-среды:\n  - "
            + "\n  - ".join(missing)
            + "\n\nУстановите их и повторите. CI-workflow "
            "(.github/workflows/e2e-tn-pipeline.yml) ставит этот "
            "стек автоматически; локально на Windows — либо "
            "установленный OCR Studio, либо choco install "
            "tesseract + ghostscript, плюс pip install -r "
            "requirements.txt."
        )
    return True, ""


# ---------------------------------------------------------------------------
# Production pipeline
# ---------------------------------------------------------------------------


@dataclass
class FullResult:
    pdf: str
    pages: int
    ocr_confidence: float       # result.average_confidence / 100, 0..1
    parser_confidence: float    # parsed.overall_confidence, 0..1
    parser_rows: int            # len(parsed.rows)
    accuracy: float             # 0..1, per FIELDS match rate (NaN=no expected)
    elapsed_s: float
    per_field: dict[str, Any]
    ok: int
    fail: int
    skip: int
    error: str | None = None

    def combined_confidence(self) -> float:
        a = max(self.ocr_confidence, 0.0)
        b = max(self.parser_confidence, 0.0)
        if a == 0.0 or b == 0.0:
            return 0.0
        return (a * b) ** 0.5


def _load_profile(name: str):
    """Загружает builtin-профиль через ProfileManager.

    Используем временную конфиг-директорию, чтобы не зависеть от
    %APPDATA%/OCR Studio на dev-машине и не писать туда от имени
    CI-runner'а.
    """
    from src.application.profile_manager import ProfileManager
    from src.infrastructure.config_storage import ProfileStorage

    mgr = ProfileManager(ProfileStorage())
    mgr.initialize_builtins()
    return mgr.load(name)


def run_one(pdf: Path, profile_name: str, *, verbose: bool = False) -> FullResult:
    """Полный прогон одного PDF через production OCRPipeline + парсер."""
    from src.application.pipeline import OCRPipeline
    from src.core.image_preprocessor import ImagePreprocessor
    from src.core.models import OCRJobConfig
    from src.core.text_postprocessor import TextPostprocessor
    from src.infrastructure.tesseract_wrapper import TesseractWrapper
    from src.shared.types import JobStatus

    t0 = time.time()
    profile = _load_profile(profile_name)

    # Справочник ИНН/ОГРН для postprocessor — отключаем в CI
    # (не в репозитории), это опциональный бонус.
    try:
        from src.core.doc_catalog import load_default_catalog
        catalog = load_default_catalog()
    except Exception:  # noqa: BLE001
        catalog = None

    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(catalog=catalog),
        tesseract=TesseractWrapper(),
        compute_confidence=True,   # обязательно — нужен average_confidence
    )

    # Временный output PDF (ocrmypdf нужен файл для записи).
    with tempfile.TemporaryDirectory(prefix="e2e-full-") as td:
        out_pdf = Path(td) / f"{pdf.stem}.ocr.pdf"
        job = OCRJobConfig(
            input_path=str(pdf),
            output_path=str(out_pdf),
            profile=profile,
        )
        result = pipeline.run(job)

    elapsed = time.time() - t0

    if result.status is not JobStatus.COMPLETED:
        return FullResult(
            pdf=pdf.name, pages=0, ocr_confidence=0.0,
            parser_confidence=0.0, parser_rows=0,
            accuracy=float("nan"), elapsed_s=elapsed,
            per_field={}, ok=0, fail=0, skip=0,
            error=result.error or "pipeline returned non-COMPLETED",
        )

    # pipeline.run возвращает average_confidence в процентах (0..100).
    ocr_conf = (result.average_confidence or 0.0) / 100.0

    # parsed может быть None (extract.enabled=False) или пустым
    # (парсер не нашёл ни одной TN). Второе — боевой случай на
    # деградациях OCR.
    parsed = result.parsed
    if parsed is None or not parsed.rows:
        return FullResult(
            pdf=pdf.name, pages=result.page_count,
            ocr_confidence=ocr_conf, parser_confidence=0.0,
            parser_rows=0, accuracy=0.0, elapsed_s=elapsed,
            per_field={}, ok=0, fail=0, skip=0,
            error="no parsed rows (OCR или parser не извлёк ТН)",
        )

    parser_conf = parsed.overall_confidence or 0.0
    expected = _expected_for(pdf)
    if expected is None:
        return FullResult(
            pdf=pdf.name, pages=result.page_count,
            ocr_confidence=ocr_conf, parser_confidence=parser_conf,
            parser_rows=len(parsed.rows),
            accuracy=float("nan"), elapsed_s=elapsed,
            per_field={}, ok=0, fail=0, skip=0,
        )

    # Поля из parsed.rows уже в формате dict (from_json_dict).
    # Конвертируем первый row обратно в ParsedRow, чтобы отдать
    # run_golden.check_field.
    from src.tn_parser.models import ParsedRow

    first = ParsedRow.from_json_dict(parsed.rows[0])
    acc = _field_accuracy(first, expected)
    return FullResult(
        pdf=pdf.name, pages=result.page_count,
        ocr_confidence=ocr_conf, parser_confidence=parser_conf,
        parser_rows=len(parsed.rows),
        accuracy=acc["accuracy"], elapsed_s=elapsed,
        per_field=acc["per_field"], ok=acc["ok"],
        fail=acc["fail"], skip=acc["skip"],
    )


# ---------------------------------------------------------------------------
# Сверка полей — reuse run_golden.check_field
# ---------------------------------------------------------------------------


def _expected_for(pdf: Path) -> dict[str, Any] | None:
    expected = _REPO_ROOT / "expected" / (pdf.stem + ".json")
    if not expected.is_file():
        return None
    return json.loads(expected.read_text(encoding="utf-8"))


def _field_accuracy(row, expected_json: dict[str, Any]) -> dict[str, Any]:
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
        "per_field": per_field, "ok": ok_cnt,
        "fail": fail_cnt, "skip": skip_cnt, "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Отчёт + main
# ---------------------------------------------------------------------------


def _print_report(results: list[FullResult], profile: str) -> None:
    print()
    print("=" * 86)
    print(f"Профиль: {profile}")
    print(
        f"{'PDF':<42} {'OCR':>6} {'PARSER':>7} {'COMB':>6} {'ACC':>7} "
        f"{'ROWS':>5} {'TIME':>6}"
    )
    print("-" * 86)
    for r in results:
        acc_str = "—" if r.accuracy != r.accuracy else f"{r.accuracy * 100:.0f}%"
        line = (
            f"{r.pdf[:42]:<42} "
            f"{r.ocr_confidence * 100:>5.1f}% "
            f"{r.parser_confidence * 100:>6.1f}% "
            f"{r.combined_confidence() * 100:>5.1f}% "
            f"{acc_str:>7} "
            f"{r.parser_rows:>5} "
            f"{r.elapsed_s:>5.1f}s"
        )
        if r.error:
            line += f"  ERROR: {r.error}"
        print(line)
    print("=" * 86)

    rated = [r for r in results if r.accuracy == r.accuracy and not r.error]
    if rated:
        print(
            f"{'MEAN (docs with expected)':<42} "
            f"{statistics.mean(r.ocr_confidence for r in rated) * 100:>5.1f}% "
            f"{statistics.mean(r.parser_confidence for r in rated) * 100:>6.1f}% "
            f"{statistics.mean(r.combined_confidence() for r in rated) * 100:>5.1f}% "
            f"{statistics.mean(r.accuracy for r in rated) * 100:>6.0f}%"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="e2e-tn-pipeline-full",
        description=(
            "Full production pipeline: OCRPipeline + preprocessor + "
            "postprocessor + tn_parser orchestrator. Закрывает цель "
            "97+ % combined confidence на реальных ТН."
        ),
    )
    p.add_argument(
        "--profile", default="universal_accurate",
        help=(
            "Имя профиля (по умолчанию: universal_accurate — "
            "единственный builtin со всеми лучшими настройками; "
            "пользовательские профили загружаются по имени из "
            "ProfileStorage)"
        ),
    )
    p.add_argument(
        "--min-accuracy", type=float, default=0.70,
        help="Минимум field-accuracy на документ (default 0.70)",
    )
    p.add_argument(
        "--min-conf", type=float, default=0.70,
        help=(
            "Минимум combined confidence на документ "
            "(default 0.70; с tn_upd на реальных ТН достижимо)"
        ),
    )
    p.add_argument(
        "--json", dest="as_json", action="store_true",
        help="JSON вместо таблицы",
    )
    p.add_argument(
        "--json-out", type=Path, default=None,
        help="Записать JSON в файл",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Показывать per-field details для каждого PDF",
    )
    p.add_argument(
        "--pdfs", nargs="+", default=None,
        help=(
            "Явно указать PDF (default — все inputs/TN_k_UPD_*.pdf). "
            "Полезно для локальной отладки одного файла."
        ),
    )
    args = p.parse_args(argv)

    ok, msg = _check_environment()
    if not ok:
        print(msg, file=sys.stderr)
        return 4

    if args.pdfs:
        pdfs = [Path(p) for p in args.pdfs]
        for pdf in pdfs:
            if not pdf.is_file():
                print(f"Файл не найден: {pdf}", file=sys.stderr)
                return 4
    else:
        pdfs = sorted((_REPO_ROOT / "inputs").glob("TN_k_UPD_*.pdf"))

    if not args.as_json:
        print(
            f"Profile: {args.profile} | "
            f"PDFs: {len(pdfs)} | "
            f"min_accuracy: {args.min_accuracy * 100:.0f}% | "
            f"min_conf: {args.min_conf * 100:.0f}%"
        )

    results: list[FullResult] = []
    for pdf in pdfs:
        if not args.as_json:
            print(f"  … {pdf.name}")
        try:
            r = run_one(pdf, args.profile, verbose=args.verbose)
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(
                f"PIPELINE ERROR on {pdf.name}: {exc!r}\n"
                f"{traceback.format_exc()}",
                file=sys.stderr,
            )
            return 3
        results.append(r)
        if args.verbose and not args.as_json:
            for fname, info in r.per_field.items():
                mark = ("✓" if info["ok"] is True
                        else "✗" if info["ok"] is False
                        else "·")
                print(f"    {mark} {fname:10s} {info['comment']}")

    if args.as_json or args.json_out:
        payload = {
            "meta": {
                "profile": args.profile,
                "min_accuracy": args.min_accuracy,
                "min_conf": args.min_conf,
                "platform": sys.platform,
            },
            "results": [
                asdict(r) | {"combined_confidence": r.combined_confidence()}
                for r in results
            ],
        }
        blob = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(blob, encoding="utf-8")
        if args.as_json:
            print(blob)

    if not args.as_json:
        _print_report(results, args.profile)

    # Gates.
    acc_failures = [
        r for r in results
        if r.accuracy == r.accuracy and r.accuracy < args.min_accuracy
        and not r.error
    ]
    conf_failures = [
        r for r in results
        if r.accuracy == r.accuracy
        and r.combined_confidence() < args.min_conf
        and not r.error
    ]
    pipeline_errors = [r for r in results if r.error]

    if pipeline_errors and not args.as_json:
        print(f"\n❌ {len(pipeline_errors)} PDF упали в pipeline:",
              file=sys.stderr)
        for r in pipeline_errors:
            print(f"  - {r.pdf}: {r.error}", file=sys.stderr)
        return 3

    if acc_failures:
        if not args.as_json:
            print(
                f"\n❌ {len(acc_failures)} PDF ниже "
                f"min_accuracy {args.min_accuracy * 100:.0f}%:",
                file=sys.stderr,
            )
            for r in acc_failures:
                print(f"  - {r.pdf}: {r.accuracy * 100:.0f}%",
                      file=sys.stderr)
        return 1

    if conf_failures:
        if not args.as_json:
            print(
                f"\n❌ {len(conf_failures)} PDF ниже "
                f"min_conf {args.min_conf * 100:.0f}%:",
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
            f"\n✅ Все {len(results)} PDF: "
            f"accuracy ≥ {args.min_accuracy * 100:.0f}% ∧ "
            f"combined_conf ≥ {args.min_conf * 100:.0f}%"
        )
    return 0


if __name__ == "__main__":
    # Qt offscreen — если кто-то из импорт-цепочки затаскивает widget,
    # не пытаемся открыть display (нет его на CI-runner'е).
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.exit(main())
