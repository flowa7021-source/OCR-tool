"""Command-line interface for OCR Studio.

Runs the full pipeline (preprocess → OCRmyPDF → postprocess) without
starting Qt. Useful for batch scripting, CI smoke tests, and headless
servers.

Examples:
    # Process a single file with defaults (bundled "default" profile)
    python -m src.cli document.pdf

    # Pick a profile and write the searchable PDF to a custom path
    python -m src.cli --profile low_quality_scan -o out.pdf in.pdf

    # Also dump TXT and DOCX next to the PDF
    python -m src.cli --txt --docx in.pdf

    # Process an entire directory, 4 parallel workers
    python -m src.cli --workers 4 ~/scans/

    # List available profiles
    python -m src.cli --list-profiles
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from src.shared.constants import APP_NAME, APP_VERSION

logger = logging.getLogger("ocr-cli")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argparse parser."""
    p = argparse.ArgumentParser(
        prog="ocr-studio",
        description=f"{APP_NAME} {APP_VERSION} — command-line OCR",
    )
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="PDF файлы или директории (рекурсивный поиск *.pdf)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Путь к выходному PDF (только при единственном входе). "
            "По умолчанию: рядом с оригиналом с суффиксом _ocr."
        ),
    )
    p.add_argument(
        "-p",
        "--profile",
        default="default",
        help="Имя профиля OCR (по умолчанию: default)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Количество параллельных процессов (1–4). 1 — обрабатывать в текущем процессе.",
    )
    p.add_argument(
        "--txt",
        action="store_true",
        help="Дополнительно выгрузить TXT рядом с PDF",
    )
    p.add_argument(
        "--docx",
        action="store_true",
        help="Дополнительно выгрузить DOCX рядом с PDF",
    )
    p.add_argument(
        "--list-profiles",
        action="store_true",
        help="Показать доступные профили и завершить работу",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Увеличить детализацию логов (-v = INFO, -vv = DEBUG)",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"{APP_NAME} {APP_VERSION}",
    )
    return p


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_inputs(inputs: Iterable[Path]) -> list[Path]:
    """Expand a mix of files and directories to a sorted list of *.pdf paths.

    Each candidate file goes through :func:`validate_pdf_path` so we
    skip empty or non-PDF entries with a warning instead of crashing
    deep in the pipeline.
    """
    from src.shared.validators import ValidationError, validate_pdf_path

    found: list[Path] = []
    for entry in inputs:
        entry = entry.expanduser()
        if entry.is_dir():
            for candidate in sorted(entry.rglob("*.pdf")):
                try:
                    found.append(validate_pdf_path(candidate))
                except ValidationError as exc:
                    logger.warning("Пропускаю %s: %s", candidate, exc)
        else:
            try:
                found.append(validate_pdf_path(entry))
            except ValidationError as exc:
                logger.warning("Пропускаю %s: %s", entry, exc)
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Processing orchestration
# ---------------------------------------------------------------------------


def process_single(
    input_path: Path,
    output_path: Path,
    profile_name: str,
    want_txt: bool,
    want_docx: bool,
) -> int:
    """Run the pipeline on one file. Returns 0 on success, nonzero on failure."""
    from src.application.export_manager import ExportManager
    from src.application.pipeline import OCRPipeline
    from src.application.profile_manager import ProfileManager
    from src.core.image_preprocessor import ImagePreprocessor
    from src.core.models import OCRJobConfig
    from src.core.text_postprocessor import TextPostprocessor
    from src.infrastructure.config_storage import ProfileStorage
    from src.infrastructure.tesseract_wrapper import TesseractWrapper
    from src.shared.types import ExportFormat, JobStatus

    storage = ProfileStorage()
    manager = ProfileManager(storage)
    manager.initialize_builtins()

    try:
        profile = manager.load(profile_name)
    except FileNotFoundError:
        logger.error("Профиль '%s' не найден. Доступные: %s",
                     profile_name,
                     ", ".join(p.name for p in manager.list_profiles()))
        return 2

    tesseract = TesseractWrapper()
    try:
        tesseract.configure_pytesseract()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tesseract не сконфигурирован корректно: %s", exc)

    job = OCRJobConfig(
        input_path=str(input_path),
        output_path=str(output_path),
        profile=profile,
    )

    def _progress(current: int, total: int, stage: str) -> None:
        if total > 0:
            pct = int(100 * current / total)
            logger.info("  [%3d%%] %s %d/%d", pct, stage, current, total)

    autosave_interval = 0
    try:
        from src.infrastructure.config_storage import SettingsStorage

        autosave_interval = int(SettingsStorage().load().autosave_interval_pages)
    except Exception:  # noqa: BLE001
        autosave_interval = 0

    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        tesseract=tesseract,
        progress_callback=_progress,
        autosave_interval_pages=autosave_interval,
    )

    t0 = time.time()
    logger.info(
        "Обработка: %s → %s\n"
        "  Профиль: %s | Движок: %s | DPI: %s | Языки: %s\n"
        "  Бинаризация: %s | Deskew: %s | CLAHE: %s | Timeout: %s с",
        input_path, output_path,
        profile.name,
        profile.ocr.engine.value,
        profile.ocr.dpi,
        profile.ocr.tesseract_language_string,
        profile.preprocess.binarization.method.value,
        "вкл" if profile.preprocess.deskew.enabled else "выкл",
        "вкл" if profile.preprocess.contrast.clahe_enabled else "выкл",
        profile.ocr.tesseract_timeout,
    )
    result = pipeline.run(job)
    elapsed = time.time() - t0

    if result.status is not JobStatus.COMPLETED:
        logger.error("❌ Ошибка: %s", result.error or "неизвестная")
        return 1

    logger.info(
        "✅ Готово за %.1f с (страниц: %d, средний confidence: %.1f%%)",
        elapsed, result.page_count, result.average_confidence,
    )

    # Surface the "completed but empty" advisory so the CLI user
    # sees the same hint the GUI user would see in the status bar.
    if result.error:
        logger.warning("⚠ %s", result.error)

    exporter = ExportManager()
    try:
        if want_txt:
            txt_path = output_path.with_suffix(".txt")
            exporter.export(result, txt_path, ExportFormat.TXT)
            logger.info("   Сохранён TXT: %s", txt_path)
        if want_docx:
            docx_path = output_path.with_suffix(".docx")
            exporter.export(result, docx_path, ExportFormat.DOCX)
            logger.info("   Сохранён DOCX: %s", docx_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка экспорта: %s", exc)
        return 3
    return 0


def process_batch(
    inputs: list[Path],
    profile_name: str,
    want_txt: bool,
    want_docx: bool,
    workers: int,
) -> int:
    """Process a batch. Returns 0 if every file succeeded."""
    from src.infrastructure.file_utils import safe_unique_path, suggest_output_path

    failures = 0
    if workers <= 1:
        for input_path in inputs:
            out = safe_unique_path(suggest_output_path(input_path))
            rc = process_single(input_path, out, profile_name, want_txt, want_docx)
            if rc != 0:
                failures += 1
    else:
        from src.application.parallel_processor import ParallelProcessor
        from src.application.profile_manager import ProfileManager
        from src.core.models import OCRJobConfig
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage()
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        try:
            profile = manager.load(profile_name)
        except FileNotFoundError:
            logger.error("Профиль '%s' не найден", profile_name)
            return 2

        jobs: list[OCRJobConfig] = []
        out_map: dict[str, Path] = {}
        for input_path in inputs:
            out = safe_unique_path(suggest_output_path(input_path))
            job = OCRJobConfig(
                input_path=str(input_path),
                output_path=str(out),
                profile=profile,
            )
            jobs.append(job)
            out_map[str(input_path)] = out

        pp = ParallelProcessor(max_workers=min(max(1, workers), 4))
        try:
            futures = pp.submit_all(jobs)
            for fut, job in zip(futures, jobs, strict=True):
                try:
                    _ = fut.result()
                    logger.info("✓ %s", job.input_path)
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    logger.error("✗ %s — %s", job.input_path, exc)
        finally:
            pp.shutdown(wait=True)

        # Secondary export pass is skipped in parallel mode since it would
        # require loading every JobResult; callers who need TXT/DOCX can
        # re-run single-file CLI on the output, or use workers=1.
        if want_txt or want_docx:
            logger.warning(
                "Флаги --txt/--docx игнорируются при --workers>1. "
                "Запустите однопоточно для экспорта вспомогательных форматов."
            )

    return 0 if failures == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def list_profiles() -> int:
    """Print available profiles and return 0."""
    from src.application.profile_manager import ProfileManager
    from src.infrastructure.config_storage import ProfileStorage

    storage = ProfileStorage()
    manager = ProfileManager(storage)
    manager.initialize_builtins()
    profiles = manager.list_profiles()
    print(f"Найдено профилей: {len(profiles)}")
    for p in profiles:
        marker = "⭐" if p.builtin else " "
        desc = f" — {p.description}" if p.description else ""
        print(f"  {marker} {p.name}{desc}")
    return 0


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8 (Windows cp1252 console fix).

    The CLI prints Russian strings (profile listings, progress, error
    messages). On Windows the default console encoding is cp1252 /
    cp866 / whatever the system codepage is, and those can't encode
    Cyrillic — the process crashes with ``UnicodeEncodeError``.
    ``TextIOWrapper.reconfigure`` (Python 3.7+) switches the streams
    to UTF-8 without needing an env var. Safe no-op on *nix where
    the console already speaks UTF-8.

    Best-effort: if stdout has been replaced by something exotic
    (e.g. a StringIO in tests, or a non-TextIOWrapper pipe) we leave
    it alone — the caller controls what happens to the output.
    """
    import contextlib

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            # StringIO / pipes / closed streams surface as AttributeError,
            # OSError, or ValueError depending on what they swallowed.
            with contextlib.suppress(AttributeError, OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    # Must happen BEFORE any subprocess-spawning code runs (Tesseract,
    # Ghostscript, OCRmyPDF all fork children). Without this, CLI users
    # on Windows see transient console windows flash every time a page
    # is OCR'd — same problem ``src/main.py`` guards the GUI path with.
    from src.infrastructure.subprocess_hygiene import (
        install_windows_console_hide,
    )

    install_windows_console_hide()

    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.list_profiles:
        return list_profiles()

    if not args.inputs:
        parser.error("укажите хотя бы один PDF-файл или директорию")
        return 2

    inputs = discover_inputs(args.inputs)
    if not inputs:
        logger.error("Не найдено ни одного PDF")
        return 2

    logger.info("Файлов к обработке: %d", len(inputs))

    if args.output is not None:
        if len(inputs) > 1:
            parser.error("--output можно использовать только с одним входным файлом")
            return 2
        return process_single(
            inputs[0], args.output, args.profile, args.txt, args.docx
        )

    return process_batch(inputs, args.profile, args.txt, args.docx, args.workers)


if __name__ == "__main__":
    sys.exit(main())
