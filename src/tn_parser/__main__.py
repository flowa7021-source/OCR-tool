"""CLI-режим парсера.

Запуск:
    python -m src.tn_parser <папка-или-файл> [--out FILE.xlsx] [--log FILE.log]
                                             [--no-cache] [-v]

По умолчанию:
    --out — <input>/extraction.xlsx (для папки) или ./extraction.xlsx (для файла)
    --log — тот же путь, что --out, но с расширением .log
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import (
    build_log_lines,
    iter_pdfs,
    process_batch,
    write_excel_safe,
    write_log_safe,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.tn_parser",
        description="Парсер транспортных накладных из PDF в Excel (CLI).",
    )
    p.add_argument(
        "input",
        help="Папка с PDF-файлами или путь к одному PDF.",
    )
    p.add_argument(
        "-o", "--out",
        dest="output",
        default=None,
        help="Путь к Excel-файлу (по умолчанию — extraction.xlsx рядом со входом).",
    )
    p.add_argument(
        "-l", "--log",
        dest="log",
        default=None,
        help="Путь к файлу лога (по умолчанию — extraction.log рядом с Excel).",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Не использовать кэш (перепарсить все файлы заново).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Печатать прогресс по каждому файлу.",
    )
    return p


def _default_output(input_path: str) -> str:
    if os.path.isdir(input_path):
        return os.path.join(input_path, "extraction.xlsx")
    return os.path.join(os.getcwd(), "extraction.xlsx")


def _default_log(output_xlsx: str) -> str:
    base, _ = os.path.splitext(output_xlsx)
    return base + ".log"


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Путь не найден: {input_path}", file=sys.stderr)
        return 2

    pdfs = iter_pdfs(input_path)
    if not pdfs:
        print(f"PDF-файлы не найдены в {input_path}", file=sys.stderr)
        return 1

    output = os.path.abspath(args.output) if args.output else _default_output(input_path)
    log_path = os.path.abspath(args.log) if args.log else _default_log(output)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    def _on_progress(done: int, total: int, pdf_path: str, rows) -> None:
        if not args.verbose:
            return
        fname = os.path.basename(pdf_path)
        if rows and rows[0].note.startswith("ERROR:"):
            mark = "ERR "
        elif any(r.confidence.overall() < 0.5 for r in rows):
            mark = "WARN"
        else:
            mark = "OK  "
        avg = round(sum(r.confidence.overall() for r in rows) / max(len(rows), 1) * 100)
        print(f"  [{mark}] {done}/{total} ({avg:>3}%) {fname}")

    t0 = time.time()
    print(f"Обработка {len(pdfs)} файлов…")
    results = process_batch(
        pdfs,
        use_cache=not args.no_cache,
        progress=_on_progress,
    )
    elapsed = time.time() - t0

    # Сохраняем в исходном порядке файлов.
    ordered_rows = []
    for p in pdfs:
        ordered_rows.extend(results.get(p, []))

    try:
        actual_xlsx = write_excel_safe(ordered_rows, output)
    except PermissionError as exc:
        print(f"Не удалось сохранить Excel: {exc}", file=sys.stderr)
        print("Закройте файл, если он открыт в Excel, и попробуйте снова.",
              file=sys.stderr)
        return 4

    if actual_xlsx != output:
        print(f"⚠ {output} занят — сохранили как {actual_xlsx}", file=sys.stderr)

    rows_by_fname = {os.path.basename(p): results[p] for p in pdfs if p in results}
    log_lines = build_log_lines(
        input_path=input_path,
        output_path=actual_xlsx,
        elapsed_s=elapsed,
        rows_by_file=rows_by_fname,
    )
    try:
        actual_log = write_log_safe(log_path, log_lines)
    except PermissionError:
        actual_log = None

    ok = sum(
        1 for rows in results.values()
        if not any(r.note.startswith("ERROR:") for r in rows)
    )
    err = len(results) - ok
    print(
        f"Готово: {len(pdfs)} файлов ({ok} OK, {err} ошибок), "
        f"{len(ordered_rows)} строк за {elapsed:.1f} с"
    )
    print(f"Excel: {actual_xlsx}")
    if actual_log:
        print(f"Лог:   {actual_log}")
    return 0 if err == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
