# -*- coding: utf-8 -*-
"""Генератор/обновлятор golden-снэпшотов для парсера ТН.

Использование:
    # Создать/обновить ожидание для одного кейса из его .txt:
    python tools/update_golden.py tests/fixtures/golden/<case>.txt

    # То же самое, но принять существующие контракты (contains/not_contains)
    # без перезаписи — только заполнить отсутствующие поля:
    python tools/update_golden.py tests/fixtures/golden/<case>.txt --merge

Сценарий:
    1. Пользователь приносит сырой OCR-текст накладной, на которой парсер
       работал плохо.
    2. Мы вносим точечный фикс в парсер, ПОКА разработка не устаканилась —
       ведём `.expected.json` вручную.
    3. Когда результат идеален — запускаем этот скрипт, чтобы зафиксировать
       текущий вывод парсера как контрактный.
    4. Снэпшот попадает в репозиторий; CI ловит любую регрессию мгновенно.

Формат записи соответствует описанию в `tests/test_golden.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Делаем пакет tn_parser доступным, если запускают из корня репо.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tn_parser import parse_text  # noqa: E402
from tn_parser.models import ParsedRow  # noqa: E402
from tn_parser.normalize import normalize_for_sections  # noqa: E402


_FIELDS = (
    "number", "date", "shipper", "consignee",
    "cargo", "driver", "vehicle", "reception",
)


def _row_to_expected(row: ParsedRow) -> Dict[str, Any]:
    """Снэпшот ParsedRow в формат expected.json (строгое equals по всем полям)."""
    out: Dict[str, Any] = {}
    for f in _FIELDS:
        out[f] = {"equals": getattr(row, f)}
    out["conf_min"] = round(row.confidence.overall() - 0.05, 2)  # небольшой запас
    if row.note:
        # Если в note есть ERROR — это не нормально для golden, оставляем заметку.
        if "ERROR" in row.note:
            out["note_not_contains"] = []  # пользователь должен осознанно утвердить
        else:
            out["note_not_contains"] = ["ERROR"]
    else:
        out["note_not_contains"] = ["ERROR"]
    return out


def _merge_expected(prev: Dict[str, Any], fresh: Dict[str, Any]) -> Dict[str, Any]:
    """Слить свежий срез с существующим: сохраняем contains/not_contains,
    добавляем отсутствующие поля.
    """
    out = dict(prev)
    for k, v in fresh.items():
        if k not in out:
            out[k] = v
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Обновить expected.json golden-кейса из текущего вывода парсера.",
    )
    p.add_argument("txt_path", help="Путь к <case>.txt в tests/fixtures/golden/")
    p.add_argument(
        "--merge",
        action="store_true",
        help="Слить свежий срез с существующим (не перезаписывать contains-правила).",
    )
    args = p.parse_args(argv)

    txt_path = Path(args.txt_path).resolve()
    if not txt_path.exists():
        print(f"Нет файла: {txt_path}", file=sys.stderr)
        return 2
    if not txt_path.suffix == ".txt":
        print(f"Ожидался .txt, получено: {txt_path.name}", file=sys.stderr)
        return 2

    expected_path = txt_path.with_suffix("").with_suffix(".expected.json")
    case_name = txt_path.stem

    raw = txt_path.read_text(encoding="utf-8")
    rows = parse_text(normalize_for_sections(raw), f"{case_name}.pdf")

    fresh_rows = [_row_to_expected(r) for r in rows]
    fresh_doc: Dict[str, Any] = {"rows": fresh_rows}

    if args.merge and expected_path.exists():
        prev = json.loads(expected_path.read_text(encoding="utf-8"))
        prev_rows = prev.get("rows") or []
        merged_rows: List[Dict[str, Any]] = []
        for i, fr in enumerate(fresh_rows):
            if i < len(prev_rows):
                merged_rows.append(_merge_expected(prev_rows[i], fr))
            else:
                merged_rows.append(fr)
        fresh_doc = {"rows": merged_rows}

    expected_path.write_text(
        json.dumps(fresh_doc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Обновлён: {expected_path}")
    for i, r in enumerate(rows):
        print(f"  row {i}: number={r.number!r}, conf={r.confidence.overall()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
