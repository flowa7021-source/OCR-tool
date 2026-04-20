# -*- coding: utf-8 -*-
"""Статистика по накопленному корпусу правок оператора.

Использование:

    python tools/feedback_stats.py
    python tools/feedback_stats.py --log feedback/corrections.jsonl
    python tools/feedback_stats.py --top 20

Показывает:
    - Сколько правок всего и по каким полям.
    - Средняя confidence парсера для правленых значений (чем ниже,
      тем лучше парсер «знал», что он ошибается).
    - Типичные паттерны (commonly replaced values).
    - Поля, которые оператор добавил (fill) vs исправил (fix).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tn_parser.feedback import TRACKABLE_FIELDS, load_corrections  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--log",
        default="feedback/corrections.jsonl",
        help="Путь к JSONL-логу правок",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Сколько самых частых паттернов показать",
    )
    args = p.parse_args()

    log_path = Path(args.log).resolve()
    corrections = load_corrections(log_path)
    if not corrections:
        print(f"Корпус пуст: {log_path} не существует или без правок.")
        return 0

    print(f"Всего правок: {len(corrections)}  ({log_path})")
    print()

    # Группировка по полю.
    by_field: dict = defaultdict(list)
    for c in corrections:
        by_field[c.field].append(c)

    print("По полям:")
    print(f"  {'Поле':12s}  {'Всего':>6s}  {'fix':>5s}  {'fill':>5s}  "
          f"{'clear':>6s}  {'ср.conf':>8s}")
    for fld in TRACKABLE_FIELDS:
        items = by_field.get(fld, [])
        if not items:
            continue
        fix = sum(1 for c in items if c.kind == "fix")
        fill = sum(1 for c in items if c.kind == "fill")
        clear = sum(1 for c in items if c.kind == "clear")
        avg_conf = sum(c.original_confidence for c in items) / len(items)
        print(f"  {fld:12s}  {len(items):>6d}  {fix:>5d}  {fill:>5d}  "
              f"{clear:>6d}  {avg_conf:>8.2f}")

    print()
    print(f"Топ-{args.top} правок по повторяемости (original → corrected):")
    pairs = Counter(
        (c.field, c.original[:40], c.corrected[:40]) for c in corrections
    )
    for (fld, orig, corr), n in pairs.most_common(args.top):
        print(f"  [{n:3d}] {fld}: {orig!r} → {corr!r}")

    # Самые «уверенно неправильные» — парсер был уверен, но оказался неправ.
    high_conf_fails = sorted(
        (c for c in corrections if c.original_confidence >= 0.8),
        key=lambda c: -c.original_confidence,
    )
    if high_conf_fails:
        print()
        print(f"«Уверенно-неправильных» (conf ≥ 0.8): {len(high_conf_fails)}")
        print("Это — критичные слабости парсера: он считал ответ верным.")
        for c in high_conf_fails[:args.top]:
            print(f"  [{c.original_confidence:.2f}] {c.field}: "
                  f"{c.original!r} → {c.corrected!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
