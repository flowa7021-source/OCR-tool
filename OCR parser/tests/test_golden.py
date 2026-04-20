# -*- coding: utf-8 -*-
"""Golden-тесты: каждая пара `<case>.txt` + `<case>.expected.json` в
`tests/fixtures/golden/` автоматически становится отдельным тест-кейсом.

Назначение — зафиксировать «правильный» вывод парсера для реальных OCR-
текстов, которые пользователь приносит в поддержку. После того как парсер
правильно обрабатывает файл, мы снимаем JSON-срез ожидания и коммитим его.
Любая регрессия в будущем сразу ловится CI — без необходимости пользователю
снова присылать сырой текст и просить ручной фикс.

Формат `<case>.expected.json`:

    {
        "rows": [
            {
                "number":    "7145/Б",
                "date":      "23.07.2022",
                "shipper":   {"contains": ["Бекам", "7743553262"],
                              "not_contains": ["является экспедитором"]},
                "consignee": {"contains": ["Моспроект"]},
                "cargo":     {"equals": "Блок облицовочный..."},
                # Driver («Самовывоз» — не ФИО, поле должно остаться MISSING).
                "vehicle":   {"equals": "RENAULT Р 814 НР 152"},
                "reception": {"contains": ["Подолино"]},
                "conf_min":  0.8,
                "note_not_contains": ["LOW_TEXT", "ERROR"]
            }
        ]
    }

Правила сопоставления (любой ключ опционален):
    equals         — точное совпадение значения поля
    contains       — список подстрок, которые ДОЛЖНЫ быть в значении
    not_contains   — список подстрок, которых в значении быть не должно
    conf_min       — минимальная `confidence.overall()` строки
    note_not_contains — список подстрок, которых в `note` быть не должно

Для новой накладной достаточно положить два файла рядом — тест подхватит
пару автоматически и проверит все указанные ожидания.

Чтобы пересоздать snapshot ожидаемых значений из текущего парсера, используйте
`python tools/update_golden.py <case>` — см. модуль в tools/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tn_parser import parse_text
from tn_parser.models import ParsedRow
from tn_parser.normalize import normalize_for_sections


GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


def _discover_cases() -> List[Path]:
    if not GOLDEN_DIR.exists():
        return []
    return sorted(GOLDEN_DIR.glob("*.expected.json"))


def _load_expected(expected_path: Path) -> Dict[str, Any]:
    return json.loads(expected_path.read_text(encoding="utf-8"))


def _row_as_dict(row: ParsedRow) -> Dict[str, str]:
    return {
        "number": row.number,
        "date": row.date,
        "shipper": row.shipper,
        "consignee": row.consignee,
        "cargo": row.cargo,
        "driver": row.driver,
        "vehicle": row.vehicle,
        "reception": row.reception,
        "note": row.note,
    }


def _check_field(value: str, expectation: Any, field: str, case: str) -> None:
    """Проверяет одно поле против правил из expected.json.

    Значение `expectation` может быть:
        * str                — эквивалент {"equals": <str>}
        * dict с ключами     — equals / contains / not_contains
    """
    if isinstance(expectation, str):
        assert value == expectation, (
            f"[{case}] {field}: expected {expectation!r}, got {value!r}"
        )
        return
    if not isinstance(expectation, dict):
        raise AssertionError(
            f"[{case}] {field}: expectation must be str or dict, got {type(expectation).__name__}"
        )
    if "equals" in expectation:
        assert value == expectation["equals"], (
            f"[{case}] {field}: expected equals {expectation['equals']!r}, got {value!r}"
        )
    for needle in expectation.get("contains", []) or []:
        assert needle in value, (
            f"[{case}] {field}: expected to contain {needle!r}, got {value!r}"
        )
    for needle in expectation.get("not_contains", []) or []:
        assert needle not in value, (
            f"[{case}] {field}: must NOT contain {needle!r}, got {value!r}"
        )


_CASES = _discover_cases()


@pytest.mark.parametrize(
    "expected_path",
    _CASES,
    ids=[p.stem.replace(".expected", "") for p in _CASES],
)
def test_golden_case(expected_path: Path) -> None:
    case = expected_path.stem.replace(".expected", "")
    txt_path = expected_path.with_name(f"{case}.txt")
    if not txt_path.exists():
        pytest.skip(f"raw text missing for golden case {case}: {txt_path.name}")

    raw = txt_path.read_text(encoding="utf-8")
    rows = parse_text(normalize_for_sections(raw), f"{case}.pdf")

    spec = _load_expected(expected_path)
    expected_rows = spec.get("rows") or []
    if not expected_rows:
        pytest.fail(f"[{case}] expected.json has no 'rows' array")

    assert len(rows) == len(expected_rows), (
        f"[{case}] parsed {len(rows)} rows, expected {len(expected_rows)}"
    )

    for i, (row, exp) in enumerate(zip(rows, expected_rows)):
        scope = f"{case}#{i}"
        for field in ("number", "date", "shipper", "consignee",
                      "cargo", "driver", "vehicle", "reception"):
            if field in exp:
                _check_field(getattr(row, field), exp[field], field, scope)

        if "conf_min" in exp:
            got = row.confidence.overall()
            assert got >= exp["conf_min"], (
                f"[{scope}] confidence {got} < {exp['conf_min']}"
            )
        for needle in exp.get("note_not_contains", []) or []:
            assert needle not in row.note, (
                f"[{scope}] note must NOT contain {needle!r}, got {row.note!r}"
            )


def test_golden_directory_is_not_silently_empty():
    """Страховка: если кто-то удалит все golden-кейсы, тест упадёт и мы
    увидим, что защитная сетка пропала."""
    assert GOLDEN_DIR.exists(), f"golden directory missing: {GOLDEN_DIR}"
    cases = _discover_cases()
    assert cases, (
        f"no golden cases in {GOLDEN_DIR}. "
        f"Add <case>.txt and <case>.expected.json pairs."
    )
