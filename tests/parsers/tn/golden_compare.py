"""Helpers для golden-сравнения parser-output vs expected.

Используется в :mod:`tests.parsers.tn.test_parser_golden_real_pdf`
для per-field сравнения с поле-специфичными tolerance'ами. Золотые
фикстуры лежат в :file:`golden_fixtures/*.json` и описывают, какими
ДОЛЖНЫ быть ``ParsedRow``-поля для конкретной реальной ТН.

Поддерживаемые tolerance-режимы:

* ``"exact"`` — значение совпадает посимвольно после strip.
  Применяется к идентификаторам (ИНН/КПП/ОГРН, ГРЗ, номера и даты).

* ``"contains"`` — expected-подстрока должна встречаться в actual.
  Минимально: hit anywhere (case-insensitive). Применяется к полям
  где OCR привносит шум, но ключевое слово должно выжить (``cargo``
  с «TENSAR», ``shipper`` с «ГЕКСАФОРМ», ``driver`` с фамилией).

* ``"fuzzy:N"`` — rapidfuzz ``token_sort_ratio`` ≥ N% (0-100).
  Применяется к полям с OCR-typo потенциалом (``shipper`` полностью,
  ``consignee``, ``address``).

* ``"optional"`` — поле в expected не указано / equal "?" → любое
  actual-значение проходит (в т.ч. пустое). Для полей которые
  формально могут отсутствовать в документе.

Результат сравнения — dict ``{field: {"pass": bool, "score": float,
"actual": str, "expected": str, "note": str}}`` плюс overall-score
(взвешенное среднее). Тест решает fail/pass по overall-threshold'у
чтобы одна OCR-ошибка не роняла весь golden: главное что 70%+ полей
извлекаются корректно.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldCheck:
    """Результат сравнения одного поля."""

    name: str
    passed: bool
    score: float  # 0.0..1.0
    actual: str
    expected: str
    tolerance: str
    note: str = ""


def _normalise(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    # MISSING/GARBAGE-литералы трактуем как «пусто» — соответствует
    # тому, что Excel-exporter делает в ``_normalise_for_excel``.
    if s in ("отсутствует", "неразборчиво"):
        return ""
    return s


def _compare_exact(actual: str, expected: str) -> tuple[bool, float]:
    return (actual.strip() == expected.strip(), 1.0 if actual.strip() == expected.strip() else 0.0)


def _compare_contains(actual: str, expected: str) -> tuple[bool, float]:
    a = actual.lower().strip()
    e = expected.lower().strip()
    if not e:
        return True, 1.0
    hit = e in a
    return hit, 1.0 if hit else 0.0


def _compare_fuzzy(actual: str, expected: str, min_ratio: int) -> tuple[bool, float]:
    """Fuzzy-ratio через rapidfuzz. Возвращает (passed, normalised_score)."""
    if not actual or not expected:
        # Один из них пуст — fallback на «равны ли оба пусты».
        return actual == expected, 1.0 if actual == expected else 0.0
    try:
        from rapidfuzz.fuzz import token_sort_ratio
    except ImportError:
        # Фолбэк: очень грубо по общим токенам.
        actual_tokens = set(actual.lower().split())
        expected_tokens = set(expected.lower().split())
        if not expected_tokens:
            return True, 1.0
        overlap = len(actual_tokens & expected_tokens) / len(expected_tokens)
        return overlap * 100 >= min_ratio, overlap
    ratio = token_sort_ratio(actual, expected)
    return ratio >= min_ratio, ratio / 100.0


def compare_field(
    name: str,
    actual: Any,
    expected: Any,
    tolerance: str,
) -> FieldCheck:
    """Сравнить одно поле по заданному tolerance-режиму.

    Args:
        name: Имя поля (для отчёта).
        actual: Значение из ``ParsedRow``. ``None`` / ``""`` / MISSING
            нормализуются к пустой строке.
        expected: Значение из golden-фикстуры. ``"?"`` или ``None``
            трактуется как optional-поле (любое actual проходит).
        tolerance: Один из ``"exact"``, ``"contains"``, ``"fuzzy:N"``,
            ``"optional"``.

    Returns:
        :class:`FieldCheck` с результатом.
    """
    a = _normalise(actual)
    e = _normalise(expected)

    if tolerance == "optional" or e in ("", "?"):
        return FieldCheck(
            name=name, passed=True, score=1.0, actual=a, expected=e,
            tolerance=tolerance, note="optional — skipped",
        )

    if tolerance == "exact":
        ok, score = _compare_exact(a, e)
    elif tolerance == "contains":
        ok, score = _compare_contains(a, e)
    elif tolerance.startswith("fuzzy:"):
        try:
            min_ratio = int(tolerance.split(":", 1)[1])
        except (ValueError, IndexError):
            min_ratio = 75
        ok, score = _compare_fuzzy(a, e, min_ratio)
    else:
        raise ValueError(f"Unknown tolerance mode: {tolerance!r}")

    return FieldCheck(
        name=name, passed=ok, score=score, actual=a, expected=e,
        tolerance=tolerance,
    )


#: Веса полей для overall-score. Идентификаторы и основные поля —
#: высокий вес, auxiliary fields — низкий. Поля отсутствующие в
#: expected автоматически не учитываются (optional).
DEFAULT_WEIGHTS: dict[str, float] = {
    "date": 3.0,
    "number": 2.0,
    "shipper": 2.0,
    "shipper_inn": 3.0,
    "shipper_kpp": 2.0,
    "shipper_ogrn": 2.0,
    "consignee": 2.0,
    "consignee_inn": 3.0,
    "consignee_kpp": 2.0,
    "consignee_ogrn": 2.0,
    "cargo": 1.5,
    "volume": 1.0,
    "driver": 1.5,
    "vehicle": 1.5,
    "reception": 1.0,
}


def compare_row(
    actual_row: dict[str, Any],
    expected_spec: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> tuple[list[FieldCheck], float]:
    """Сравнить ParsedRow-dict с golden-spec.

    Expected-spec — dict вида::

        {
            "date":        {"value": "29.08.2022",  "tolerance": "exact"},
            "shipper_inn": {"value": "7813266190",  "tolerance": "exact"},
            "cargo":       {"value": "Георешетка",  "tolerance": "contains"},
            "shipper":     {"value": "ГЕКСАФОРМ",   "tolerance": "fuzzy:70"},
            "driver":      {"value": "?",            "tolerance": "optional"},
            ...
        }

    Returns:
        ``(field_checks, overall_score)``. ``overall_score`` — взвешенное
        среднее score'ов по non-optional полям, 0.0..1.0.
    """
    weights = weights or DEFAULT_WEIGHTS
    checks: list[FieldCheck] = []
    weighted_sum = 0.0
    total_weight = 0.0

    for field_name, spec in expected_spec.items():
        if not isinstance(spec, dict):
            # Сокращённая форма: field_name: "value" при tolerance=exact
            spec = {"value": spec, "tolerance": "exact"}
        check = compare_field(
            name=field_name,
            actual=actual_row.get(field_name),
            expected=spec.get("value", ""),
            tolerance=spec.get("tolerance", "exact"),
        )
        checks.append(check)
        if check.tolerance == "optional" or "optional" in check.note:
            continue
        w = weights.get(field_name, 1.0)
        weighted_sum += check.score * w
        total_weight += w

    overall = weighted_sum / total_weight if total_weight > 0 else 1.0
    return checks, overall


def format_report(checks: list[FieldCheck], overall: float) -> str:
    """Человекочитаемый отчёт для assertion-message."""
    lines = [f"Overall golden score: {overall*100:.1f}%"]
    for c in sorted(checks, key=lambda x: x.passed):
        sym = "✓" if c.passed else "✗"
        a = (c.actual[:60] + "…") if len(c.actual) > 60 else c.actual
        e = (c.expected[:60] + "…") if len(c.expected) > 60 else c.expected
        lines.append(
            f"  {sym} {c.name:18} [{c.tolerance:10}] "
            f"score={c.score:.2f}  actual={a!r}  expected={e!r}"
        )
    return "\n".join(lines)
