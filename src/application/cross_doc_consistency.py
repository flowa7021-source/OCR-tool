"""Cross-document consistency checks for TN/UPD bundles.

When processing a delivery packet (Транспортная накладная +
Универсальный передаточный документ + доверенность), the parties,
dates and amounts must agree across documents: same shipper, same
consignee, same goods total. Mismatches signal an OCR error in at
least one document — the consistency check flags it so a human can
review.

Exposes a single entry point ``check_bundle(parsed_docs)`` that takes
a list of already-parsed documents (from :mod:`src.tn_parser`) and
returns a list of :class:`Inconsistency` objects. No network,
no database — pure logic.

This module complements, not replaces, ``src.tn_parser.cross_requisites``
which covers within-document cross-field checks (e.g., КПП starts with
same region as ИНН). Cross-DOCUMENT is additional — here we compare
between two separate documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Inconsistency:
    """A discrepancy between two documents in the bundle."""
    severity: str        # "error" / "warning" / "info"
    doc_a: str           # identifier for first document
    doc_b: str           # identifier for second document
    field: str           # "shipper_inn" / "date" / "total_amount" / …
    value_a: Any
    value_b: Any
    message: str         # human-readable description

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "doc_a": self.doc_a,
            "doc_b": self.doc_b,
            "field": self.field,
            "value_a": str(self.value_a) if self.value_a is not None else None,
            "value_b": str(self.value_b) if self.value_b is not None else None,
            "message": self.message,
        }


@dataclass
class ParsedDoc:
    """Minimal parsed-document shape this module depends on.

    Replace with the real ProjectData / ParsedDocument from the
    tn_parser module when wiring into the pipeline. Kept as a
    separate dataclass here to avoid a circular import and make the
    module unit-testable without the full parser chain.
    """
    id: str                          # file name or similar label
    doc_type: str                    # "TN" / "UPD" / …
    shipper_inn: str | None = None
    consignee_inn: str | None = None
    number: str | None = None
    date: str | None = None          # ISO YYYY-MM-DD preferred
    total_amount: float | None = None
    items: list[dict] = field(default_factory=list)


def _normalise_inn(s: str | None) -> str | None:
    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    return digits or None


def _inn_match(a: str | None, b: str | None) -> bool:
    na, nb = _normalise_inn(a), _normalise_inn(b)
    if na is None or nb is None:
        return True  # missing data — cannot check
    return na == nb


def _date_diff_days(a: str | None, b: str | None) -> int | None:
    if not a or not b:
        return None
    import datetime
    parsed: list[datetime.date] = []
    for d in (a, b):
        for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
            try:
                parsed.append(datetime.datetime.strptime(d, fmt).date())
                break
            except ValueError:
                continue
        else:
            return None
    return abs((parsed[0] - parsed[1]).days)


def _amount_close(a: float | None, b: float | None,
                  relative_tol: float = 0.01) -> bool:
    """Amounts within ±1% of each other (default)."""
    if a is None or b is None:
        return True
    if a == 0 and b == 0:
        return True
    mean = (abs(a) + abs(b)) / 2
    return abs(a - b) / mean <= relative_tol if mean else a == b


def check_bundle(
    docs: list[ParsedDoc],
    max_date_gap_days: int = 30,
) -> list[Inconsistency]:
    """Run cross-document consistency checks on a bundle.

    For every pair of documents that could plausibly relate to the same
    transaction (typically a TN + its UPD), check:
      - shipper_inn match
      - consignee_inn match
      - date within ``max_date_gap_days`` of each other
      - total_amount match (within 1% tolerance)

    Mismatches are returned as a list of :class:`Inconsistency` objects.
    Empty list means the bundle is internally consistent (or there's
    not enough data to say).
    """
    inconsistencies: list[Inconsistency] = []
    for i, a in enumerate(docs):
        for b in docs[i + 1:]:
            # shipper_inn
            if not _inn_match(a.shipper_inn, b.shipper_inn):
                inconsistencies.append(Inconsistency(
                    severity="error",
                    doc_a=a.id, doc_b=b.id,
                    field="shipper_inn",
                    value_a=a.shipper_inn, value_b=b.shipper_inn,
                    message="Грузоотправитель ИНН отличается между документами — "
                            "как минимум один распознан с ошибкой.",
                ))
            # consignee_inn
            if not _inn_match(a.consignee_inn, b.consignee_inn):
                inconsistencies.append(Inconsistency(
                    severity="error",
                    doc_a=a.id, doc_b=b.id,
                    field="consignee_inn",
                    value_a=a.consignee_inn, value_b=b.consignee_inn,
                    message="Грузополучатель ИНН отличается между документами.",
                ))
            # date gap
            gap = _date_diff_days(a.date, b.date)
            if gap is not None and gap > max_date_gap_days:
                inconsistencies.append(Inconsistency(
                    severity="warning",
                    doc_a=a.id, doc_b=b.id,
                    field="date",
                    value_a=a.date, value_b=b.date,
                    message=(
                        f"Даты расходятся на {gap} дней — больше порога "
                        f"({max_date_gap_days}). Возможно, документы "
                        "относятся к разным поставкам."
                    ),
                ))
            # total amount
            if not _amount_close(a.total_amount, b.total_amount):
                inconsistencies.append(Inconsistency(
                    severity="warning",
                    doc_a=a.id, doc_b=b.id,
                    field="total_amount",
                    value_a=a.total_amount, value_b=b.total_amount,
                    message="Итоговые суммы не совпадают (вне допуска ±1%).",
                ))
    return inconsistencies


__all__ = ["ParsedDoc", "Inconsistency", "check_bundle"]
