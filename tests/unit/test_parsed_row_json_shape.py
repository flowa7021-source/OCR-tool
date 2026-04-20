"""Regression tests for :class:`src.tn_parser.models.ParsedRow` JSON shape.

Two bugs were discovered during post-integration review:

1. **``from_json_dict`` mutated its input.** It called ``d.pop(
   "confidence")`` directly on the caller's dict. This bit users who
   held onto the dict for UI rendering AFTER calling the method
   (e.g. :meth:`ExportManager.export_excel` rehydrates rows from
   ``job_result.parsed.rows`` and leaves those dicts stripped of
   their confidence data, making subsequent panel redraws show 0 %
   everywhere).

2. **``to_json_dict`` didn't expose a top-level ``overall_confidence``
   scalar.** The UI panel and the snapshot sidecar both want a cheap
   0–1 float without recomputing the per-field mean. Adding it
   makes each row dict self-describing — useful both for the panel's
   fast path and for external consumers reading the
   ``.xlsx.snapshot.json`` sidecar.

These tests lock the contract so a future refactor can't silently
regress either one.
"""

from __future__ import annotations

from src.tn_parser.models import FieldConfidence, ParsedRow


def _sample_row() -> ParsedRow:
    row = ParsedRow(
        waybill="Транспортная накладная № 001",
        date="01.01.2024",
        number="001",
        shipper="ООО Ромашка",
        confidence=FieldConfidence(
            date=1.0, number=0.9, shipper=0.7,
        ),
    )
    return row


# ---------------------------------------------------------------------------
# to_json_dict: overall_confidence scalar is present and correct
# ---------------------------------------------------------------------------


def test_to_json_dict_includes_overall_confidence_scalar() -> None:
    """Consumers use the scalar without recomputing the mean."""
    row = _sample_row()
    d = row.to_json_dict()
    assert "overall_confidence" in d
    # (1.0 + 0.9 + 0.7 + 0 + 0 + 0 + 0 + 0 + 0) / 9 ≈ 0.29 → rounded to 2 dp
    expected = round((1.0 + 0.9 + 0.7) / 9, 2)
    assert d["overall_confidence"] == expected


def test_to_json_dict_preserves_nested_confidence_dict() -> None:
    """The per-field ``confidence`` sub-dict must still be there."""
    row = _sample_row()
    d = row.to_json_dict()
    assert isinstance(d["confidence"], dict)
    assert d["confidence"]["date"] == 1.0
    assert d["confidence"]["number"] == 0.9


def test_to_json_dict_returns_json_serialisable_dict() -> None:
    """Whole structure survives a round-trip through ``json`` stdlib."""
    import json

    d = _sample_row().to_json_dict()
    round_tripped = json.loads(json.dumps(d, ensure_ascii=False))
    assert round_tripped["number"] == "001"
    assert round_tripped["overall_confidence"] == d["overall_confidence"]


# ---------------------------------------------------------------------------
# from_json_dict: does NOT mutate input
# ---------------------------------------------------------------------------


def test_from_json_dict_does_not_mutate_source() -> None:
    """Rebuilding a ParsedRow leaves the source dict untouched.

    This is the regression guard for the
    ``ExportManager.export_excel → ParsedRow.from_json_dict → UI panel``
    sequence: if from_json_dict pops keys from the dict the panel is
    still holding, every subsequent render collapses to 0 % colouring.
    """
    original = _sample_row().to_json_dict()
    snapshot = dict(original)  # shallow copy for later comparison
    snapshot["confidence"] = dict(original["confidence"])  # confidence is nested

    ParsedRow.from_json_dict(original)

    assert original == snapshot, (
        "from_json_dict mutated its input; "
        f"before={snapshot!r}, after={original!r}"
    )
    # Specifically: the ``confidence`` sub-dict must still be on
    # ``original`` — this is the key a future UI re-render reads.
    assert "confidence" in original
    assert original["confidence"]["date"] == 1.0


def test_from_json_dict_accepts_overall_confidence_key() -> None:
    """New top-level scalar is ignored (dataclass has no field for it)."""
    d = _sample_row().to_json_dict()
    # Before the fix, passing ``overall_confidence`` through **kwargs
    # would raise ``TypeError: unexpected keyword argument``.
    rebuilt = ParsedRow.from_json_dict(d)
    assert rebuilt.number == "001"
    assert rebuilt.confidence.date == 1.0


# ---------------------------------------------------------------------------
# End-to-end: orchestrator → panel → export_excel → panel again
# ---------------------------------------------------------------------------


def test_double_rehydration_preserves_confidence() -> None:
    """Simulate the real ExportManager sequence that tripped the bug.

    1. Orchestrator serialises rows into ``ParsedDocument.rows``.
    2. ExportManager.export_excel rehydrates via ``from_json_dict``.
    3. UI panel re-reads the same ``ParsedDocument.rows`` after export.

    The panel's confidence band must still render correctly on step 3.
    """
    orch_output = [_sample_row().to_json_dict()]

    # Step 2: Excel exporter rehydrates.
    rebuilt = [ParsedRow.from_json_dict(d) for d in orch_output]
    assert rebuilt[0].confidence.date == 1.0

    # Step 3: UI re-reads the SAME list of dicts. Confidence data
    # must still be reachable via either the scalar fast-path or
    # the nested-dict fallback path.
    row_dict = orch_output[0]
    assert row_dict.get("overall_confidence") is not None, (
        "fast path broken — scalar missing after rehydration"
    )
    assert isinstance(row_dict.get("confidence"), dict), (
        "fallback path broken — nested dict wiped by from_json_dict"
    )
    assert row_dict["confidence"]["date"] == 1.0
