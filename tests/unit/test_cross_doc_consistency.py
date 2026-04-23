"""Tests for cross-document consistency checks."""

from __future__ import annotations

from src.application.cross_doc_consistency import (
    Inconsistency,
    ParsedDoc,
    check_bundle,
)


def test_empty_bundle_no_issues():
    assert check_bundle([]) == []


def test_single_doc_no_checks():
    doc = ParsedDoc(id="a.pdf", doc_type="TN", shipper_inn="7813266190")
    assert check_bundle([doc]) == []


def test_matching_shippers_no_issue():
    a = ParsedDoc(id="tn.pdf", doc_type="TN", shipper_inn="7813266190")
    b = ParsedDoc(id="upd.pdf", doc_type="UPD", shipper_inn="7813266190")
    assert check_bundle([a, b]) == []


def test_mismatched_shipper_inn_is_error():
    a = ParsedDoc(id="tn.pdf", doc_type="TN", shipper_inn="7813266190")
    b = ParsedDoc(id="upd.pdf", doc_type="UPD", shipper_inn="7707820890")
    out = check_bundle([a, b])
    assert len(out) == 1
    assert out[0].severity == "error"
    assert out[0].field == "shipper_inn"


def test_date_gap_warning():
    a = ParsedDoc(id="tn.pdf", doc_type="TN", date="2022-01-01")
    b = ParsedDoc(id="upd.pdf", doc_type="UPD", date="2022-03-15")
    out = check_bundle([a, b], max_date_gap_days=30)
    assert len(out) == 1
    assert out[0].severity == "warning"
    assert out[0].field == "date"


def test_close_dates_ok():
    a = ParsedDoc(id="tn.pdf", doc_type="TN", date="2022-09-02")
    b = ParsedDoc(id="upd.pdf", doc_type="UPD", date="2022-09-05")
    assert check_bundle([a, b]) == []


def test_amount_mismatch_warning():
    a = ParsedDoc(id="tn.pdf", doc_type="TN", total_amount=1000.00)
    b = ParsedDoc(id="upd.pdf", doc_type="UPD", total_amount=1100.00)
    out = check_bundle([a, b])
    assert any(i.field == "total_amount" for i in out)


def test_small_amount_diff_tolerated():
    a = ParsedDoc(id="tn.pdf", doc_type="TN", total_amount=1000.00)
    b = ParsedDoc(id="upd.pdf", doc_type="UPD", total_amount=1005.00)
    # Within 1% tolerance
    out = check_bundle([a, b])
    assert not any(i.field == "total_amount" for i in out)


def test_inconsistency_to_dict():
    inc = Inconsistency(
        severity="error", doc_a="a.pdf", doc_b="b.pdf",
        field="shipper_inn", value_a="123", value_b="456",
        message="mismatch",
    )
    d = inc.to_dict()
    assert d["severity"] == "error"
    assert d["value_a"] == "123"
