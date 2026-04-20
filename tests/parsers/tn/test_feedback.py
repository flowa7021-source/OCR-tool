"""Тесты feedback loop: снапшот → diff с правками → корпус JSONL."""


from src.tn_parser.feedback import (
    TRACKABLE_FIELDS,
    Correction,
    append_corrections,
    diff_rows,
    load_corrections,
    load_snapshot,
    save_snapshot,
    snapshot_from_rows,
)
from src.tn_parser.models import MISSING, FieldConfidence, ParsedRow


def _make_row(**kwargs) -> ParsedRow:
    conf = FieldConfidence(**kwargs.pop("_confidence", {}))
    row = ParsedRow(**kwargs)
    row.confidence = conf
    return row


class TestSnapshot:
    def test_snapshot_preserves_all_fields(self):
        rows = [
            _make_row(
                waybill="Транспортная накладная № 001",
                date="01.01.2024", number="001",
                shipper="ООО Ромашка", consignee="АО Василёк",
                cargo="Мука", volume="500 шт", driver="Иванов И.И.",
                vehicle="А111ВВ77", reception="г. Москва",
                source="a.pdf", note="",
                _confidence={
                    "date": 1.0, "number": 0.9, "shipper": 0.8,
                    "consignee": 0.7, "cargo": 0.9, "volume": 0.9,
                    "driver": 1.0, "vehicle": 0.8, "reception": 0.7,
                },
            ),
        ]
        snap = snapshot_from_rows(rows)
        assert len(snap) == 1
        s = snap[0]
        assert s["source"] == "a.pdf"
        for fld in TRACKABLE_FIELDS:
            assert fld in s
            assert f"{fld}_conf" in s
        assert s["number"] == "001"
        assert s["number_conf"] == 0.9

    def test_save_load_snapshot(self, tmp_path):
        rows = [_make_row(source="x.pdf", number="123")]
        p = tmp_path / "out.xlsx.snapshot.json"
        save_snapshot(rows, p)
        loaded = load_snapshot(p)
        assert loaded is not None
        assert loaded["rows"][0]["source"] == "x.pdf"
        assert loaded["rows"][0]["number"] == "123"


class TestDiff:
    def test_detects_fix(self):
        snap = [{"source": "a.pdf", "number": "1", "number_conf": 0.5}]
        corr = [{"source": "a.pdf", "number": "7145/Б"}]
        result = diff_rows(snap, corr)
        assert len(result) == 1
        c = result[0]
        assert c.field == "number"
        assert c.original == "1"
        assert c.corrected == "7145/Б"
        assert c.kind == "fix"
        assert c.original_confidence == 0.5

    def test_detects_fill(self):
        snap = [{"source": "a.pdf", "shipper": MISSING, "shipper_conf": 0.0}]
        corr = [{"source": "a.pdf", "shipper": "ООО Новая"}]
        result = diff_rows(snap, corr)
        assert len(result) == 1
        assert result[0].kind == "fill"

    def test_detects_clear(self):
        snap = [{"source": "a.pdf", "driver": "Иванов И.И.", "driver_conf": 1.0}]
        corr = [{"source": "a.pdf", "driver": ""}]
        result = diff_rows(snap, corr)
        assert len(result) == 1
        assert result[0].kind == "clear"

    def test_no_diff_when_identical(self):
        snap = [{"source": "a.pdf", "number": "001", "number_conf": 0.9}]
        corr = [{"source": "a.pdf", "number": "001"}]
        assert diff_rows(snap, corr) == []

    def test_multiple_fields_same_row(self):
        snap = [{
            "source": "a.pdf",
            "number": "1", "number_conf": 0.5,
            "shipper": "X", "shipper_conf": 0.5,
        }]
        corr = [{
            "source": "a.pdf",
            "number": "7145",
            "shipper": "ООО Y",
        }]
        result = diff_rows(snap, corr)
        assert {c.field for c in result} == {"number", "shipper"}

    def test_unknown_source_ignored(self):
        snap = [{"source": "a.pdf", "number": "001"}]
        corr = [{"source": "unknown.pdf", "number": "999"}]
        assert diff_rows(snap, corr) == []


class TestPersist:
    def test_append_and_load(self, tmp_path):
        log = tmp_path / "feedback" / "corrections.jsonl"
        corrections = [
            Correction(
                source="a.pdf", field="number",
                original="1", corrected="7145/Б",
                original_confidence=0.5, kind="fix",
            ),
        ]
        append_corrections(corrections, log)
        # Второй раз — append, не перезапись.
        append_corrections(
            [Correction(
                source="b.pdf", field="driver",
                original=MISSING, corrected="Иванов И.И.",
                original_confidence=0.0, kind="fill",
            )],
            log,
        )
        loaded = load_corrections(log)
        assert len(loaded) == 2
        assert loaded[0].field == "number"
        assert loaded[1].field == "driver"

    def test_load_empty(self, tmp_path):
        assert load_corrections(tmp_path / "nope.jsonl") == []
