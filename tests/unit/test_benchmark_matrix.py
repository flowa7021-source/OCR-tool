"""Unit tests for :mod:`scripts.benchmark_matrix` helpers.

Exercises the pure-Python bits that don't touch the OCR pipeline:
corpus discovery, CSV output, best-cell summary. The OCR-exercising
paths are covered indirectly by ``test_accuracy_benchmark.py`` —
duplicating those here would slow the unit suite to a crawl.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover — path bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmark_matrix import (  # noqa: E402
    Cell,
    _load_user_corpus,
    _write_csv,
)


class TestLoadUserCorpus:
    def test_pairs_each_pdf_with_matching_gt(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.pdf").write_bytes(b"%PDF-1.7\n")
        (tmp_path / "alpha.gt.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "beta.pdf").write_bytes(b"%PDF-1.7\n")
        (tmp_path / "beta.gt.txt").write_text("world", encoding="utf-8")

        docs = _load_user_corpus(tmp_path)

        names = sorted(d.name for d in docs)
        assert names == ["alpha", "beta"]
        by_name = {d.name: d for d in docs}
        assert by_name["alpha"].ground_truth == "hello"
        assert by_name["beta"].ground_truth == "world"

    def test_pdf_without_gt_is_skipped(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "orphan.pdf").write_bytes(b"%PDF-1.7\n")
        (tmp_path / "paired.pdf").write_bytes(b"%PDF-1.7\n")
        (tmp_path / "paired.gt.txt").write_text("ok", encoding="utf-8")

        docs = _load_user_corpus(tmp_path)
        assert [d.name for d in docs] == ["paired"]
        # The orphan pdf should have produced a warning on stderr so
        # the user knows WHY it didn't show up in the run.
        err = capsys.readouterr().err
        assert "orphan.pdf" in err

    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as exc_info:
            _load_user_corpus(tmp_path)
        msg = str(exc_info.value)
        assert "No PDF" in msg


class TestWriteCsv:
    def test_header_and_row_layout(self, tmp_path: Path) -> None:
        cells = [
            Cell(
                document="doc1",
                profile="universal_accurate",
                dpi=400,
                cer=0.012,
                wer=0.03,
                mean_conf=92.5,
                elapsed_sec=3.4,
            ),
            Cell(
                document="doc1",
                profile="quick_reliable",
                dpi=300,
                cer=0.04,
                wer=0.08,
                mean_conf=81.2,
                elapsed_sec=2.1,
                error="retry tier fallback",
            ),
        ]
        out = tmp_path / "result.csv"
        _write_csv(cells, out)

        with out.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0] == [
            "document", "profile", "dpi", "cer", "wer",
            "mean_conf", "elapsed_sec", "error",
        ]
        assert rows[1] == [
            "doc1", "universal_accurate", "400", "0.012",
            "0.03", "92.5", "3.4", "",
        ]
        assert rows[2][-1] == "retry tier fallback"
