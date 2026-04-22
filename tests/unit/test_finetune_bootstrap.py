"""Tests for scripts/finetune/bootstrap_dataset.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import fitz

from scripts.finetune import bootstrap_dataset as bs


def _make_pdf(path: Path, text: str = "ГЕКСАФОРМ 7813266190") -> None:
    doc = fitz.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((20, 100), text, fontsize=20)
    doc.save(str(path))
    doc.close()


def test_best_match_prefers_gt() -> None:
    gt = {"гексаформ", "7813266190"}
    # Off-by-one in a short token → corrected.
    assert bs._best_match("Гексафор", gt) == "гексаформ"
    # Already exact.
    assert bs._best_match("гексаформ", gt) == "гексаформ"
    # Too far → no match.
    assert bs._best_match("газпромнефть", gt) is None


def test_split_bucket_is_deterministic() -> None:
    assert bs._split_bucket("foo:1:2") == bs._split_bucket("foo:1:2")
    assert bs._split_bucket("x:0:0") in {"training", "validation"}


def test_load_gt_tokens_skips_short_and_markers(tmp_path: Path) -> None:
    (tmp_path / "doc1.txt").write_text(
        "====\nФАЙЛ: a\nООО ГЕКСАФОРМ ИНН 7813266190 29.08.2022 Z",
        encoding="utf-8",
    )
    tokens = bs._load_gt_tokens("doc1", inputs_dir=tmp_path)
    assert "гексаформ" in tokens
    assert "7813266190" in tokens
    assert "z" not in tokens  # single-char tokens filtered


def test_build_dataset_with_fake_reader(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _make_pdf(inputs / "doc1.pdf", "ГЕКСАФОРМ")
    (inputs / "doc1.txt").write_text(
        "ООО ГЕКСАФОРМ ИНН 7813266190", encoding="utf-8",
    )
    out = tmp_path / "ds"

    class FakeReader:
        def readtext(self, arr, **kwargs):
            h, w = arr.shape[:2]
            return [
                # EasyOCR shape: ([4-point bbox], text, conf)
                ([[10, 10], [150, 10], [150, 40], [10, 40]], "Гексаформ", 0.9),
                ([[10, 60], [150, 60], [150, 90], [10, 90]], "7813266190", 0.95),
                ([[10, 110], [20, 110], [20, 120], [10, 120]], "x", 0.1),  # below conf gate
            ]

    class FakeEasyOCR:
        def Reader(self, *args, **kwargs):  # noqa: N802 — mimic upstream API
            return FakeReader()

    with patch.dict("sys.modules", {"easyocr": FakeEasyOCR()}):
        rc = bs.build_dataset(inputs, out, dpi=150)

    assert rc == 0
    gt_lines = (out / "training" / "gt.txt").read_text(encoding="utf-8").splitlines()
    gt_lines += (out / "validation" / "gt.txt").read_text(encoding="utf-8").splitlines()
    # Two non-trivial crops → two labels total; both routed via bucket split.
    assert len(gt_lines) == 2
    labels = [line.split("\t")[1] for line in gt_lines]
    # 'Гексаформ' (conf 0.9) auto-corrects to 'гексаформ' from GT.
    assert "гексаформ" in labels
    assert "7813266190" in labels
