"""Tests for the confidence-weighted OCR ensemble."""

from __future__ import annotations

from src.application.confidence_ensemble import MergeStats, merge_results


def _bbox(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_empty_inputs():
    merged, stats = merge_results([], [])
    assert merged == []
    assert stats.total_a == 0 and stats.total_b == 0


def test_a_only_pass_through():
    a = [(_bbox(0, 0, 10, 10), "hello", 0.9)]
    merged, stats = merge_results(a, [])
    assert merged == a
    assert stats.from_a == 1
    assert stats.from_b == 0


def test_consensus_keeps_max_conf():
    a = [(_bbox(0, 0, 10, 10), "hello", 0.8)]
    b = [(_bbox(1, 1, 11, 11), "hello", 0.6)]
    merged, stats = merge_results(a, b)
    assert len(merged) == 1
    assert merged[0][1] == "hello"
    assert merged[0][2] == 0.8
    assert stats.consensus == 1


def test_conflict_higher_conf_wins():
    a = [(_bbox(0, 0, 10, 10), "hello", 0.5)]
    b = [(_bbox(1, 1, 11, 11), "HELLO", 0.9)]
    merged, stats = merge_results(a, b)
    assert len(merged) == 1
    assert merged[0][1] == "HELLO"
    assert merged[0][2] == 0.9
    assert stats.conflicts == 1


def test_no_overlap_both_kept():
    a = [(_bbox(0, 0, 10, 10), "top-left", 0.9)]
    b = [(_bbox(100, 100, 110, 110), "bottom-right", 0.9)]
    merged, stats = merge_results(a, b)
    assert len(merged) == 2
    assert {m[1] for m in merged} == {"top-left", "bottom-right"}


def test_low_iou_treated_as_separate():
    # Same bbox vertical area but no overlap
    a = [(_bbox(0, 0, 10, 10), "left", 0.9)]
    b = [(_bbox(20, 0, 30, 10), "right", 0.9)]
    merged, stats = merge_results(a, b, iou_threshold=0.3)
    assert len(merged) == 2


def test_stats_serialisation():
    s = MergeStats(total_a=3, total_b=2, from_a=3, from_b=0,
                   consensus=2, conflicts=0)
    d = s.to_dict()
    assert d["total_a"] == 3
    assert d["consensus"] == 2
