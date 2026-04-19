"""Unit tests for pure helpers in :mod:`scripts.benchmark_universal`.

The full benchmark spins up the OCR pipeline + real Tesseract — too
heavy for per-PR CI and covered by manual dev-loop use. But the
formatting + aggregation helpers are pure Python, cheap to test,
and exactly the ones that'll silently drift if the script is
refactored. These tests pin their output shapes so any breakage
surfaces on the next CI run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_BENCH_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "benchmark_universal.py"
# Load the script as a module without importing any heavy pipeline
# dependencies — we only need the pure helpers.
_spec = importlib.util.spec_from_file_location("benchmark_universal", _BENCH_PATH)
assert _spec and _spec.loader
_bench = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_universal"] = _bench
_spec.loader.exec_module(_bench)


class TestHistogram:
    """``_histogram`` buckets a list of confidences into 0-100 bands."""

    def test_default_10_buckets(self) -> None:
        confs = [5.0, 15.0, 35.0, 55.0, 75.0, 95.0]
        hist = _bench._histogram(confs)
        # 10 buckets of width 10: one entry per band we populated.
        assert len(hist) == 10
        # Each value lands in exactly one bucket.
        assert sum(hist) == len(confs)
        # Every populated band contains exactly one sample.
        populated = [i for i, c in enumerate(hist) if c > 0]
        assert populated == [0, 1, 3, 5, 7, 9]

    def test_ceiling_value_caps_in_last_bucket(self) -> None:
        """100 % confidence is a valid Tesseract value; the histogram
        must not IndexError trying to put it in bucket ``10`` of a
        10-bucket array. It caps in the last bucket instead."""
        hist = _bench._histogram([100.0])
        assert hist[-1] == 1
        assert sum(hist) == 1

    def test_empty_input_returns_all_zeros(self) -> None:
        hist = _bench._histogram([])
        assert hist == [0] * 10


class TestTopWords:
    """``_top_words`` aggregates by word + mean conf + count."""

    def test_drops_side_prefers_low_conf_first(self) -> None:
        words = ["foo", "bar", "foo"]
        confs = [20.0, 10.0, 30.0]
        # ``foo`` mean = 25, ``bar`` mean = 10 — both below 60.
        top = _bench._top_words(
            words, confs, threshold=60.0, keep_side="below", n=2,
        )
        # ``bar`` has the lowest mean_conf so it appears first.
        assert [w for w, _, _ in top] == ["bar", "foo"]
        # Aggregated counts: foo × 2, bar × 1.
        counts = {w: c for w, _, c in top}
        assert counts == {"foo": 2, "bar": 1}

    def test_keeps_side_prefers_high_conf_first(self) -> None:
        words = ["foo", "bar"]
        confs = [80.0, 95.0]
        top = _bench._top_words(
            words, confs, threshold=60.0, keep_side="above", n=2,
        )
        # ``bar`` has the highest mean_conf so it appears first.
        assert [w for w, _, _ in top] == ["bar", "foo"]

    def test_threshold_boundary_in_kept_side(self) -> None:
        """A word at exactly ``threshold`` is KEPT (same convention as
        ``_compute_confidences`` — ``conf >= threshold``)."""
        words = ["border"]
        confs = [60.0]
        assert _bench._top_words(
            words, confs, threshold=60.0, keep_side="above", n=10,
        ) == [("border", 60.0, 1)]
        assert _bench._top_words(
            words, confs, threshold=60.0, keep_side="below", n=10,
        ) == []

    def test_limits_to_n_entries(self) -> None:
        """``n`` caps the returned list even when more distinct words
        qualify — without this the CLI output bloats on long scans."""
        words = [f"w{i}" for i in range(50)]
        confs = [5.0 + i for i in range(50)]
        top = _bench._top_words(
            words, confs, threshold=60.0, keep_side="below", n=5,
        )
        assert len(top) == 5


class TestFormatHistogram:
    """Smoke: the ASCII bar chart is printable and contains bucket labels."""

    def test_emits_one_line_per_bucket(self) -> None:
        out = _bench._format_histogram([1] * 10)
        assert out.count("\n") == 9  # 10 lines → 9 newlines between them

    def test_bucket_labels_cover_zero_to_hundred(self) -> None:
        out = _bench._format_histogram([1] * 10)
        # First bucket starts at 0, last at 90.
        assert "[  0-" in out
        assert "[ 90-100]" in out
