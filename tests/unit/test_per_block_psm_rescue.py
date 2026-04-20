"""Unit tests for :mod:`src.core.per_block_psm_rescue`."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.core.per_block_psm_rescue import (
    _compute_block_aggregates,
    _select_rescue_candidates,
    rescue_low_conf_blocks,
)


def _tsv(
    *rows: tuple[str, float, int, int, int, int, int, int, int, int],
) -> dict[str, list]:
    """``(text, conf, block, par, line, word_num, left, top, width, height)``."""
    return {
        "text":      [r[0] for r in rows],
        "conf":      [str(r[1]) for r in rows],
        "block_num": [r[2] for r in rows],
        "par_num":   [r[3] for r in rows],
        "line_num":  [r[4] for r in rows],
        "word_num":  [r[5] for r in rows],
        "left":      [r[6] for r in rows],
        "top":       [r[7] for r in rows],
        "width":     [r[8] for r in rows],
        "height":    [r[9] for r in rows],
    }


def _result_tsv(words: list[str], confs: list[float]) -> dict:
    """``image_to_data`` output shape — minimal fields the rescue reads."""
    return {
        "text":     list(words),
        "conf":     [str(c) for c in confs],
        "line_num": [1] * len(words),
        "word_num": list(range(len(words))),
    }


class TestComputeBlockAggregates:
    def test_single_block_aggregate(self) -> None:
        tsv = _tsv(
            ("hello", 80.0, 1, 1, 1, 1, 10, 20, 40, 15),
            ("world", 90.0, 1, 1, 1, 2, 60, 20, 40, 15),
        )
        agg = _compute_block_aggregates(tsv)
        assert 1 in agg
        assert agg[1]["word_count"] == 2
        assert agg[1]["mean_conf"] == 85.0
        assert agg[1]["bbox"] == (10, 20, 100, 35)

    def test_multiple_blocks(self) -> None:
        tsv = _tsv(
            ("a", 90.0, 1, 1, 1, 1, 10, 20, 30, 15),
            ("b", 85.0, 1, 1, 1, 2, 50, 20, 30, 15),
            ("c", 40.0, 2, 1, 1, 1, 10, 50, 30, 15),
            ("d", 50.0, 2, 1, 1, 2, 50, 50, 30, 15),
        )
        agg = _compute_block_aggregates(tsv)
        assert set(agg.keys()) == {1, 2}
        assert agg[1]["mean_conf"] == 87.5
        assert agg[2]["mean_conf"] == 45.0

    def test_negative_conf_rows_ignored(self) -> None:
        tsv = _tsv(
            ("",   -1.0, 1, 1, 1, 0,  0,  0, 0, 0),  # placeholder
            ("real", 80.0, 1, 1, 1, 1, 10, 20, 40, 15),
        )
        agg = _compute_block_aggregates(tsv)
        assert agg[1]["word_count"] == 1
        assert agg[1]["mean_conf"] == 80.0

    def test_empty_tsv(self) -> None:
        assert _compute_block_aggregates({"text": [], "conf": []}) == {}


class TestSelectRescueCandidates:
    def test_low_conf_with_enough_words_selected(self) -> None:
        agg = {
            1: {"mean_conf": 50.0, "word_count": 5},
            2: {"mean_conf": 90.0, "word_count": 5},
        }
        assert _select_rescue_candidates(agg) == [1]

    def test_low_conf_short_block_rejected(self) -> None:
        """2-word block below the 3-word minimum."""
        agg = {1: {"mean_conf": 40.0, "word_count": 2}}
        assert _select_rescue_candidates(agg) == []

    def test_high_conf_block_not_selected(self) -> None:
        agg = {1: {"mean_conf": 75.0, "word_count": 10}}
        assert _select_rescue_candidates(agg) == []


class TestRescueLowConfBlocks:
    def test_replaces_tsv_when_rescue_wins(self) -> None:
        """Rescue swaps block 1's text with PSM=6 re-OCR output
        when the re-OCR mean conf clears the 5-point lift."""
        tsv = _tsv(
            ("garbage1", 45.0, 1, 1, 1, 1, 10, 20, 40, 15),
            ("garbage2", 50.0, 1, 1, 1, 2, 60, 20, 40, 15),
            ("garbage3", 55.0, 1, 1, 1, 3, 10, 40, 40, 15),
            # Different high-conf block — shouldn't be touched.
            ("clean",    92.0, 2, 1, 1, 1, 10, 100, 40, 15),
            ("prose",    88.0, 2, 1, 1, 2, 60, 100, 40, 15),
            ("here",     90.0, 2, 1, 1, 3, 110, 100, 40, 15),
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        def _fake(crop, lang, config, output_type):
            # PSM=6 pass returns cleaner text at higher conf.
            return _result_tsv(
                ["Иван", "Иванов", "подпись"],
                [85.0, 82.0, 80.0],
            )

        with patch(
            "src.core.per_block_psm_rescue.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = _fake
            mock_py.Output.DICT = "dict"
            rescued = rescue_low_conf_blocks(
                tsv, image, "rus+eng", "--oem 1",
            )

        assert rescued == 1
        # Block 1 entries swapped to new words.
        assert tsv["text"][0] == "Иван"
        assert tsv["text"][1] == "Иванов"
        assert tsv["text"][2] == "подпись"
        # Block 2 (high-conf) left untouched.
        assert tsv["text"][3] == "clean"
        assert tsv["text"][4] == "prose"
        assert tsv["text"][5] == "here"

    def test_no_swap_when_lift_too_small(self) -> None:
        tsv = _tsv(
            ("foo", 50.0, 1, 1, 1, 1, 10, 20, 40, 15),
            ("bar", 55.0, 1, 1, 1, 2, 60, 20, 40, 15),
            ("baz", 52.0, 1, 1, 1, 3, 110, 20, 40, 15),
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        def _fake(crop, lang, config, output_type):
            # Mean 54 vs original 52.3 — only 1.7 lift < 5-pt floor.
            return _result_tsv(["xxx", "yyy", "zzz"], [54.0, 54.0, 54.0])

        with patch(
            "src.core.per_block_psm_rescue.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = _fake
            mock_py.Output.DICT = "dict"
            rescued = rescue_low_conf_blocks(
                tsv, image, "rus+eng", "",
            )
        assert rescued == 0
        assert tsv["text"][0] == "foo"

    def test_blanks_leftover_slots_when_fewer_new_words(self) -> None:
        """Primary had 4 words in the block; PSM=6 recovered only 2.
        The extra two slots get blanked (text = "", conf = -1) so
        the downstream filter skips them."""
        tsv = _tsv(
            ("w1", 40.0, 1, 1, 1, 1, 10, 20, 30, 15),
            ("w2", 42.0, 1, 1, 1, 2, 50, 20, 30, 15),
            ("w3", 45.0, 1, 1, 1, 3, 90, 20, 30, 15),
            ("w4", 48.0, 1, 1, 1, 4, 130, 20, 30, 15),
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        with patch(
            "src.core.per_block_psm_rescue.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = (
                lambda *a, **kw: _result_tsv(
                    ["clean", "words"], [88.0, 85.0],
                )
            )
            mock_py.Output.DICT = "dict"
            rescued = rescue_low_conf_blocks(
                tsv, image, "rus+eng", "",
            )
        assert rescued == 1
        assert tsv["text"][0] == "clean"
        assert tsv["text"][1] == "words"
        assert tsv["text"][2] == ""
        assert tsv["text"][3] == ""
        assert tsv["conf"][2] == "-1"
        assert tsv["conf"][3] == "-1"

    def test_image_none_returns_zero(self) -> None:
        tsv = _tsv(
            ("x", 40.0, 1, 1, 1, 1, 10, 20, 30, 15),
            ("y", 42.0, 1, 1, 1, 2, 50, 20, 30, 15),
            ("z", 45.0, 1, 1, 1, 3, 90, 20, 30, 15),
        )
        assert rescue_low_conf_blocks(tsv, None, "rus+eng", "") == 0

    def test_pytesseract_exception_returns_zero_rescued(self) -> None:
        tsv = _tsv(
            ("x", 40.0, 1, 1, 1, 1, 10, 20, 30, 15),
            ("y", 42.0, 1, 1, 1, 2, 50, 20, 30, 15),
            ("z", 45.0, 1, 1, 1, 3, 90, 20, 30, 15),
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        with patch(
            "src.core.per_block_psm_rescue.pytesseract"
        ) as mock_py:
            mock_py.image_to_data.side_effect = RuntimeError("boom")
            mock_py.Output.DICT = "dict"
            rescued = rescue_low_conf_blocks(
                tsv, image, "rus+eng", "",
            )
        assert rescued == 0
        # Original TSV untouched on failure.
        assert tsv["text"][0] == "x"

    def test_no_candidates_returns_zero(self) -> None:
        """All blocks above threshold — no rescue attempt."""
        tsv = _tsv(
            ("a", 90.0, 1, 1, 1, 1, 10, 20, 30, 15),
            ("b", 88.0, 1, 1, 1, 2, 50, 20, 30, 15),
            ("c", 91.0, 1, 1, 1, 3, 90, 20, 30, 15),
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        with patch(
            "src.core.per_block_psm_rescue.pytesseract"
        ) as mock_py:
            rescued = rescue_low_conf_blocks(
                tsv, image, "rus+eng", "",
            )
            mock_py.image_to_data.assert_not_called()
        assert rescued == 0
