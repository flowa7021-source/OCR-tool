"""Unit tests for :mod:`src.core.user_words_rescue`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.user_words_rescue import (
    UserWordsCatalog,
    _levenshtein,
    load_user_words_catalog,
    user_words_fuzzy_rescue,
)


class TestLevenshtein:
    def test_equal_strings(self) -> None:
        assert _levenshtein("hello", "hello") == 0

    def test_empty_strings(self) -> None:
        assert _levenshtein("", "") == 0
        assert _levenshtein("abc", "") == 3
        assert _levenshtein("", "xyz") == 3

    def test_single_substitution(self) -> None:
        assert _levenshtein("ИНН", "ИНЦ") == 1

    def test_insertion(self) -> None:
        assert _levenshtein("КПП", "КППI") == 1

    def test_deletion(self) -> None:
        assert _levenshtein("ТЕСТ", "ТЕТ") == 1

    def test_multiple_edits(self) -> None:
        # ``Scania`` (all Latin) vs ``Скаnia`` (С, к, а Cyrillic,
        # n/i/a Latin): 3 substitutions by code point — the visually-
        # identical a and the N+i pair are actually different Unicode
        # characters across the scripts.
        assert _levenshtein("Scania", "Скаnia") == 3


class TestUserWordsCatalog:
    def test_empty(self) -> None:
        c = UserWordsCatalog([])
        assert len(c) == 0
        assert c.closest("anything") is None

    def test_exact_match(self) -> None:
        c = UserWordsCatalog(["ИНН", "КПП", "ОГРН"])
        assert c.contains("ИНН")
        assert c.contains("КПП")
        assert not c.contains("ZZZ")

    def test_casefold_match(self) -> None:
        c = UserWordsCatalog(["ГЕКСАФОРМ"])
        assert c.contains("гексаформ")  # lowercase still matches via casefold

    def test_closest_finds_nearest(self) -> None:
        c = UserWordsCatalog(["ИНН", "КПП", "ОГРН", "СНИЛС"])
        result = c.closest("ИНЦ")
        assert result is not None
        assert result[0] == "ИНН"
        assert result[1] == 1

    def test_closest_none_when_beyond_budget(self) -> None:
        """3-char word with distance ≥ 2 exceeds the short-word budget."""
        c = UserWordsCatalog(["ИНН"])
        result = c.closest("XYZ")  # distance 3
        assert result is None

    def test_closest_accepts_distance_2_for_long_words(self) -> None:
        """Words ≥ 6 chars accept up to distance 2."""
        c = UserWordsCatalog(["грузоотправитель"])
        result = c.closest("грузощправитель")  # 1 substitution
        assert result is not None
        assert result[0] == "грузоотправитель"

    def test_length_filtering(self) -> None:
        """Candidates of wildly different length shouldn't win."""
        c = UserWordsCatalog(["ABC", "ABCDEFGHIJ"])
        # 8-char query: 3-char entries are at distance 5+, outside budget.
        result = c.closest("ABCDEFGH")
        assert result is not None
        assert result[0] == "ABCDEFGHIJ"  # length 10, distance 2 = within budget

    def test_from_file(self, tmp_path: Path) -> None:
        """Reads entries from disk, skips blank lines."""
        p = tmp_path / "words.rus"
        p.write_text("ИНН\n\nКПП\n  \nОГРН\n", encoding="utf-8")
        c = UserWordsCatalog.from_file(p)
        assert len(c) == 3
        assert c.contains("ИНН")

    def test_from_file_missing(self, tmp_path: Path) -> None:
        """Missing file yields an empty catalog, not a crash."""
        c = UserWordsCatalog.from_file(tmp_path / "nope.rus")
        assert len(c) == 0


class TestUserWordsFuzzyRescue:
    @pytest.fixture
    def catalog(self) -> UserWordsCatalog:
        return UserWordsCatalog([
            "ИНН", "КПП", "ОГРН", "ГЕКСАФОРМ",
            "Scania", "TENSAR", "договор",
            "грузоотправитель",
        ])

    def test_swaps_typo_to_canonical(
        self, catalog: UserWordsCatalog
    ) -> None:
        word, conf = user_words_fuzzy_rescue("ИНЦ", 55.0, catalog)
        assert word == "ИНН"
        assert conf == 85.0  # bumped to floor

    def test_keeps_text_when_already_canonical(
        self, catalog: UserWordsCatalog
    ) -> None:
        """Already-correct word: keep text, bump conf to floor."""
        word, conf = user_words_fuzzy_rescue("ИНН", 55.0, catalog)
        assert word == "ИНН"
        assert conf == 85.0

    def test_casefold_exact_match_keeps_original(
        self, catalog: UserWordsCatalog
    ) -> None:
        """Lowercase variant is already canonical via casefold — no swap."""
        word, conf = user_words_fuzzy_rescue("гексаформ", 55.0, catalog)
        assert word == "гексаформ"  # preserved (casefold match, not swap)
        assert conf == 85.0

    def test_long_word_accepts_distance_2(
        self, catalog: UserWordsCatalog
    ) -> None:
        word, conf = user_words_fuzzy_rescue(
            "грузощправитель", 55.0, catalog,
        )
        assert word == "грузоотправитель"

    def test_skips_below_conf_band(
        self, catalog: UserWordsCatalog
    ) -> None:
        word, conf = user_words_fuzzy_rescue("ИНЦ", 20.0, catalog)
        assert word == "ИНЦ"
        assert conf == 20.0

    def test_skips_above_conf_band(
        self, catalog: UserWordsCatalog
    ) -> None:
        """Above 75 %: line-level result is trusted, don't churn."""
        word, conf = user_words_fuzzy_rescue("ИНЦ", 85.0, catalog)
        assert word == "ИНЦ"
        assert conf == 85.0

    def test_skips_short_word(self, catalog: UserWordsCatalog) -> None:
        """2-char tokens at Levenshtein 1 from every 3-char entry
        would false-positive."""
        word, conf = user_words_fuzzy_rescue("ИН", 55.0, catalog)
        assert word == "ИН"

    def test_skips_no_match_beyond_budget(
        self, catalog: UserWordsCatalog
    ) -> None:
        word, conf = user_words_fuzzy_rescue("ZZZZZ", 55.0, catalog)
        assert word == "ZZZZZ"
        assert conf == 55.0

    def test_empty_catalog_keeps_original(self) -> None:
        empty = UserWordsCatalog([])
        word, conf = user_words_fuzzy_rescue("ИНЦ", 55.0, empty)
        assert word == "ИНЦ"
        assert conf == 55.0


class TestLoadUserWordsCatalog:
    def test_none_path_returns_none(self) -> None:
        assert load_user_words_catalog(None) is None

    def test_missing_path_returns_none(self, tmp_path: Path) -> None:
        assert load_user_words_catalog(tmp_path / "missing.rus") is None

    def test_real_file_returns_catalog(self, tmp_path: Path) -> None:
        p = tmp_path / "w.rus"
        p.write_text("A\nB\nC\n", encoding="utf-8")
        c = load_user_words_catalog(p)
        assert c is not None
        assert len(c) == 3
