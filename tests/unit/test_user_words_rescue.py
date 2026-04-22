"""Tests for :mod:`src.core.user_words_rescue` on the rapidfuzz path.

The module switched from pure-Python Levenshtein to rapidfuzz's
C-native ``process.extractOne`` when the bundled dictionary grew to
2M forms (the pure-Python walk became multi-second per page on
``universal_accurate`` profiles). These tests lock the new behaviour:

* Small in-memory catalogs still match to 1-2 edit distance.
* Exact hits short-circuit without touching closest().
* Tokens outside the rescue confidence band are returned unchanged.
* Catalog lookups are case-fold insensitive for the membership check.
"""

from __future__ import annotations

import pytest

from src.core.user_words_rescue import (
    UserWordsCatalog,
    user_words_fuzzy_rescue,
)

pytest.importorskip(
    "rapidfuzz",
    reason="rapidfuzz required for the closest() lookup path",
)


def _cat(words: list[str]) -> UserWordsCatalog:
    return UserWordsCatalog(words)


def test_exact_match_returns_original() -> None:
    cat = _cat(["ИНН", "КПП", "ОГРН"])
    assert cat.contains("ИНН") is True
    assert cat.contains("инн") is True  # casefold
    assert cat.contains("ОАО") is False


def test_closest_finds_single_edit() -> None:
    cat = _cat(["грузоотправитель", "грузополучатель", "перевозчик"])
    # One-edit typo: ь → ъ at the tail (same length, one substitution).
    hit = cat.closest("грузоотправителъ")
    assert hit is not None
    match, dist = hit
    assert match == "грузоотправитель"
    assert dist == 1


def test_closest_respects_edit_budget_on_short_tokens() -> None:
    # 3-char tokens: max_edits=1. «ЗАЗ» vs «КАЗ» = 1 edit → match.
    cat = _cat(["КАЗ", "ЗИЛ", "МАЗ"])
    hit = cat.closest("ЗАЗ")
    assert hit is not None
    _match, dist = hit
    assert dist == 1


def test_closest_rejects_too_far() -> None:
    # 3-char tokens: max_edits=1. «АБВ» vs «ЯЯЯ» = 3 edits → no match.
    cat = _cat(["АБВ"])
    assert cat.closest("ЯЯЯ") is None


def test_closest_returns_none_on_empty_input() -> None:
    cat = _cat(["документ"])
    assert cat.closest("") is None


def test_rescue_skips_below_min_conf() -> None:
    cat = _cat(["документ"])
    # Conf too low → pass through unchanged regardless of catalog hit.
    word, conf = user_words_fuzzy_rescue("докумэнт", 20.0, cat)
    assert word == "докумэнт"
    assert conf == 20.0


def test_rescue_skips_above_max_conf() -> None:
    cat = _cat(["документ"])
    # Conf too high → line-level read is trusted.
    word, conf = user_words_fuzzy_rescue("докумэнт", 95.0, cat)
    assert word == "докумэнт"
    assert conf == 95.0


def test_rescue_swaps_in_band() -> None:
    cat = _cat(["документ"])
    # Conf in [30, 75] → fuzzy match fires, conf floors at 85.
    word, conf = user_words_fuzzy_rescue("докумэнт", 55.0, cat)
    assert word == "документ"
    assert conf >= 85.0


def test_rescue_preserves_original_case_mapping() -> None:
    """Catalog keeps the original-case form; the rescuer returns it verbatim."""
    cat = _cat(["ГЕКСАФОРМ"])
    word, conf = user_words_fuzzy_rescue("ГЕКСАФОРМа", 55.0, cat)
    assert word == "ГЕКСАФОРМ"
    assert conf >= 85.0


def test_large_catalog_still_fast() -> None:
    """Smoke: a 10k-entry catalog completes closest() in well under 1s.

    Regression guard against accidentally reverting to the pure-Python
    Levenshtein walk — the old O(N) loop on 2M forms ran 100×+ slower.
    """
    import time

    words = [f"слово{i:05d}" for i in range(10_000)]
    cat = _cat(words)
    t0 = time.perf_counter()
    for _ in range(50):
        cat.closest("слово01234х")  # 1-edit typo
    elapsed = time.perf_counter() - t0
    # 50 lookups on 10k entries must finish in under 1 second on
    # any reasonable machine. The pure-Python Levenshtein fallback
    # would take ~20x this. 1s gives huge headroom for slow runners.
    assert elapsed < 1.0, f"closest() too slow: {elapsed:.2f}s"
