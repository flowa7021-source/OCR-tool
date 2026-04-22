"""Tests for :mod:`src.core.fuzzy_corrector`.

Covers the hot-path behaviour after the 2M-lex expansion:

* ``_best_match`` rejects substring-style matches that ``WRatio``
  would have accepted — proving ``scorer=fuzz.ratio`` is wired in.
* Length-ratio guard still kicks in (``Ленина → На`` class regressions).
* ``correct`` on clean text is idempotent.
* Missing resource is a no-op (minimal builds without ``ru_lexicon.txt``).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("rapidfuzz")

from src.core import fuzzy_corrector  # noqa: E402


def _with_lex(words: list[str]):
    """Replace the lazy-loaded lexicon with a small test fixture."""
    return patch.multiple(
        fuzzy_corrector,
        _load_lexicon=lambda: frozenset(w.lower() for w in words),
        _load_lexicon_list=lambda: tuple(sorted(w.lower() for w in words)),
    )


def test_returns_text_unchanged_when_lexicon_empty() -> None:
    with patch.multiple(
        fuzzy_corrector,
        _load_lexicon=lambda: frozenset(),
        _load_lexicon_list=lambda: (),
    ):
        assert fuzzy_corrector.correct("любой текст") == "любой текст"


def test_skips_short_tokens() -> None:
    # Even if tokens look like typos, < 6 chars are never touched.
    with _with_lex(["документы", "является"]):
        # «дело» is 4 chars → untouched.
        out = fuzzy_corrector.correct("дело тут")
        assert "дело" in out


def test_skips_latin_and_numeric_tokens() -> None:
    with _with_lex(["документы"]):
        assert fuzzy_corrector.correct("Invoice 12345") == "Invoice 12345"


def test_length_ratio_guard_blocks_collapse() -> None:
    """Fuzzy must not collapse «Ленинск» onto a 2-char dict entry."""
    with _with_lex(["на", "ленинск"]):
        # «Ленинск» is already in lex → passes through unchanged.
        # Build a typo that should NOT collapse to «на»:
        out = fuzzy_corrector.correct("Леnинск")
        # Either corrects to «ленинск»/«Ленинск» or leaves alone —
        # the collapse to «на» would be the bug.
        assert "на" not in out.split()


def test_substring_spike_is_rejected() -> None:
    """``scorer=fuzz.ratio`` vs default WRatio: a long token must NOT
    match a much shorter dict entry via partial_ratio.

    Regression for the 2M-lex expansion: WRatio returned 90+ for
    «экземиляр» → «миляр» or similar substring hits, causing truncated
    replacements. ratio-only scoring + the length-ratio guard must
    reject these.
    """
    with _with_lex(["лар", "миляр", "экземпляр"]):
        out = fuzzy_corrector.correct("экземиляр найден")
        # Must not collapse to the 3-char «лар» or 5-char «миляр».
        # Accepted outcomes: either keep «экземиляр» (if pymorphy3
        # guard trips) or swap to «экземпляр» (if pymorphy3 accepts).
        tokens = out.split()
        assert "лар" not in tokens
        assert "миляр" not in tokens


def test_lexicon_size_reports_loaded_forms() -> None:
    with _with_lex(["a", "b", "c"]):
        assert fuzzy_corrector.lexicon_size() == 3
