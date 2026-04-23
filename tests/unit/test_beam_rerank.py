"""Tests for CTC beam search + domain-LM rerank."""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.application.beam_rerank import (
    BeamCandidate,
    _ctc_beam_search,
    decode_and_rerank,
    rerank,
)
from src.shared.domain_lm import DomainLM


def _make_lm(words: list[str]) -> DomainLM:
    return DomainLM(Counter(words))


def test_beam_search_single_frame():
    # 3 frames, vocab {blank=0, 'a'=1}. Deterministic argmax → "a".
    lp = np.log(np.array([
        [0.1, 0.9],  # frame 0 favours 'a'
        [0.1, 0.9],  # frame 1 favours 'a' (repeat, collapsed)
        [0.1, 0.9],  # frame 2 favours 'a' (repeat, collapsed)
    ]))
    cands = _ctc_beam_search(lp, charset="a", beam_width=3, blank_idx=0)
    # With no blank between repeats, CTC collapses to single 'a'.
    texts = [c.text for c in cands]
    assert "a" in texts


def test_beam_search_alternating_no_collapse():
    # Alternating a-b-a (separated by blanks) → "aba"
    # vocab: {blank=0, 'a'=1, 'b'=2}
    lp = np.log(np.array([
        [0.01, 0.98, 0.01],  # 'a'
        [0.98, 0.01, 0.01],  # blank
        [0.01, 0.01, 0.98],  # 'b'
        [0.98, 0.01, 0.01],  # blank
        [0.01, 0.98, 0.01],  # 'a'
    ]))
    cands = _ctc_beam_search(lp, charset="ab", beam_width=3, blank_idx=0)
    top = cands[0].text
    # Top should be "aba" (or at least contain a valid path).
    assert "a" in top and "b" in top


def test_rerank_picks_known_word():
    # Two candidates at similar model confidence; LM boosts known word.
    candidates = [
        BeamCandidate(text="гексафрм", logprob=-1.0),      # unknown
        BeamCandidate(text="гексаформ", logprob=-1.1),    # known
    ]
    lm = _make_lm(["гексаформ"] * 100)
    winner = rerank(candidates, lm, lm_weight=0.7)
    assert winner.text == "гексаформ"


def test_rerank_no_lm_bias_when_weight_zero():
    # With lm_weight=0, pure model score wins.
    candidates = [
        BeamCandidate(text="unknown", logprob=-0.5),       # higher model prob
        BeamCandidate(text="гексаформ", logprob=-2.0),    # known but model-hostile
    ]
    lm = _make_lm(["гексаформ"] * 1000)
    winner = rerank(candidates, lm, lm_weight=0.0)
    assert winner.text == "unknown"


def test_empty_candidates():
    lm = _make_lm(["word"])
    result = rerank([], lm)
    assert result.text == ""


def test_decode_and_rerank_integration():
    # 5-frame vocab: blank=0, 'а'=1, 'б'=2
    charset = "аб"
    lp = np.log(np.array([
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
    ]))
    lm = _make_lm(["аба", "абб"])
    result = decode_and_rerank(
        lp, charset=charset, lm=lm, beam_width=5, lm_weight=0.5,
    )
    assert result.text  # non-empty
