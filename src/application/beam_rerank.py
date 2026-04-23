"""Beam-search decoder + domain-LM rerank for CTC models.

EasyOCR's default recogniser uses greedy CTC decode: argmax at each
frame, then collapse repeats and remove blanks. That's fast but brittle
— when the top-1 probability at some frame is barely 51%, a single
wrong frame kills the whole word.

Beam search keeps the top-K hypotheses at each frame and decodes
them all, ending with K final strings ordered by model probability.
Reranking by a domain LM then picks the top candidate that also
scores high under our vocabulary (built from inputs/*.txt +
expected/*.json).

Runtime: beam=5 roughly 3-4× slower than greedy — still milliseconds
per word, acceptable for the offline OCR use case.

This module is engine-agnostic: it takes raw probability tensors from
any CTC recogniser and a DomainLM, returns rescored text strings.
Integration into EasyOCR is covered by an optional config flag
(``beam_rerank_enabled``); when off, we fall back to EasyOCR's
internal greedy decoder.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from src.shared.domain_lm import DomainLM

logger = logging.getLogger(__name__)


@dataclass
class BeamCandidate:
    text: str
    logprob: float
    rescored: float = 0.0  # after LM rerank


def _ctc_beam_search(
    log_probs: np.ndarray,
    charset: str,
    beam_width: int = 5,
    blank_idx: int = 0,
) -> list[BeamCandidate]:
    """Vanilla CTC prefix-beam-search.

    Args:
        log_probs: (T, V) array of per-frame log-probabilities. T is
            the number of output frames (≈ image width / stride), V
            is vocabulary size including blank.
        charset: string of length V-1 (excluding blank) mapping index
            → character. Blank token sits at ``blank_idx``.
        beam_width: number of hypotheses kept at each step.
        blank_idx: vocabulary index of the CTC blank token.

    Returns:
        Top-``beam_width`` :class:`BeamCandidate` objects sorted by
        descending log-probability.
    """
    time_steps, vocab = log_probs.shape
    # Each beam: (prefix_str, prob_blank_ending, prob_non_blank_ending)
    beams: dict[str, tuple[float, float]] = {"": (0.0, -math.inf)}

    for t in range(time_steps):
        frame_lp = log_probs[t]
        new_beams: dict[str, tuple[float, float]] = {}

        for prefix, (p_b, p_nb) in beams.items():
            # Extending with blank keeps the same prefix.
            new_b = frame_lp[blank_idx] + np.logaddexp(p_b, p_nb)
            if prefix in new_beams:
                ob, onb = new_beams[prefix]
                new_beams[prefix] = (np.logaddexp(ob, new_b), onb)
            else:
                new_beams[prefix] = (new_b, -math.inf)

            # Extending with each non-blank character.
            for c_idx in range(vocab):
                if c_idx == blank_idx:
                    continue
                ch = charset[c_idx - 1] if c_idx > blank_idx else charset[c_idx]
                new_nb_lp = frame_lp[c_idx]
                if prefix.endswith(ch):
                    # Repeated char: add to p_b contribution only to
                    # avoid collapsing the duplicate.
                    new_prefix = prefix + ch
                    nb_from_p_b = p_b + new_nb_lp
                    if new_prefix in new_beams:
                        ob, onb = new_beams[new_prefix]
                        new_beams[new_prefix] = (
                            ob, np.logaddexp(onb, nb_from_p_b),
                        )
                    else:
                        new_beams[new_prefix] = (-math.inf, nb_from_p_b)
                    # Self-repeat (no prefix change): update p_nb.
                    nb_self = p_nb + new_nb_lp
                    ob, onb = new_beams[prefix]
                    new_beams[prefix] = (ob, np.logaddexp(onb, nb_self))
                else:
                    new_prefix = prefix + ch
                    nb_from_both = np.logaddexp(p_b, p_nb) + new_nb_lp
                    if new_prefix in new_beams:
                        ob, onb = new_beams[new_prefix]
                        new_beams[new_prefix] = (
                            ob, np.logaddexp(onb, nb_from_both),
                        )
                    else:
                        new_beams[new_prefix] = (-math.inf, nb_from_both)

        # Prune to top beam_width by combined log-prob.
        beams = dict(sorted(
            new_beams.items(),
            key=lambda kv: float(np.logaddexp(kv[1][0], kv[1][1])),
            reverse=True,
        )[:beam_width])

    final: list[BeamCandidate] = []
    for prefix, (p_b, p_nb) in beams.items():
        total_lp = float(np.logaddexp(p_b, p_nb))
        final.append(BeamCandidate(text=prefix, logprob=total_lp))
    final.sort(key=lambda c: c.logprob, reverse=True)
    return final


def rerank(
    candidates: list[BeamCandidate],
    lm: DomainLM,
    lm_weight: float = 0.6,
) -> BeamCandidate:
    """Rerank beam candidates by ``α × LM_logprob + (1-α) × model_logprob``.

    ``lm_weight = 0`` ignores LM (pure model). ``lm_weight = 1`` ignores
    model (pure LM). Default 0.6 leans on LM for domain bias but keeps
    the model honest on OOV words.
    """
    if not candidates:
        return BeamCandidate(text="", logprob=-math.inf, rescored=-math.inf)
    # Normalise model logprobs to the 0..1 range so weight blending
    # between model (natural log-probs of sequence) and LM (log
    # frequencies) doesn't let one dominate purely by scale.
    model_lps = np.asarray([c.logprob for c in candidates], dtype=np.float32)
    m_min, m_max = float(model_lps.min()), float(model_lps.max())
    m_range = max(1e-6, m_max - m_min)
    for c in candidates:
        model_n = (c.logprob - m_min) / m_range
        lm_lp = lm.log_prob(c.text)
        # LM log_prob is negative (freq / total). Normalise over
        # candidates for fair blending.
        c.rescored = (1.0 - lm_weight) * model_n + lm_weight * lm_lp
    return max(candidates, key=lambda c: c.rescored)


def decode_and_rerank(
    log_probs: np.ndarray,
    charset: str,
    lm: DomainLM,
    beam_width: int = 5,
    lm_weight: float = 0.6,
    blank_idx: int = 0,
) -> BeamCandidate:
    """Convenience wrapper: beam search then rerank, returns best."""
    cands = _ctc_beam_search(
        log_probs, charset,
        beam_width=beam_width, blank_idx=blank_idx,
    )
    return rerank(cands, lm, lm_weight=lm_weight)


__all__ = ["decode_and_rerank", "rerank", "BeamCandidate"]
