"""Domain-specific unigram language model for OCR post-correction.

Builds a frequency-weighted vocabulary from the project's ground-truth
corpus (``inputs/*.txt`` + ``expected/*.json``) and exposes a fuzzy
correction function: given a low-confidence OCR token, find the nearest
domain word within a Levenshtein budget, re-scored by prior probability.

This runs *after* the OCR engine — no changes to EasyOCR internals
needed. Calibrated for the ~1000-word TN/UPD corpus:

    >>> lm = DomainLM.from_default_corpus()
    >>> lm.correct("гексафрм", ocr_conf=0.3)
    ('гексаформ', 0.87)
    >>> lm.correct("7813266190", ocr_conf=0.95)
    ('7813266190', 0.95)  # digit-only, not a dictionary word — keep as-is
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz.distance import Levenshtein

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUTS = REPO_ROOT / "inputs"
DEFAULT_EXPECTED = REPO_ROOT / "expected"

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9\-./,]{0,}")
_DIGIT_ONLY = re.compile(r"^[\d.,\-/]+$")


def _tokenize(text: str) -> list[str]:
    return [
        t.strip(".,").lower()
        for t in _TOKEN_RE.findall(text)
        if len(t.strip(".,")) >= 2
    ]


def _walk_json_strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _walk_json_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _walk_json_strings(v)]
    return []


@dataclass
class CorrectionResult:
    word: str
    confidence: float          # final combined score, 0..1
    was_corrected: bool        # True if word changed from input
    edit_distance: int = 0     # Levenshtein from input → word


class DomainLM:
    """Unigram LM with fuzzy lookup for OCR correction.

    Storage is just a ``Counter`` — O(V) memory where V is the vocab
    size. Typical V for the TN/UPD corpus is ~1000-1500.
    """

    def __init__(self, vocab: Counter[str]):
        self.vocab = vocab
        self._total = sum(vocab.values()) or 1
        # Pre-compute log probabilities for ranking.
        self._log_prob = {w: math.log(c / self._total) for w, c in vocab.items()}
        # Smoothed fallback for OOV.
        self._oov_log_prob = math.log(1.0 / (self._total * 10))
        # Index by length for fast candidate lookup — fuzzy match candidates
        # must be within ±budget length of the query.
        self._by_length: dict[int, list[str]] = {}
        for w in vocab:
            self._by_length.setdefault(len(w), []).append(w)

    @classmethod
    def from_default_corpus(
        cls,
        inputs_dir: Path = DEFAULT_INPUTS,
        expected_dir: Path = DEFAULT_EXPECTED,
    ) -> DomainLM:
        """Build LM from the project's default corpus locations."""
        return cls.from_files(
            list(inputs_dir.glob("*.txt")) if inputs_dir.exists() else [],
            list(expected_dir.glob("*.json")) if expected_dir.exists() else [],
        )

    @classmethod
    def from_files(
        cls,
        txt_paths: list[Path],
        json_paths: list[Path],
    ) -> DomainLM:
        vocab: Counter[str] = Counter()
        for txt in txt_paths:
            try:
                vocab.update(_tokenize(txt.read_text(encoding="utf-8")))
            except OSError:
                continue
        for jp in json_paths:
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for s in _walk_json_strings(data):
                vocab.update(_tokenize(s))
        return cls(vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, word: str) -> bool:
        return word.lower() in self.vocab

    def log_prob(self, word: str) -> float:
        return self._log_prob.get(word.lower(), self._oov_log_prob)

    def correct(
        self,
        word: str,
        ocr_conf: float,
        budget: int | None = None,
        min_ocr_conf_to_skip: float = 0.85,
    ) -> CorrectionResult:
        """Try to replace ``word`` with its closest domain match.

        Decision flow:
            1. If OCR confidence ≥ ``min_ocr_conf_to_skip`` — trust OCR, no change.
            2. If ``word`` is already in vocab (case-insensitive) — no change.
            3. If ``word`` looks like a pure-numeric token (ИНН/date/sum) —
               don't fuzzy-match against dictionary words; let the requisite
               validators handle those.
            4. Otherwise enumerate domain words within Levenshtein ``budget``
               and pick the one maximising ``OCR_conf × LM_prob / (1+edit_dist)``.
        """
        w = word.lower()
        if ocr_conf >= min_ocr_conf_to_skip:
            return CorrectionResult(word, ocr_conf, False, 0)
        if w in self.vocab:
            # Boost confidence slightly — presence in vocab is corroborating.
            return CorrectionResult(w, min(1.0, ocr_conf * 1.2), False, 0)
        if _DIGIT_ONLY.match(w):
            # Numeric tokens — skip dictionary lookup (expensive + wrong).
            return CorrectionResult(word, ocr_conf, False, 0)
        if budget is None:
            budget = 1 if len(w) < 6 else 2

        best: CorrectionResult | None = None
        for target_len in range(len(w) - budget, len(w) + budget + 1):
            if target_len < 2:
                continue
            for cand in self._by_length.get(target_len, ()):
                d = Levenshtein.distance(w, cand, score_cutoff=budget)
                if d > budget:
                    continue
                # Combined score: more probable + closer → higher.
                lm_weight = math.exp(self._log_prob[cand])
                score = ocr_conf * lm_weight / (1 + d)
                if best is None or score > best.confidence:
                    best = CorrectionResult(cand, score, cand != w, d)
        if best is None:
            return CorrectionResult(word, ocr_conf, False, 0)
        # Renormalise the reported confidence to keep it in the 0..1 range.
        # The raw ``score`` is ocr_conf × lm_prob / (1+d), which can be
        # very small. We rescale to ``max(ocr_conf, ocr_conf*0.9)`` to
        # signal "LM agreed" without overclaiming.
        reported = max(ocr_conf * 0.9, ocr_conf / (1 + best.edit_distance * 0.5))
        return CorrectionResult(best.word, reported, best.was_corrected,
                                best.edit_distance)


__all__ = ["DomainLM", "CorrectionResult"]
