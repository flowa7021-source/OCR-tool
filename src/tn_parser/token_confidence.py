"""Token-level confidence propagation.

The parser assigns a discrete structural confidence (0.0 / 0.5 / 0.7
/ 0.9 / 1.0) per extraction rule. Structural confidence alone ignores
how certain the OCR engine was about the characters — a checksum
collision on a noisy digit run looks identical to a clean read.

This module plumbs per-word OCR-confidence into the validators:

  * :class:`TokenConfMap` holds ``(start_char_idx, end_char_idx, conf)``
    ranges for the whole page text.
  * :meth:`TokenConfMap.for_substring` returns the mean confidence of
    tokens that overlap the substring.

Formula used by callers:

    final_conf = structural_conf * (0.5 + 0.5 * ocr_conf)

* OCR=1 → no change; OCR=0 → half structural; OCR=None → structural.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenConfMap:
    """Mapping ``(start_char, end_char) → OCR-confidence 0..100``.

    ``ranges`` — non-overlapping intervals in text coordinates. Empty
    map → :meth:`for_substring` returns ``None`` ("no info, don't
    apply the formula").
    """

    ranges: list[tuple[int, int, float]] = field(default_factory=list)

    @classmethod
    def from_ranges(
        cls, ranges: list[tuple[int, int, float]]
    ) -> TokenConfMap:
        return cls(ranges=list(ranges))

    def for_substring(self, text: str, value: str) -> float | None:
        """Mean conf (0..1) of tokens overlapping ``value`` inside ``text``.

        Returns:
            * ``None`` — map is empty.
            * ``0.0`` — ``value`` not found or no overlapping tokens.
            * ``0..1`` — mean conf normalised from 0-100.
        """
        if not self.ranges:
            return None
        if not value or not text:
            return 0.0
        idx = text.find(value)
        if idx < 0:
            return 0.0
        vstart, vend = idx, idx + len(value)
        confs: list[float] = []
        for rs, re, c in self.ranges:
            if re <= vstart or rs >= vend:
                continue
            confs.append(float(c))
        if not confs:
            return 0.0
        avg = sum(confs) / len(confs)
        return max(0.0, min(1.0, avg / 100.0))


def combine_confidences(
    structural: float,
    ocr_conf: float | None,
) -> float:
    """Combine structural and OCR-conf into a final confidence.

    * ``structural`` — 0..1 from a validator (checksum/regex/catalog).
    * ``ocr_conf`` — 0..1 from :class:`TokenConfMap`, or ``None``.
    """
    if ocr_conf is None:
        return max(0.0, min(1.0, structural))
    penalty = 0.5 + 0.5 * ocr_conf
    return max(0.0, min(1.0, structural * penalty))


__all__ = ["TokenConfMap", "combine_confidences"]
