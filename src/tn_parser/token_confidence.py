"""Token-level confidence propagation (idea #5 top-10).

Парсер выставляет discrete confidence (0.0 / 0.5 / 0.7 / 0.9 / 1.0)
по правилам extraction-layer'а. Если ИНН прошёл checksum — conf
= 1.0, независимо от того была ли OCR уверена в цифрах. На
real-data случаи:

  * OCR прочитал «7701234567» с per-char conf 40-60 %, checksum
    совпал случайно (не такой уж redкий случай при OCR-шуме на
    10-digit строках). Парсер даёт 1.0; пользователь доверяет;
    данные могут быть неверными.

Решение — прокидывать per-word OCR-conf в валидаторы:

  * :class:`TokenConfMap` хранит (start_char_idx, end_char_idx,
    conf) ranges для всего text'а.
  * :meth:`TokenConfMap.for_substring` возвращает среднюю conf
    символов substring'а.
  * Валидаторы (``find_inn_with_conf``) возвращают (value,
    ocr_conf) и при integration'е в core оба сигнала объединяются.

Formulaic:

    final_conf = min(1.0, structural_conf * (0.5 + 0.5 * ocr_conf))

* structural_conf — checksum/regex-based (1.0 если passed).
* ocr_conf — 0..1, OCR certainty на токенах.
* min — clamp к [0, 1].
* формула 0.5 + 0.5×ocr_conf: OCR=1 → никаких штрафов;
  OCR=0 → confidence уполовинивается; OCR=0.5 → ~0.75 (balanced).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenConfMap:
    """Mapping (start_char, end_char) → OCR-confidence 0..100.

    ``ranges`` — список non-overlapping intervals в координатах text'а.
    Составляется pipeline'ом из ``pytesseract.image_to_data`` где
    известно: text-offset каждого токена + его conf. Если map пуст,
    ``for_substring`` возвращает ``None`` (сигнал «ocr-conf
    недоступен, не применяй формулу, оставь structural-conf как есть»).
    """

    ranges: list[tuple[int, int, float]] = field(default_factory=list)

    @classmethod
    def from_ranges(
        cls, ranges: list[tuple[int, int, float]]
    ) -> TokenConfMap:
        return cls(ranges=list(ranges))

    def for_substring(self, text: str, value: str) -> float | None:
        """Средняя conf (0..1) токенов покрывающих substring ``value``.

        Args:
            text: Исходный text в котором был extract'ен value.
            value: Substring для которого считаем conf.

        Returns:
            * ``None`` — map пуст (нет info).
            * ``0.0`` — value не найден в text или нет overlapping
              токенов.
            * ``0..1`` — средняя conf по токенам, нормализованная.
        """
        if not self.ranges:
            return None
        if not value or not text:
            return 0.0
        idx = text.find(value)
        if idx < 0:
            return 0.0
        vstart, vend = idx, idx + len(value)
        # Для каждого range: если он перекрывается с [vstart, vend),
        # берём его conf в среднее.
        confs: list[float] = []
        for rs, re, c in self.ranges:
            if re <= vstart or rs >= vend:
                continue  # disjoint
            confs.append(float(c))
        if not confs:
            return 0.0
        avg = sum(confs) / len(confs)
        # Нормализуем 0-100 → 0-1.
        return max(0.0, min(1.0, avg / 100.0))


def combine_confidences(
    structural: float,
    ocr_conf: float | None,
) -> float:
    """Объединить structural и OCR-conf в итоговую confidence.

    * ``structural`` — 0..1 от validator'а (checksum/regex/catalog).
    * ``ocr_conf`` — 0..1 от TokenConfMap, или None (не влияет).

    Formula: final = structural * (0.5 + 0.5 * ocr_conf).
    OCR=1 → no change; OCR=0 → half structural; OCR=None → structural.
    """
    if ocr_conf is None:
        return max(0.0, min(1.0, structural))
    penalty = 0.5 + 0.5 * ocr_conf
    return max(0.0, min(1.0, structural * penalty))


__all__ = ["TokenConfMap", "combine_confidences"]
