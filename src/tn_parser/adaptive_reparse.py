"""Adaptive re-parse (idea #10 top-10).

При low overall_confidence пробуем альтернативные parsing стратегии
и выбираем best-of. Бесплатно для high-conf случаев (short-circuit).

**Lightweight contract** — re-parse ТОЛЬКО на parser-уровне, не
повторяя OCR. Альтернативные стратегии:

  * Primary: page-aware split (default в splitter).
  * Fallback 1: чистый anchor-based (игнорирует ``\\f``-границы,
    весь text как один document).
  * Fallback 2: no-split extraction (регексы по всему тексту).

Entry point — :func:`reparse_if_low_conf`. Принимает primary-
function + list of fallback-functions, каждая возвращает row-dict
с ключом ``overall_confidence``. Best-of выбирается max-conf.

Exception-safety: если fallback падает (import error / runtime
bug) — логируем, пропускаем, идём к следующему. Primary всегда
доступен как «last resort» — даже если он low-conf, мы возвращаем
его a не ``None``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

#: Сигнатура pass-функции. Принимает text, возвращает row-dict.
PassFn = Callable[[str], dict[str, Any]]


def reparse_if_low_conf(
    text: str,
    *,
    primary_fn: PassFn,
    fallback_fns: list[PassFn] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Попробовать primary + fallbacks, вернуть лучший по conf.

    Args:
        text: OCR-text для парсинга.
        primary_fn: Основная parse-функция.
        fallback_fns: Альтернативные strategy (если primary
            conf < threshold). Вызываются только при low-conf
            primary'е.
        threshold: Если primary.overall_confidence ≥ threshold,
            fallback'и не вызываются (short-circuit для perf).

    Returns:
        Row-dict с max overall_confidence среди всех удавшихся
        passes. Primary — baseline, который всегда доступен.
    """
    primary_result = primary_fn(text)
    primary_conf = float(primary_result.get("overall_confidence", 0.0) or 0.0)

    # Short-circuit: если primary уверенный, не тратим время на
    # альтернативы. Главный performance-gate — reparse работает
    # только на marginal cases.
    if primary_conf >= threshold:
        return primary_result
    if not fallback_fns:
        return primary_result

    best_result = primary_result
    best_conf = primary_conf

    for fallback in fallback_fns:
        try:
            alt_result = fallback(text)
        except Exception as exc:  # noqa: BLE001 — fallback isolation
            logger.warning(
                "adaptive_reparse: fallback %s raised %s; skipping",
                getattr(fallback, "__name__", "<?>"), exc,
            )
            continue
        alt_conf = float(alt_result.get("overall_confidence", 0.0) or 0.0)
        if alt_conf > best_conf:
            best_result = alt_result
            best_conf = alt_conf
            logger.info(
                "adaptive_reparse: fallback %s won with conf %.2f "
                "over primary %.2f",
                getattr(fallback, "__name__", "<?>"),
                alt_conf, primary_conf,
            )

    return best_result


__all__ = ["PassFn", "reparse_if_low_conf"]
