"""Organization-name fuzzy normalizer (idea #9 top-10).

OCR типично мангнет 1-2 буквы в названии организации (особенно
когда печать смазана или шрифт нестандартный):

    Ground truth: «ООО "ГЕКСАФОРМ СПБ"»
    OCR typo:     «ООО "ГЕКСАФОРМ СГБ"»   (П → Г)
    OCR typo:     «ООО "Векoм"»           (Б → В, Cyrillic o → Latin o)

Catalog-based exact match таких не ловит. ``lexicon_corrector``
специализирован на ТН-терминах (Грузоотправитель / Перевозчик),
не на названиях. Этот модуль делает fuzzy-match через rapidfuzz
и нормализует raw-name к каноническому из catalog'а.

Usage::

    from src.tn_parser.org_normalizer import normalize_org_name

    raw = "ООО ГЕКСАФОРМ СГБ"  # OCR typo: П→Г
    candidates = [r.name for r in catalog.orgs]
    normalized = normalize_org_name(raw, candidates, threshold=85)
    # → "ООО ГЕКСАФОРМ СПБ"

Интеграция в ``_catalog_crossvalidate``: после exact-match lookup
пробуем fuzzy-match. При успешном match'е заменяем shipper/
consignee на canonical. Notes'ом в row.note добавляем
«fuzzy_normalized_from: {raw}» чтобы пользователь видел когда
нормализатор что-то поменял.
"""

from __future__ import annotations

from collections.abc import Iterable


def _try_rapidfuzz():
    """Optional import. Возвращает ``token_sort_ratio`` функцию или None."""
    try:
        from rapidfuzz.fuzz import token_sort_ratio
        return token_sort_ratio
    except ImportError:
        return None


def normalize_org_name(
    raw: str | None,
    candidates: Iterable[str] | None,
    *,
    threshold: int = 85,
) -> str:
    """Нормализовать OCR-typo название организации к catalog-
    каноническому.

    Args:
        raw: Сырое имя из OCR (e.g. ``"ООО ГЕКСАФОРМ СГБ"``). Пустое
            → возвращаем пустую строку.
        candidates: Список canonical-имён из catalog'а. Пустой /
            ``None`` → возвращаем raw без изменений (нечего матчить).
        threshold: rapidfuzz ``token_sort_ratio`` минимум (0-100) для
            принятия fuzzy-match'а. 85 — балансит false-positives:
            «ООО А» vs «ООО Б» на short names даст ≥ 85, но более
            отличающиеся пары отсекает.

    Returns:
        Canonical-имя из candidates если найден match ≥ threshold,
        иначе raw без изменений.
    """
    if not raw:
        return ""
    if not candidates:
        return raw

    # Уберём "?"-entries и пустые строки из candidates.
    clean_cands = [c for c in candidates if c and c.strip() and c != "?"]
    if not clean_cands:
        return raw

    # Exact match — короткий путь.
    if raw in clean_cands:
        return raw

    ratio_fn = _try_rapidfuzz()
    if ratio_fn is None:
        # Fallback без rapidfuzz: примитивное token-overlap. Хуже
        # качеством, но не ломаем pipeline при отсутствии deps.
        raw_tokens = set(raw.lower().split())
        best = raw
        best_score = 0.0
        for cand in clean_cands:
            cand_tokens = set(cand.lower().split())
            if not cand_tokens:
                continue
            overlap = len(raw_tokens & cand_tokens)
            score = overlap / max(len(raw_tokens), len(cand_tokens))
            if score > best_score:
                best_score = score
                best = cand
        return best if best_score * 100 >= threshold else raw

    # С rapidfuzz — предпочитаем token_sort_ratio (порядок слов
    # не важен, толерантен к typo через token-level LCS).
    best = raw
    best_ratio = 0.0
    for cand in clean_cands:
        r = ratio_fn(raw, cand)
        if r > best_ratio:
            best_ratio = r
            best = cand
    return best if best_ratio >= threshold else raw


__all__ = ["normalize_org_name"]
