"""pymorphy3-based validator для OCR-correction candidates (fix #C
после top-10, доработка idea #9/#5).

Fuzzy corrector (``src.core.fuzzy_corrector``) предлагает candidate
из 17k словарных форм. Без валидации он иногда выбирает legitimate
form-variant (организация → организации) вместо настоящего
OCR-typo fix'а. Pymorphy3 знает ~3M форм русского и может
гарантировать что candidate — реальное слово, а original —
гаремон (unknown → typo).

Decision logic:

    original known + candidate known → reject (form disambig)
    original known + candidate unknown → reject (candidate — не слово)
    original unknown + candidate known → accept (typo fix!)
    original unknown + candidate unknown → reject (candidate тоже typo)

pymorphy3 — soft dependency. Если не установлен, обе функции
возвращают conservative default (``is_known=True`` / ``should_
accept=False``), т.е. fuzzy_corrector ведёт себя как раньше без
validator'а.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_morph_lock = threading.Lock()
_morph_instance = None  # lazy singleton


def _get_morph():
    """Thread-safe lazy-init MorphAnalyzer. Возвращает None если
    pymorphy3 не установлен."""
    global _morph_instance
    if _morph_instance is not None:
        return _morph_instance
    with _morph_lock:
        if _morph_instance is not None:
            return _morph_instance
        try:
            import pymorphy3
            _morph_instance = pymorphy3.MorphAnalyzer()
        except ImportError:
            logger.debug("pymorphy3 недоступен, morphology-validator =no-op")
            _morph_instance = False  # sentinel «not available»
    return _morph_instance if _morph_instance is not False else None


def is_known_russian_word(word: str) -> bool:
    """True если pymorphy3 знает ``word`` как реальное русское слово.

    ``is_known=True`` у parse означает что лемма есть в OpenCorpora
    dictionary — не наугад угаданная морфология на unknown word'е.

    Fallback при отсутствии pymorphy3 — возвращаем True (консервативно,
    чтобы не ломать существующее поведение fuzzy-corrector'а).
    """
    if not word or not word.strip():
        return False
    morph = _get_morph()
    if morph is None:
        # Soft fallback: conservative = "известно", чтобы
        # should_accept_correction вернул False и не триггерил
        # agressive replace без actual validation.
        return True
    try:
        parses = morph.parse(word.strip().lower())
    except Exception:  # noqa: BLE001
        return True
    return any(p.is_known for p in parses[:3])


# POS-классы, которые ХАРАКТЕРНЫ для content-слов, т.е. допустимые
# цели fuzzy-замены:
#
#   NOUN — существительное («экземпляр», «организация»)
#   VERB / INFN — глагол («является», «подписать»)
#   ADJF / PRTF / GRND — полное прилагательное / причастие / деепричастие
#   NPRO / NUMR — местоимение / числительное
#
# Proper nouns (фамилии, названия деревень типа «Подолино») часто
# отсутствуют в OpenCorpora — fuzzy_corrector, получив кандидата
# из 2M-lex, рискует подставить функциональное слово («Подолино»
# → «Подлинно», ADVB). Эти POS ↓ почти НЕ бывают proper-noun-подстановкой:
#
#   ADVB — наречие («подлинно», «тепло»)
#   PRED — предикатив («надо», «жаль»)
#   PREP / CONJ / PRCL / INTJ / COMP — предлог/союз/частица/междометие
#   ADJS / PRTS — КРАТКИЕ прилаг./прич. («красив», «пройден») —
#     тоже редко имеют capitalized-форму как имя собственное
#
# Если ВСЕ parses кандидата — в этом «неконтентном» множестве, а
# original Capitalized, то это скорее всего proper-noun ложное
# срабатывание. Reject.
_CONTENT_POS: frozenset[str] = frozenset({
    "NOUN", "VERB", "INFN", "ADJF", "PRTF", "GRND", "NPRO", "NUMR",
})


def _has_content_pos_reading(word: str) -> bool:
    """True если у ``word`` есть хотя бы один parse с POS из
    ``_CONTENT_POS``. Для unknown words pymorphy3 гадает — если ВСЕ
    угаданные теги в «некcontent» множестве, возвращаем False.
    """
    morph = _get_morph()
    if morph is None:
        return True  # no data → не блокируем
    try:
        parses = morph.parse(word.strip().lower())[:5]
    except Exception:  # noqa: BLE001
        return True
    for p in parses:
        pos = p.tag.POS
        if pos in _CONTENT_POS:
            return True
    return False


def should_accept_correction(original: str, candidate: str) -> bool:
    """Решить, стоит ли применять fuzzy-correction ``original →
    candidate``.

    Правила:
      * Accept только когда original unknown + candidate known.
      * Proper-noun guard: если ``original`` капитализирован (первая
        буква прописная, остальные строчные — типичный proper-noun
        layout) И все parses кандидата — функциональные/служебные
        (ADVB/PRED/PREP/…), отклоняем. Это защищает «Подолино» (geo-
        имя) от превращения в «Подлинно» (ADVB).
      * В остальных случаях — conservative reject.
      * Пустые значения → reject.
    """
    if not original or not candidate:
        return False
    if original == candidate:
        return False
    orig_known = is_known_russian_word(original)
    cand_known = is_known_russian_word(candidate)
    if not ((not orig_known) and cand_known):
        return False
    # Proper-noun guard. Только для токенов с классическим именным
    # case-pattern: первая заглавная, остальные строчные. ALL-UPPERCASE
    # и lowercase токены пропускаем (для них guard не нужен — они не
    # выглядят как proper nouns).
    looks_proper = (
        len(original) > 1
        and original[0].isupper()
        and original[1:].islower()
    )
    return not (looks_proper and not _has_content_pos_reading(candidate))


__all__ = ["is_known_russian_word", "should_accept_correction"]
