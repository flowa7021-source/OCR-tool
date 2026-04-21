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


def should_accept_correction(original: str, candidate: str) -> bool:
    """Решить, стоит ли применять fuzzy-correction ``original →
    candidate``.

    Правила (см. module-docstring):
      * Accept только когда original unknown + candidate known.
      * В остальных случаях — conservative reject.
      * Пустые значения → reject.
    """
    if not original or not candidate:
        return False
    if original == candidate:
        return False
    orig_known = is_known_russian_word(original)
    cand_known = is_known_russian_word(candidate)
    return (not orig_known) and cand_known


__all__ = ["is_known_russian_word", "should_accept_correction"]
