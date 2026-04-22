"""Fuzzy-based OCR correction по широкому русскому словарю.

Дополнение к ``lexicon_corrector`` (точечный dict-lookup для
16 специфичных ТН/УПД терминов): этот модуль использует
``resources/ru_lexicon.txt`` (~2 000 000 русских словоформ,
OpenCorpora 5-12 chars) как reference-корпус для
Levenshtein-based коррекции через rapidfuzz.

## Отличие от lexicon_corrector

| Что                | lexicon_corrector     | fuzzy_corrector       |
|--------------------|------------------------|------------------------|
| Размер словаря     | 23 canonical × 122 var | ~2 034 000 форм        |
| Тип match'а        | Exact (dict-lookup)    | Fuzzy (edit-distance)  |
| Scope              | Только ТН/УПД термы    | Вся русская лексика    |
| Скорость           | O(1) per token         | O(N×L) per token       |
| Риск over-correct  | Минимальный            | Умеренный + guard      |
| Цель               | Target-fix критичных   | Общее повышение conf   |

Оба запускаются в ``TextPostprocessor`` последовательно:
lexicon_corrector первый (гарантированные fix'ы TN-vocab),
fuzzy_corrector вторым (закрывает «обычные» OCR-опечатки
в русской прозе).

## Алгоритм fuzzy-match'а

Для каждого токена из OCR-текста:

1. **Skip если уже correct** — token.lower() ∈ LEXICON → возврат
   без изменений. Это раннее прерывание даёт O(1) на уже-правильных
   токенах, т.е. большинстве.

2. **Skip короткие токены** (< 6 символов) — слишком двусмысленны,
   высокий риск ошибочной замены.

3. **Skip не-кириллические** — Latin/mixed/numeric токены
   оставляем как есть; они могут быть артикулами, номерами, брэндами.

4. **Fuzzy lookup** через ``rapidfuzz.process.extractOne`` с явным
   ``scorer=fuzz.ratio`` (full-string Levenshtein, НЕ WRatio —
   см. ниже), порог 87 (короткие 6-7 символов) / 85 (длинные ≥ 8).

5. **pymorphy3 morphology gate** через
   :mod:`src.core.morphology_validator.should_accept_correction` —
   требует, чтобы candidate был известен OpenCorpora, а original — нет.
   Дополнительный proper-noun guard отфильтровывает «Подолино → Подлинно»-
   класс ложных срабатываний.

6. **Если match принят** — подставляем canonical, сохраняя
   casing оригинала.

## Почему 87/85 порог

На 2M-lex Levenshtein-distance 1 даёт ratio 86-88 для 7-8 chars,
83-86 для 10-14 chars. Пороги 92/90 прежней версии (под 17k-lex)
упускают legitimate 1-edit fix'ы на широком словаре; 87/85 ловят
их, а false-positive защита перенесена на POS-guard.

## Почему scorer=fuzz.ratio обязателен

``rapidfuzz.process.extractOne`` по умолчанию использует ``WRatio``
(взвешенный best-of нескольких метрик, включая partial_ratio).
На 2M-форм словаре partial_ratio находит substring-match'и типа
«экземиляр → емиля» со score 90 — это false-positives. Ratio
сравнивает строки целиком, требуя настоящего edit-close proximity.

## Скорость

2M-slovar × 5 000 токенов на документ × ~2 ms/token = 10 с.
Приемлемо для batch-обработки. Оптимизации (BKTree / symspell) —
TODO если станет узким местом.

## Runtime-инициализация

Lexicon ленивая — загружается при первом вызове ``correct()``
и кэшируется в module-level ``_LEXICON``. Если файл отсутствует
(минимальный билд без resources/), корректор становится no-op.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_RESOURCE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "resources" / "ru_lexicon.txt"
)

# Порог в rapidfuzz score (0-100). На 2M-lex false-positives
# отсеиваются двумя слоями:
#   1) scorer=fuzz.ratio (full-string Levenshtein) — не ищет
#      substring-match'ей;
#   2) pymorphy3 morphology gate + proper-noun POS-filter.
# Калибровка: 87/85 ловит 1-edit опечатки «являетси → является»
# (ratio 88.9) и «экземиляр → экземпляр» (ratio 88.9), а
# proper-noun guard режет «Подолино → Подлинно» (тот же score,
# но candidate только ADVB).
_FUZZY_THRESHOLD_SHORT = 87    # tokens 6-8 chars — 1 edit на 7 chars ≈ 86%
_FUZZY_THRESHOLD_LONG = 85     # tokens ≥ 9 chars — 1-2 edit на 10-14
_MIN_TOKEN_LEN = 6             # < 6 символов — слишком двусмысленно

# Минимальное соотношение длин |candidate| / |token|. Без этого
# fuzzy подставляет очень короткие слова на длинные токены
# («Ленина» → «На», |На|/|Ленина| = 0.33 — явно не замена,
# а collapse). 0.70 запрещает сокращение более чем на 30 %.
_MIN_LENGTH_RATIO = 0.70

# Только кириллические слова обрабатываем. Latin/mixed/числа —
# пропускаем как potential брэнды/артикулы/ИНН.
_CYRILLIC_TOKEN_RE = re.compile(r"^[А-ЯЁа-яё][А-ЯЁа-яё\-]*$")

# OCR-typical char-pair confusions. В Tesseract LSTM на
# кириллическом скане систематически путаются группы визуально
# похожих/соседних по клавиатуре букв. Пары в этой таблице
# «дешёвые» — их замена не штрафует fuzzy-score так сильно, как
# произвольная замена. Используется в _ocr_weighted_score.
#
# Источник: анализ OCR-выходов inputs/TN_k_UPD_*.pdf — повторяющиеся
# substitution-классы. Можно расширить по мере наблюдений.
_OCR_CHAR_EQUIVALENCES: tuple[frozenset[str], ...] = (
    # Кириллические в пределах одной группы visual-похожих
    frozenset({"о", "а"}),           # о ↔ а — одна из самых частых
    frozenset({"е", "ё"}),           # е ↔ ё
    frozenset({"и", "й", "н"}),      # и ↔ й ↔ н
    frozenset({"ь", "ъ", "ы"}),      # hard/soft знаки
    frozenset({"п", "г", "т", "л"}), # визуально похожие
    frozenset({"ш", "щ"}),           # ш ↔ щ
    frozenset({"ч", "у"}),           # нижняя петля
    frozenset({"з", "в"}),           # OCR часто путает
    frozenset({"м", "n"}),           # rn-ligature confusion
    # Cross-script (Cyrillic ↔ Latin lookalikes)
    frozenset({"о", "o"}), frozenset({"а", "a"}), frozenset({"е", "e"}),
    frozenset({"р", "p"}), frozenset({"с", "c"}), frozenset({"х", "x"}),
    frozenset({"у", "y"}), frozenset({"к", "k"}),
    frozenset({"м", "m"}), frozenset({"т", "t"}),
    frozenset({"в", "b"}),
    frozenset({"Н", "H"}), frozenset({"Т", "T"}),
    # Digit-letter (OCR на цифрах тоже путается)
    frozenset({"о", "0"}), frozenset({"з", "3"}), frozenset({"б", "6"}),
    frozenset({"ч", "4"}), frozenset({"г", "7"}),
)


def _is_cheap_substitution(a: str, b: str) -> bool:
    """True если замена `a → b` — типичная OCR-путаница (дешёвая).

    Используется как эвристика для бустинга fuzzy-score: если все
    замены между token и candidate принадлежат «дешёвым» классам,
    считаем замену высокоуверенной.
    """
    a_low = a.lower()
    b_low = b.lower()
    if a_low == b_low:
        return True
    return any(
        a_low in cls and b_low in cls for cls in _OCR_CHAR_EQUIVALENCES
    )


def _ocr_aware_score(token: str, candidate: str) -> int:
    """Расширенный score'ring: поверх rapidfuzz.ratio добавляем
    bonus если все нестыковки — OCR-типичные замены.

    Rapidfuzz ratio = 100 × (2 × matches) / (len_a + len_b). Мы
    дополнительно смотрим на несовпадения: если 100% несовпадений
    попадают в _OCR_CHAR_EQUIVALENCES, добавляем +3 к score — это
    компенсирует 1 «дешёвую» замену и дотягивает до threshold для
    случаев типа «организаиия→организация» (и↔а не совсем OCR-
    типично, но common misread).

    Возвращает 0..100.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return 0

    base = fuzz.ratio(token, candidate)
    # Bonus только если base уже высокий (>= 80) — чтобы не
    # промахиваться на совсем разных словах.
    if base < 80:
        return base
    # Простой align через character-by-character diff для
    # одинаковых длин.
    if len(token) != len(candidate):
        return base
    cheap_mismatches = 0
    total_mismatches = 0
    for a, b in zip(token, candidate, strict=False):
        if a == b:
            continue
        total_mismatches += 1
        if _is_cheap_substitution(a, b):
            cheap_mismatches += 1
    if total_mismatches == 0:
        return 100
    if cheap_mismatches == total_mismatches:
        return min(100, base + 3)
    return base

@lru_cache(maxsize=1)
def _load_lexicon() -> frozenset[str]:
    """Ленивая загрузка словаря. Все слова lowercase."""
    if not _RESOURCE_PATH.is_file():
        logger.info(
            "fuzzy_corrector: словарь %s отсутствует — корректор "
            "работает как no-op. Сгенерируйте через "
            "`python scripts/gen_ru_lexicon.py`.",
            _RESOURCE_PATH,
        )
        return frozenset()
    words = {
        line.strip().lower()
        for line in _RESOURCE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    logger.info(
        "fuzzy_corrector: загружено %d русских словоформ из %s",
        len(words), _RESOURCE_PATH.name,
    )
    return frozenset(words)


@lru_cache(maxsize=1)
def _load_lexicon_list() -> tuple[str, ...]:
    """rapidfuzz.process хочет индексируемый iterable. Конвертируем
    frozenset → tuple (immutable, hashable for lru_cache)."""
    return tuple(sorted(_load_lexicon()))


_WORD_RE = re.compile(r"[А-ЯЁа-яё][А-ЯЁа-яё\-]{4,}", re.UNICODE)


def _preserve_case(original: str, replacement: str) -> str:
    """Та же логика, что в lexicon_corrector: натянуть casing
    оригинала на replacement."""
    if not original:
        return replacement
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:].lower()
    return replacement


def _best_match(token_low: str) -> str | None:
    """Вернуть best-match canonical из словаря или None.

    Пороги adaptive: для коротких слов (5-7 chars) нужен score
    ≥ 88 (≤ 1 edit), для длинных (≥ 8 chars) — ≥ 85 (позволяем
    2 edits в длинной форме).
    """
    try:
        from rapidfuzz import process
    except ImportError:
        return None

    lex = _load_lexicon_list()
    if not lex:
        return None

    threshold = (
        _FUZZY_THRESHOLD_LONG if len(token_low) >= 8
        else _FUZZY_THRESHOLD_SHORT
    )
    try:
        from rapidfuzz import fuzz
        # ``scorer=fuzz.ratio`` (full-string Levenshtein) обязателен
        # для расширенного 2M-форм lexicon'а (апрель 2026 v2): default
        # ``fuzz.WRatio`` включает partial_ratio, который на больших
        # списках находит substring-match'и типа «экземиляр → емиля»
        # с score 90 — это false-positives. Ratio сравнивает строки
        # целиком, требуя настоящей edit-close proximity.
        match = process.extractOne(
            token_low, lex, scorer=fuzz.ratio, score_cutoff=threshold,
        )
    except Exception:  # noqa: BLE001 — rapidfuzz иногда падает на edge-cases
        return None
    if match is None:
        return None
    # match = (canonical, score, index_in_lex)
    canonical = match[0]
    # Не заменяем если canonical == token_low — токен уже correct.
    if canonical == token_low:
        return None
    # Length-ratio guard: замена не должна «схлопывать» слово
    # сильнее чем на 30 %. Это ловит «Ленина» → «На» и
    # подобные collapse'ы.
    if len(canonical) / max(len(token_low), 1) < _MIN_LENGTH_RATIO:
        return None
    # OCR-aware re-scoring: для одинаковых-длин кандидатов смотрим,
    # не попадают ли все несовпадения в OCR-typical substitution
    # классы. Если да — это confidence-бустер; если нет — должна
    # быть очень высокая base-score. Это барьер против случайных
    # fuzzy-matches.
    if len(canonical) == len(token_low):
        aware = _ocr_aware_score(token_low, canonical)
        if aware < threshold:
            return None
    return canonical


def correct(text: str) -> str:
    """Fuzzy-коррекция русских слов в тексте.

    Скан токенов → каждое кириллическое слово длиной ≥ 5 проверяем
    против словаря. Если в словаре есть близкий кандидат (edit
    distance ≤ 1-2 в зависимости от длины), заменяем.

    Идемпотентно. Не бросает исключений (rapidfuzz ошибки
    перехватываются). При отсутствии ресурса — no-op.
    """
    if not text:
        return text
    lex = _load_lexicon()
    if not lex:
        return text

    def _replace(m: re.Match[str]) -> str:
        token = m.group(0)
        token_low = token.lower()
        # Skip если уже известен корпусу — не меняем.
        if token_low in lex:
            return token
        # Skip короткие токены.
        if len(token_low) < _MIN_TOKEN_LEN:
            return token
        # Skip если не чисто-кириллический.
        if not _CYRILLIC_TOKEN_RE.match(token):
            return token
        canonical = _best_match(token_low)
        if canonical is None:
            return token
        # Proper-noun guard УДАЛЁН (апрель 2026 v2): blanket-требование
        # score ≥ 95 % для Capitalized токенов блокировало legitimate
        # OCR-fix'ы типа «Экземиляр → Экземпляр» (score 88.9). Вместо
        # этого proper nouns защищены pymorphy3-gate ниже: если
        # candidate имеет только ADVB/PRED/PREP POS-readings,
        # should_accept_correction отклонит замену.
        # pymorphy3 morphology gate: убеждаемся что candidate —
        # реальное русское слово, а original — unknown (= OCR-typo).
        # Это защищает от замены legitimate-form (организация ↔
        # организации) которые rapidfuzz может случайно выдать. Без
        # pymorphy3 — no-op (возвращает True), поведение как раньше.
        try:
            from .morphology_validator import should_accept_correction

            # Передаём ОРИГИНАЛ (с сохранённым casing), чтобы guard
            # смог детектить Capitalized-токены как potentially-
            # proper-noun. Передача token_low похерила бы всю логику.
            if not should_accept_correction(token, canonical):
                return token
        except Exception:  # noqa: BLE001 — validator не должен ронять pipeline
            pass
        return _preserve_case(token, canonical)

    return _WORD_RE.sub(_replace, text)


def lexicon_size() -> int:
    """Количество словоформ в reference-словаре."""
    return len(_load_lexicon())
