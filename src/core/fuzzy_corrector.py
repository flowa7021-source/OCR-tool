"""Fuzzy-based OCR correction по широкому русскому словарю.

Дополнение к ``lexicon_corrector`` (точечный dict-lookup для
16 специфичных ТН/УПД терминов): этот модуль использует
``resources/ru_lexicon.txt`` (~17 000 русских слов) как
reference-корпус для Levenshtein-based коррекции.

## Отличие от lexicon_corrector

| Что                | lexicon_corrector     | fuzzy_corrector       |
|--------------------|------------------------|------------------------|
| Размер словаря     | 23 canonical × 122 var | 16 828 форм            |
| Тип match'а        | Exact (dict-lookup)    | Fuzzy (edit-distance)  |
| Scope              | Только ТН/УПД термы    | Вся русская лексика    |
| Скорость           | O(1) per token         | O(N×L) per token       |
| Риск over-correct  | Минимальный            | Умеренный              |
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

2. **Skip короткие токены** (< 5 символов) — слишком двусмысленны,
   высокий риск ошибочной замены.

3. **Skip не-кириллические** — Latin/mixed/numeric токены
   оставляем как есть; они могут быть артикулами, номерами, брэндами.

4. **Fuzzy lookup** через ``rapidfuzz.process.extractOne`` с
   порогом 88 (короткие ≤ 7 символов) / 85 (длинные). Порог
   настроен так, чтобы 1 правка на 6-7 символов проходила
   (`edit_dist = 1`) а 2 — только для длинных слов.

5. **Если match найден** — подставляем canonical, сохраняя
   casing оригинала.

## Почему 88/85 порог

Levenshtein-distance 1 ≈ score 89-91 на 6-символьном слове,
85-88 на 10-символьном. Порог 88 ловит «1-edit» ошибки уверенно,
и пропускает «2-edit» только на длинных (где 2 правки на 10
символов = 80 % сходства).

## Скорость

17k-slovar × 5 000 токенов на документ × 1 ms/token = 5 с.
Приемлемо для batch-обработки, но слишком медленно для interactive.
Оптимизации (BKTree / symspell) — TODO если станет узким местом.

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

# Порог в rapidfuzz score (0-100). Высокий порог — главный способ
# избежать false-positives на proper nouns (именах, названиях
# городов, брэндах), которых в словаре нет:
#   «Иванов» → ближайший «Ано» (score 67) — должно быть отвергнуто;
#   «организаиия» → «организации» (score 94) — нормально принять.
# Калибровка на inputs/TN_k_UPD_*.txt: 92 ловит 1-edit опечатки
# и отклоняет любые замены короче 2/3 оригинала.
_FUZZY_THRESHOLD_SHORT = 92    # tokens 6-8 chars
_FUZZY_THRESHOLD_LONG = 90     # tokens ≥ 9 chars
_MIN_TOKEN_LEN = 6             # < 6 символов — слишком двусмысленно

# Минимальное соотношение длин |candidate| / |token|. Без этого
# fuzzy подставляет очень короткие слова на длинные токены
# («Ленина» → «На», |На|/|Ленина| = 0.33 — явно не замена,
# а collapse). 0.70 запрещает сокращение более чем на 30 %.
_MIN_LENGTH_RATIO = 0.70

# Только кириллические слова обрабатываем. Latin/mixed/числа —
# пропускаем как potential брэнды/артикулы/ИНН.
_CYRILLIC_TOKEN_RE = re.compile(r"^[А-ЯЁа-яё][А-ЯЁа-яё\-]*$")

# Сигнал «возможно это proper noun» (имя/фамилия/название
# города/организации): первая буква ЗАГЛАВНАЯ. Наш словарь
# lowercase, поэтому имена/названия туда не попадают. Если
# token начинается с заглавной И не найден в lex (даже в lower-
# casee) — не трогаем; риск исковеркать «Иванов», «Беляев»,
# «Москва», «Петергоф» слишком высок.
_PROPER_NOUN_GUARD = True


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
        match = process.extractOne(
            token_low, lex, score_cutoff=threshold,
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
        # Proper-noun guard: для Capitalized-token'ов требуем
        # еще более высокий score, чем для lowercase.
        # Proper nouns (Иванов, Москва, Петергоф) не в словаре;
        # fuzzy может найти ~90-score случайность. Повышаем порог
        # до 95 — только убедительные 1-edit ошибки (Грузоотпрапитель
        # → Грузоотправитель, score 98) проходят.
        if token[0].isupper() and not token.isupper():
            try:
                from rapidfuzz import fuzz
                if fuzz.ratio(token_low, canonical) < 95:
                    return token
            except ImportError:
                return token
        return _preserve_case(token, canonical)

    return _WORD_RE.sub(_replace, text)


def lexicon_size() -> int:
    """Количество словоформ в reference-словаре."""
    return len(_load_lexicon())
