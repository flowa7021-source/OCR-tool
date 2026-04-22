"""Text post-processing for OCR output.

Applies a configurable chain of transformations to the raw text produced by
Tesseract: Unicode normalization, whitespace cleanup, language-specific
autocorrection, custom regex rules, etc.

The implementation is deliberately conservative — only high-confidence
substitutions are applied by default, and every rule is context-aware to
avoid breaking legitimate content (for example, the digit ``0`` is only
converted to the Cyrillic letter ``О`` when it sits inside a Cyrillic word).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from re import Pattern
from typing import Any, Final

from src.core.garbage_filter import GarbageStrictness, filter_garbage_lines
from src.core.models import PostprocessConfig, RegexRule
from src.shared.validators import ValidationError

logger = logging.getLogger(__name__)


# Per-rule watchdog for catastrophic-backtracking protection. A typical
# well-formed regex against a page of text completes in microseconds;
# anything over one second is almost certainly a ReDoS attempt.
REGEX_TIMEOUT_SEC: float = 1.0


class RegexTimeoutError(RuntimeError):
    """Raised when a pattern's substitution exceeds :data:`REGEX_TIMEOUT_SEC`."""


def _substitute_with_timeout(
    pattern: re.Pattern[str],
    replacement: str,
    text: str,
    timeout: float = REGEX_TIMEOUT_SEC,
) -> str:
    """Run ``pattern.sub`` with a wall-clock watchdog.

    The work is offloaded to a short-lived thread so the main pipeline
    thread can abandon a runaway regex rather than hanging forever.
    This is not a perfect ReDoS defence — the regex engine itself
    still runs to completion somewhere — but the daemon thread dies
    with the process and the caller regains control in bounded time.

    Args:
        pattern: Pre-compiled regex.
        replacement: Substitution string (supports backrefs).
        text: Input text.
        timeout: Seconds to wait before giving up.

    Returns:
        The substituted text.

    Raises:
        RegexTimeoutError: If the substitution did not finish in time.
    """
    import threading

    result: list[str] = [text]
    captured_exc: list[BaseException | None] = [None]

    def _worker() -> None:
        try:
            result[0] = pattern.sub(replacement, text)
        except BaseException as exc:  # noqa: BLE001 — bubble up to caller
            captured_exc[0] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise RegexTimeoutError(
            f"Регулярное выражение {pattern.pattern!r} выполняется дольше "
            f"{timeout}s — похоже на ReDoS, правило пропущено."
        )
    if captured_exc[0] is not None:
        raise captured_exc[0]
    return result[0]


# ---------------------------------------------------------------------------
# Default rule sets
# ---------------------------------------------------------------------------

# Each rule is ``(pattern, replacement, description)``. Patterns are compiled
# with :data:`re.UNICODE` so that ``\w`` matches Cyrillic letters correctly.
#
# Context is enforced using look-around assertions. This is important: we
# don't want to transform ``3000`` (where ``3`` is a genuine digit) into
# ``З000`` (``З`` is Cyrillic "Ze").

#: Cyrillic letter (either case), excluding common OCR look-alikes we might
#: want to swap in.
_CYR = r"[А-Яа-яЁё]"

#: Any ASCII digit.
_DIGIT = r"[0-9]"

DEFAULT_RUSSIAN_RULES: Final[list[tuple[str, str, str]]] = [
    # Digit-looking-like-letter → letter, but only INSIDE a Cyrillic word.
    (rf"(?<={_CYR})0(?={_CYR})", "о", "0 → о внутри кириллического слова"),
    (rf"(?<={_CYR})0(?={_CYR})", "О", "0 → О внутри кириллического слова (верх.)"),
    (rf"(?<={_CYR})3(?={_CYR})", "з", "3 → з внутри кириллического слова"),
    (rf"(?<={_CYR})4(?={_CYR})", "ч", "4 → ч внутри кириллического слова"),
    (rf"(?<={_CYR})6(?={_CYR})", "б", "6 → б внутри кириллического слова"),
    (rf"(?<={_CYR})8(?={_CYR})", "в", "8 → в внутри кириллического слова"),
    (rf"(?<={_CYR})1(?={_CYR})", "л", "1 → л внутри кириллического слова"),
    # Letter-looking-like-digit → digit, only INSIDE a run of digits.
    (rf"(?<={_DIGIT})[ОOо](?={_DIGIT})", "0", "О → 0 внутри числа"),
    (rf"(?<={_DIGIT})[Зз](?={_DIGIT})", "3", "З → 3 внутри числа"),
    (rf"(?<={_DIGIT})[Чч](?={_DIGIT})", "4", "Ч → 4 внутри числа"),
    (rf"(?<={_DIGIT})[бB](?={_DIGIT})", "6", "б → 6 внутри числа"),
    (rf"(?<={_DIGIT})[Вв](?={_DIGIT})", "8", "В → 8 внутри числа"),
    (rf"(?<={_DIGIT})[IlI|](?={_DIGIT})", "1", "I → 1 внутри числа"),
    # Latin ↔ Cyrillic look-alikes inside Cyrillic words.
    (rf"(?<={_CYR})H(?={_CYR})", "Н", "латинская H → Н"),
    (rf"(?<={_CYR})B(?={_CYR})", "В", "латинская B → В"),
    (rf"(?<={_CYR})C(?={_CYR})", "С", "латинская C → С"),
    (rf"(?<={_CYR})P(?={_CYR})", "Р", "латинская P → Р"),
    (rf"(?<={_CYR})T(?={_CYR})", "Т", "латинская T → Т"),
    (rf"(?<={_CYR})M(?={_CYR})", "М", "латинская M → М"),
    (rf"(?<={_CYR})A(?={_CYR})", "А", "латинская A → А"),
    (rf"(?<={_CYR})E(?={_CYR})", "Е", "латинская E → Е"),
    (rf"(?<={_CYR})K(?={_CYR})", "К", "латинская K → К"),
    (rf"(?<={_CYR})X(?={_CYR})", "Х", "латинская X → Х"),
    (rf"(?<={_CYR})O(?={_CYR})", "О", "латинская O → О"),
    # Lower-case Latin ↔ Cyrillic look-alikes.
    (rf"(?<={_CYR})a(?={_CYR})", "а", "латинская a → а"),
    (rf"(?<={_CYR})c(?={_CYR})", "с", "латинская c → с"),
    (rf"(?<={_CYR})e(?={_CYR})", "е", "латинская e → е"),
    (rf"(?<={_CYR})o(?={_CYR})", "о", "латинская o → о"),
    (rf"(?<={_CYR})p(?={_CYR})", "р", "латинская p → р"),
    (rf"(?<={_CYR})x(?={_CYR})", "х", "латинская x → х"),
    (rf"(?<={_CYR})y(?={_CYR})", "у", "латинская y → у"),
    # Soft sign lowercase ``b`` → ``ь`` in Cyrillic context.
    (rf"(?<={_CYR})b(?={_CYR})", "ь", "латинская b → ь внутри слова"),
    (rf"(?<={_CYR})b(?=\b)", "ь", "латинская b → ь на конце слова"),
]

DEFAULT_ENGLISH_RULES: Final[list[tuple[str, str, str]]] = [
    # Digit-looking-like-letter → letter, only INSIDE an alphabetic word.
    (r"(?<=[A-Za-z])0(?=[A-Za-z])", "o", "0 → o внутри английского слова"),
    (r"(?<=[A-Za-z])1(?=[A-Za-z])", "l", "1 → l внутри английского слова"),
    (r"(?<=[A-Za-z])5(?=[A-Za-z])", "s", "5 → s внутри английского слова"),
    # Letter-looking-like-digit → digit, only INSIDE a number.
    (r"(?<=[0-9])[Oo](?=[0-9])", "0", "O → 0 внутри числа"),
    (r"(?<=[0-9])[Il|](?=[0-9])", "1", "I → 1 внутри числа"),
    (r"(?<=[0-9])[Ss](?=[0-9])", "5", "S → 5 внутри числа"),
    # Common bi-gram misreads — only when flanked by lower-case letters.
    (r"(?<=[a-z])rn(?=[a-z])", "m", "rn → m между строчными буквами"),
    (r"(?<=[a-z])cl(?=[a-z])", "d", "cl → d между строчными буквами"),
    (r"(?<=[a-z])vv(?=[a-z])", "w", "vv → w между строчными буквами"),
    (r"\bvv(?=[a-z])", "w", "vv → w в начале слова"),
]


# ---------------------------------------------------------------------------
# Word-level Latin↔Cyrillic look-alike normalisation.
#
# Tesseract routinely confuses visually-identical letters across the two
# scripts — notably at word boundaries where the inline
# ``DEFAULT_RUSSIAN_RULES`` look-behind / look-ahead rules above can't
# fire. ``Ивановo`` (Latin ``o`` at the end of "Иванов") and ``oткрыть``
# (Latin ``o`` at the start of "открыть") both slip through the
# regex-only pass because the Latin look-alike sits next to whitespace
# on one side.
#
# This word-level pass tokenises the text, classifies each word as
# "definitely Cyrillic" / "definitely Latin" / "ambiguous" by looking
# for unambiguous script-exclusive characters, and normalises the
# look-alikes inside unambiguous words. Ambiguous words (pure
# look-alikes like ``ABC`` or words with script-exclusive chars from
# BOTH scripts) are left untouched to avoid wrecking legitimate mixed
# content (technical terms, product codes, URLs).
# ---------------------------------------------------------------------------


#: Uppercase/lowercase Latin letters that have a visually identical
#: Cyrillic counterpart. Keeps the mapping small and explicit — adding
#: marginal look-alikes here (e.g. ``Q``→``Ԛ``) would over-correct.
#: The mirror pair ``I``↔``І`` is intentionally omitted — Russian
#: documents don't use ``І`` (pre-1918 orthography), so a Tesseract
#: confusion there is vanishingly rare and the asymmetric digit/I
#: collision is handled by the separate regex rule set.
_LATIN_TO_CYRILLIC: Final[dict[str, str]] = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н",
    "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т",
    "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р",
    "x": "х", "y": "у",
}

_CYRILLIC_TO_LATIN: Final[dict[str, str]] = {
    v: k for k, v in _LATIN_TO_CYRILLIC.items()
}

#: Characters that can ONLY be Cyrillic — no Latin look-alike.
#: Seeing any of these marks a word as unambiguously Cyrillic.
_CYRILLIC_EXCLUSIVE: Final[frozenset[str]] = frozenset(
    "БГДЖЗИЙЛПФЦЧШЩЪЫЬЭЮЯЁ"
    "бгджзийлпфцчшщъыьэюяё"
)

#: Characters that can ONLY be Latin — no Cyrillic look-alike.
#: Seeing any of these marks a word as unambiguously Latin.
_LATIN_EXCLUSIVE: Final[frozenset[str]] = frozenset(
    "DFGIJLNQRSUVWZ"
    "bdfghijklmnqrstuvwz"
)

#: Look-alike chars that carry NO script-exclusive evidence on their
#: own. A word made entirely of these is a "pure look-alike" token —
#: the word-level classifier would return ``mixed`` with no bias.
#: This frozenset isn't used by the classifier directly but documents
#: which characters the paragraph-majority / numeric-context tie-
#: breakers are expected to handle.
_LOOKALIKE_CHARS: Final[frozenset[str]] = frozenset(
    "".join(_LATIN_TO_CYRILLIC)
    + "".join(_LATIN_TO_CYRILLIC.values())
)

#: Tokens that must NOT be touched even if they look Cyrillic-majority.
#: URLs and email-shaped tokens carry Latin on purpose; a path like
#: ``/usr/local`` shouldn't have its letters swapped even if the
#: surrounding text is Russian.
_SKIP_TOKEN_RE: Final[Pattern[str]] = re.compile(
    r"(?:https?://|www\.|[\w.-]+@|[A-Za-z]:[\\/])",
)

#: Word boundary splitter. Matches runs of letter-like characters
#: plus hyphens/apostrophes (common inside words); everything else —
#: whitespace, punctuation, digits — goes through verbatim.
_WORD_CHUNK_RE: Final[Pattern[str]] = re.compile(
    r"([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'’\-]*)",
)


def _classify_word_script(word: str) -> str:
    """Return ``"cyr"``, ``"lat"`` or ``"mixed"`` for a single word.

    A word is **unambiguously Cyrillic** when it contains at least one
    character from :data:`_CYRILLIC_EXCLUSIVE` and zero characters from
    :data:`_LATIN_EXCLUSIVE`. Symmetrically for Latin. Everything else
    — pure look-alikes, or content with evidence from both scripts —
    is ``"mixed"`` and left alone.
    """
    has_cyr = any(ch in _CYRILLIC_EXCLUSIVE for ch in word)
    has_lat = any(ch in _LATIN_EXCLUSIVE for ch in word)
    if has_cyr and not has_lat:
        return "cyr"
    if has_lat and not has_cyr:
        return "lat"
    return "mixed"


def _normalize_cyrillic_latin_word(
    word: str,
    *,
    paragraph_majority: str | None = None,
    prefer_latin: bool = False,
) -> str:
    """Replace Latin ↔ Cyrillic look-alikes inside a single word.

    Skips URLs, emails and Windows-style paths outright (see
    :data:`_SKIP_TOKEN_RE`). For ordinary words, dispatches on
    :func:`_classify_word_script`:

      * ``cyr`` — swap every Latin look-alike → its Cyrillic twin
      * ``lat`` — swap every Cyrillic look-alike → its Latin twin
      * ``mixed`` — three resolution strategies, tried in order:
          1. ``prefer_latin`` — set by the caller when the word sits
             in a numeric context (adjacent to digits via
             ``-``/``_``/``/`` or at position-adjacent offsets).
             Product codes, invoice numbers and account IDs are
             overwhelmingly Latin even in Russian documents; a
             Russian-dominant paragraph majority would otherwise
             wrongly flip e.g. ``INV-12345`` to Cyrillic.
          2. ``paragraph_majority`` — the document's script
             skew (fixes short all-look-alike tokens like ``Со``
             in a Russian page that Tesseract split with a Latin
             letter in the middle).
          3. Leave untouched.
    """
    if _SKIP_TOKEN_RE.search(word):
        return word
    kind = _classify_word_script(word)
    if kind == "mixed":
        if prefer_latin:
            kind = "lat"
        elif paragraph_majority is not None:
            kind = paragraph_majority
    if kind == "cyr":
        return "".join(_LATIN_TO_CYRILLIC.get(ch, ch) for ch in word)
    if kind == "lat":
        return "".join(_CYRILLIC_TO_LATIN.get(ch, ch) for ch in word)
    return word


# Aggressive Latin→Cyrillic map для context-override (fix #B):
# в Cyrillic-majority параграфе включает также пары визуально-
# близкие но не идентичные (d↔д, g↔г, r↔р, f↔ф, n↔н, m↔м, l↔л,
# b↔б, i↔и, j↔й, u↔и/у, s↔з, h↔н). Применяется только при
# дополнительных gate'ах (см. _aggressive_cyrillify).
_AGGRESSIVE_LAT_TO_CYR: Final[dict[str, str]] = {
    **_LATIN_TO_CYRILLIC,  # наследуем strict look-alike pairs (A/E/O/...)
    # Uppercase-дополнения:
    "D": "Д", "G": "Г", "F": "Ф", "N": "Н", "L": "Л", "I": "И",
    "J": "Й", "U": "У", "R": "Р", "S": "С", "B": "В", "V": "В",
    # Lowercase-дополнения (в look-alikes только a/c/e/o/p/x/y):
    "b": "в", "d": "д", "f": "ф", "g": "г", "h": "н",
    "i": "и", "j": "й", "k": "к", "l": "л", "m": "м", "n": "н",
    "r": "р", "s": "с", "t": "т", "u": "у", "v": "в",
}

# Buквы с которыми aggressive-conversion НЕ имеет смысла: Q/W/Z
# не имеют разумного Cyrillic counterpart'а. Слово с ними — не
# OCR-typo Cyrillic'а, а legitimately Latin (английское или бренд).
_AGGRESSIVE_UNCONVERTIBLE: Final[frozenset[str]] = frozenset("QqWwZz")


def _aggressive_cyrillify(word: str) -> str | None:
    """Попытаться полностью конвертировать Latin-классифицированное
    слово в Cyrillic через :data:`_AGGRESSIVE_LAT_TO_CYR`.

    Returns:
        Cyrillic-версия word'а, либо ``None`` если word не стоит
        конвертировать:
          * содержит буквы из :data:`_AGGRESSIVE_UNCONVERTIBLE`
            (Q/W/Z — нет Cyrillic counterpart'а);
          * ALL-UPPERCASE И длина ≥ 3 (brand: TENSAR / VOLVO) —
            такие намеренно Latin;
          * содержит хотя бы одну букву без mapping'а.
    """
    if not word:
        return None
    # Brand heuristic: ALL-UPPERCASE длиной ≥ 3 буквы → likely
    # brand или abbreviation. Оставляем.
    letters = [ch for ch in word if ch.isalpha()]
    if (
        len(letters) >= 3
        and all(ch.isupper() for ch in letters)
    ):
        return None
    # Unconvertible chars — явный signal «это не OCR-typo».
    if any(ch in _AGGRESSIVE_UNCONVERTIBLE for ch in word):
        return None
    out_chars: list[str] = []
    for ch in word:
        if ch.isalpha():
            mapped = _AGGRESSIVE_LAT_TO_CYR.get(ch)
            if mapped is None:
                # Буква без aggressive mapping (в _AGGRESSIVE_LAT_TO_
                # CYR нет) — полностью не сконвертируем, лучше
                # не трогать слово.
                return None
            out_chars.append(mapped)
        else:
            out_chars.append(ch)
    return "".join(out_chars)


def _paragraph_script_majority(text: str) -> str | None:
    """Return ``"cyr"``, ``"lat"`` or ``None`` for the whole document.

    Counts unambiguous script-exclusive characters across the full
    string. Used to break ties for short all-look-alike tokens —
    e.g. ``Со`` in a Russian sentence should inherit the paragraph's
    Cyrillic majority rather than stay mixed. Requires at least 3×
    as many of one script's exclusive chars as the other, AND at
    least 3 total exclusive chars, so a single stray Latin letter
    in a Russian document doesn't flip the majority.
    """
    cyr = sum(1 for ch in text if ch in _CYRILLIC_EXCLUSIVE)
    lat = sum(1 for ch in text if ch in _LATIN_EXCLUSIVE)
    if cyr + lat < 3:
        return None
    if cyr >= 3 * lat and cyr > 0:
        return "cyr"
    if lat >= 3 * cyr and lat > 0:
        return "lat"
    return None


#: Numeric-context detector. Matches a digit optionally preceded by
#: ``-``, ``_``, ``/`` or ``.`` — separators common in product codes,
#: invoice numbers and account IDs. When a word chunk is immediately
#: followed OR preceded by such a pattern, the chunk is treated as
#: likely-Latin (``prefer_latin=True``) so e.g. ``INV-12345`` stays
#: Latin even inside a Russian-majority document.
_NUMERIC_AFTER_RE: Final[Pattern[str]] = re.compile(r"^[-_/.]?\d")
_NUMERIC_BEFORE_RE: Final[Pattern[str]] = re.compile(r"\d[-_/.]?$")


# Экзотическая латинская диакритика — ВСЁ, что не основная ASCII-
# латиница / кириллица / цифры / пунктуация. Tesseract LSTM при
# низкой уверенности на кириллице fallback-ит на «похожие по
# форме» глифы из других training-наборов: французские é/à/è/ç,
# немецкие ü/ä/ö, польские ł/ż, испанский ñ, норвежский ø и т.п.
# В русско-/англоязычных документах они НИКОГДА не нужны.
# Убираем перед autocorrect_russian чтобы regex-правила видели
# чистый текст.
_EXOTIC_DIACRITIC_MAP: Final[dict[str, str]] = {
    # Французский / общий латинский диакритический
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a",
    "À": "A", "Á": "A", "Â": "A", "Ã": "A", "Ä": "A", "Å": "A",
    "è": "e", "é": "e", "ê": "e", "ë": "e",
    "È": "E", "É": "E", "Ê": "E", "Ë": "E",
    "ì": "i", "í": "i", "î": "i", "ï": "i",
    "Ì": "I", "Í": "I", "Î": "I", "Ï": "I",
    "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o", "ø": "o",
    "Ò": "O", "Ó": "O", "Ô": "O", "Õ": "O", "Ö": "O", "Ø": "O",
    "ù": "u", "ú": "u", "û": "u", "ü": "u",
    "Ù": "U", "Ú": "U", "Û": "U", "Ü": "U",
    "ý": "y", "ÿ": "y",
    "Ý": "Y", "Ÿ": "Y",
    "ñ": "n", "Ñ": "N",
    "ç": "c", "Ç": "C",
    "œ": "oe", "Œ": "OE",
    "æ": "ae", "Æ": "AE",
    "ß": "ss",
    # Польские специфичные
    "ł": "l", "Ł": "L",
    "ż": "z", "Ż": "Z",
    "ź": "z", "Ź": "Z",
    "ą": "a", "Ą": "A",
    "ę": "e", "Ę": "E",
    "ć": "c", "Ć": "C",
    "ń": "n", "Ń": "N",
    "ś": "s", "Ś": "S",
    # Другие европейские
    "č": "c", "Č": "C",
    "š": "s", "Š": "S",
    "ž": "z", "Ž": "Z",
    "ř": "r", "Ř": "R",
    "ů": "u", "Ů": "U",
    "ě": "e", "Ě": "E",
    "ā": "a", "ē": "e", "ī": "i", "ō": "o", "ū": "u",
    "ı": "i", "İ": "I",
}


def _strip_exotic_diacritics(text: str) -> str:
    """Заменяет экзотическую латинскую диакритику на ближайший ASCII-
    эквивалент. Tesseract LSTM на плохих кириллических сканах часто
    даёт é/ü/ł/ñ/ç — для русско-/англоязычных документов это OCR-
    артефакт, засоряющий словарь. Возвращаем обратно к чистой
    ASCII-латинице (à → a, é → e, ł → l, ñ → n, …).

    Не трогает кириллицу (ё/й останутся), цифры и пунктуацию.
    """
    if not text:
        return text
    return "".join(_EXOTIC_DIACRITIC_MAP.get(ch, ch) for ch in text)


def normalize_cyrillic_latin_confusion(text: str) -> str:
    """Tokenise ``text``, normalise Latin/Cyrillic look-alikes per-word.

    Non-letter characters (digits, punctuation, whitespace) pass
    through unchanged. See :func:`_normalize_cyrillic_latin_word` for
    the per-word logic.

    Two resolution strategies for ``mixed``-script tokens:
      * Numeric-context heuristic — a word sitting next to digits
        (``INV-12345``, ``7USD``, ``INN 7701234567``) is tilted
        Latin-ward regardless of paragraph majority. Product codes,
        invoice numbers and account IDs are overwhelmingly Latin in
        Russian documents too, and the old paragraph-majority
        strategy would wrongly flip them to Cyrillic.
      * Paragraph majority — short all-look-alike tokens inherit
        the document's script skew (fixes ``Со``, ``Оно`` etc. on
        predominantly-Russian pages that Tesseract split with a
        Latin letter in the middle).

    The numeric-context check runs first so product codes win over
    prose majority; falls back to paragraph majority for everything
    else.
    """
    if not text:
        return text
    paragraph_majority = _paragraph_script_majority(text)
    out: list[str] = []
    last = 0
    for m in _WORD_CHUNK_RE.finditer(text):
        out.append(text[last:m.start()])
        word = m.group(0)
        # Look up to five chars in each direction for a digit /
        # separator-digit pattern. Five is empirically enough for the
        # "space + digit", "hyphen + number" and "dot + digit" shapes
        # seen in real invoices without being expensive.
        after = text[m.end():m.end() + 5]
        before = text[max(0, m.start() - 5):m.start()]
        numeric_context = bool(
            _NUMERIC_AFTER_RE.match(after)
            or _NUMERIC_BEFORE_RE.search(before)
        )
        normalized = _normalize_cyrillic_latin_word(
            word,
            paragraph_majority=paragraph_majority,
            prefer_latin=numeric_context,
        )
        # Aggressive context override (idea top-10 дополнение):
        # в Cyrillic-параграфе слово классифицированное как pure-
        # ``lat`` — почти наверняка OCR-typo Cyrillic'а. Пробуем
        # aggressive conversion через расширенный map. Gate'ы
        # внутри _aggressive_cyrillify: brand ALL-UPPER / чары Q-W-Z
        # / неконвертируемые буквы → None.
        if (
            paragraph_majority == "cyr"
            and not numeric_context
            and _classify_word_script(normalized) == "lat"
        ):
            aggressive = _aggressive_cyrillify(normalized)
            if aggressive is not None:
                normalized = aggressive
        out.append(normalized)
        last = m.end()
    out.append(text[last:])
    return "".join(out)


# ---------------------------------------------------------------------------
# TextPostprocessor
# ---------------------------------------------------------------------------


class TextPostprocessor:
    """Applies a chain of text-cleanup transformations.

    The transformations are gated by flags on :class:`PostprocessConfig`,
    allowing the user to enable only what they need per-profile.

    Thread safety:
        Compiled regex objects are immutable, making instances safe to share
        across threads. Custom rules are recompiled on every call to
        :meth:`process` so dynamic reconfiguration is allowed.
    """

    def __init__(
        self,
        *,
        catalog: Any = None,
    ) -> None:
        """Pre-compile the built-in rule sets for efficiency.

        Args:
            catalog: Optional :class:`src.core.doc_catalog.DocCatalog`
                used by the identifier-validation step when
                :attr:`PostprocessConfig.validate_identifiers` is on.
                Stored by reference — pass the same instance to every
                worker to share its frozen sets.

                The parameter is typed as ``Any`` (rather than
                ``DocCatalog | None``) to avoid importing the doc
                module at class-definition time: every ``pipeline.py``
                unit test that patches out OCR shouldn't need to
                import the catalog layer too.
        """
        self._russian_rules: list[tuple[Pattern[str], str]] = _compile_rules(
            DEFAULT_RUSSIAN_RULES
        )
        self._english_rules: list[tuple[Pattern[str], str]] = _compile_rules(
            DEFAULT_ENGLISH_RULES
        )
        # User regex rules were being re-compiled on every ``process()``
        # call. For a 500-page job with 10 rules that's 5 000 wasted
        # `re.compile` calls. Cache by `(pattern, flags)` so the cost
        # is paid exactly once per unique rule per Postprocessor instance.
        self._user_regex_cache: dict[tuple[str, int], Pattern[str]] = {}
        self._catalog = catalog

    # ------------------------------------------------------------------
    # Internal: compiled regex cache
    # ------------------------------------------------------------------

    def _get_or_compile(self, pattern: str, flags: int) -> Pattern[str]:
        """Return a cached compiled regex, compiling on first use."""
        key = (pattern, flags)
        compiled = self._user_regex_cache.get(key)
        if compiled is None:
            compiled = re.compile(pattern, flags)
            self._user_regex_cache[key] = compiled
        return compiled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str, config: PostprocessConfig) -> str:
        """Apply the full post-processing pipeline to ``text``.

        Order of operations:
            1. Unicode normalization (NFC).
            2. Merge hyphenated line breaks.
            3. Whitespace normalization.
            4. Remove artifact-only lines.
            5. Russian autocorrect.
            6. English autocorrect.
            7. Custom user regex rules.

        Args:
            text: Raw OCR output.
            config: Post-processing configuration.

        Returns:
            Cleaned-up text.

        Raises:
            ValidationError: If inputs are of the wrong type.
        """
        if not isinstance(text, str):
            raise ValidationError(
                f"text должен быть str, получено: {type(text).__name__}"
            )
        if not isinstance(config, PostprocessConfig):
            raise ValidationError(
                f"config должен быть PostprocessConfig, получено: {type(config).__name__}"
            )

        if not text:
            return text

        current = text

        if config.normalize_unicode:
            current = self._normalize_unicode(current)
            logger.debug("Postprocess: unicode normalized")

        if config.merge_hyphenated:
            current = self._merge_hyphenated(current)
            logger.debug("Postprocess: hyphenated words merged")

        if config.normalize_whitespace:
            current = self._normalize_whitespace(current)
            logger.debug("Postprocess: whitespace normalized")

        if config.remove_artifacts:
            current = self._remove_artifacts(current)
            logger.debug("Postprocess: artifact lines removed")

        # Line-level garbage filter runs AFTER artifact removal (which
        # strips obvious noise chars) and BEFORE the autocorrect passes
        # (which operate on individual characters). Dropping garbage
        # lines first means the regex rules don't waste cycles on
        # lines we were going to throw away anyway.
        strictness_raw = getattr(
            config, "garbage_filter_strictness", "lenient",
        )
        try:
            strictness = GarbageStrictness(strictness_raw)
        except ValueError:
            logger.warning(
                "Unknown garbage_filter_strictness=%r — falling back "
                "to 'lenient'", strictness_raw,
            )
            strictness = GarbageStrictness.LENIENT
        if strictness is not GarbageStrictness.DISABLED:
            current = filter_garbage_lines(current, strictness)
            logger.debug(
                "Postprocess: garbage filter applied (%s)", strictness.value,
            )

        # Word-level Latin↔Cyrillic look-alike fix must run BEFORE the
        # regex autocorrects — those rules rely on the text already
        # being classified consistently per word, and a lingering
        # Latin ``o`` at the end of a Russian word would confuse the
        # Cyrillic-context look-arounds downstream.
        if config.fix_cyrillic_latin_confusion:
            current = normalize_cyrillic_latin_confusion(current)
            logger.debug("Postprocess: Cyrillic/Latin look-alikes normalised")

        if config.autocorrect_russian:
            # Strip exotic Latin diacritics (é, à, ñ, ç, ý, ł, ż…) —
            # их выдаёт Tesseract LSTM когда distractor-fallback'ит
            # на кириллицу и пытается впечатать «что-то похожее» из
            # training-словаря. В наших ТН/УПД корпусах эти символы
            # никогда не нужны: нам нужны ТОЛЬКО русский + английский
            # (см. профиль universal_accurate languages=["rus", "eng"]).
            # Удаляем их ДО autocorrect_russian, чтобы regex-правила
            # видели чистый кириллический/ASCII-латинский поток.
            current = _strip_exotic_diacritics(current)

            current = self._autocorrect_russian(current)
            logger.debug("Postprocess: Russian autocorrect applied")

            # Точечная коррекция критичных ТН/УПД терминов через
            # dict-lookup (23 canonical × 122 OCR-варианта): сначала
            # быстро и гарантированно чиним Грузоотправитель /
            # Грузополучатель / Перевозчик / идентифицировать /
            # реквизиты / TENSAR /... Lexicon содержит только
            # exact-match варианты из реальных OCR-выходов —
            # false-positive риск минимальный, поэтому on-by-default
            # вместе с autocorrect_russian.
            from src.core.lexicon_corrector import correct as _lex_correct
            current = _lex_correct(current)
            logger.debug(
                "Postprocess: lexicon-corrector applied (TN/УПД vocab)"
            )

            # Широкий fuzzy-корректор через reference-словарь
            # ~2 000 000 русских словоформ (resources/ru_lexicon.txt,
            # OpenCorpora 5-12 chars). Opt-in через
            # ``postprocess.fuzzy_correction_ru`` — применяется к ВСЕМ
            # русским токенам ≥ 6 chars и может менять legitimate
            # word-forms (организация↔организации, оформил↔оформи).
            # Pymorphy3 proper-noun guard отсекает геоимена/фамилии
            # от превращения в функциональные слова. Полезно на
            # heavy-mangled OCR-выходах типичных ТН-сканов, но на
            # clean synthetic corpus'е может снижать CER/WER из-за
            # form-mismatch с ground-truth. Включайте явно в профиле
            # только для scan-corpus'а.
            if getattr(config, "fuzzy_correction_ru", False):
                from src.core.fuzzy_corrector import (
                    correct as _fuzzy_correct,
                )
                current = _fuzzy_correct(current)
                logger.debug(
                    "Postprocess: fuzzy-corrector applied (2M ref dict)"
                )

        if config.autocorrect_english:
            current = self._autocorrect_english(current)
            logger.debug("Postprocess: English autocorrect applied")

        if config.custom_rules:
            current = self._apply_custom_rules(current, config.custom_rules)
            logger.debug(
                "Postprocess: %d custom rules evaluated", len(config.custom_rules)
            )

        # Business-identifier validation runs LAST — after every other
        # step has settled on its final tokenisation. A digit-run that
        # was corrupted by hyphen-merge or unicode-NFC would get
        # spuriously fixed-up; running at the tail means the number
        # we validate is the number the user will see.
        if (
            getattr(config, "validate_identifiers", False)
        ):
            # Runs catalog-based fixup when a catalog is present, plus
            # an OCR-confusion correction pass (via
            # src.shared.requisite_validators) that works without a
            # catalog — so the flag is meaningful even in profiles that
            # don't load a doc catalog.
            current = self._validate_identifiers(current)
            logger.debug("Postprocess: identifiers validated "
                         "(catalog=%s)",
                         "yes" if self._catalog and not self._catalog.is_empty
                         else "no")

        # Entity validation — dates / amounts / phones. Independent
        # of ``validate_identifiers`` so profiles can pick and choose
        # (an invoice profile might want date + amount normalisation
        # but no catalog-backed identifier rewrite).
        if getattr(config, "validate_entities", False):
            current = self._validate_entities(current)
            logger.debug("Postprocess: dates / amounts / phones normalised")

            # Дополнительный слой entity-нормализации (декабрь 2026):
            # канонизирует даты / телефоны / адресные префиксы через
            # regex-based normalizers. _validate_entities уже чинит
            # opt-пат «12.O1.2O23» → «12.01.2023», но не перестраивает
            # формат даты (29/08/2022 → 29.08.2022) и не распаковывает
            # слипшиеся адреса (125212,г.Москва → 125212, г. Москва).
            # Делает Excel-вывод byte-stable между прогонами одного
            # документа на разных сканах/DPI.
            from src.core.entity_normalizers import normalize_all
            current = normalize_all(current)
            logger.debug(
                "Postprocess: entity normalizers applied (dates/phones/addrs)"
            )

        return current

    # ------------------------------------------------------------------
    # Individual transformations
    # ------------------------------------------------------------------

    def _autocorrect_russian(self, text: str) -> str:
        """Apply :data:`DEFAULT_RUSSIAN_RULES` to ``text``."""
        return _apply_compiled_rules(text, self._russian_rules)

    def _autocorrect_english(self, text: str) -> str:
        """Apply :data:`DEFAULT_ENGLISH_RULES` to ``text``."""
        return _apply_compiled_rules(text, self._english_rules)

    @staticmethod
    def _merge_hyphenated(text: str) -> str:
        """Merge word-break hyphens of the form ``abc-\\nxyz`` → ``abcxyz``.

        Preserves hyphens that are not at end-of-line (i.e. ordinary dashes
        inside compound words).
        """
        # Hyphen followed by newline(s) sitting between two word characters.
        return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text, flags=re.UNICODE)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace.

        - Convert CRLF/CR to LF.
        - Strip trailing whitespace from each line.
        - Collapse runs of spaces/tabs into a single space.
        - Collapse runs of 3+ blank lines into exactly 2.
        """
        # Normalize line endings.
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        processed: list[str] = []
        for line in lines:
            collapsed = re.sub(r"[ \t]+", " ", line)
            processed.append(collapsed.rstrip())
        joined = "\n".join(processed)
        # Collapse three or more consecutive newlines into two.
        return re.sub(r"\n{3,}", "\n\n", joined)

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Apply NFC normalization."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _remove_artifacts(text: str) -> str:
        """Drop lines that contain no alphanumeric characters.

        Artefact lines are pure OCR noise such as ``|``, ``~~~``, or rows
        of dashes. Lines that contain at least one letter or digit are
        always preserved, even if they also carry punctuation.
        """
        if not text:
            return text

        lines = text.split("\n")
        kept: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Preserve blank lines — they carry paragraph structure.
                kept.append(line)
                continue
            # Keep the line if it has at least one letter or digit.
            if re.search(r"[^\W_]", stripped, flags=re.UNICODE):
                kept.append(line)
            else:
                logger.debug("Postprocess: отброшена артефакт-строка: %r", stripped)
        return "\n".join(kept)

    # Digit-only tokens 10-15 chars long — the union of ИНН (10 / 12)
    # and ОГРН (13 / 15). ``\b`` anchors keep us from matching
    # sub-sequences of longer numbers (e.g. part of a 20-digit SWIFT).
    _IDENTIFIER_RE: Final[Pattern[str]] = re.compile(r"\b\d{10,15}\b")
    # Matches 10-15 character runs of digits plus common OCR lookalikes
    # (O, o, О, о, I, l, З, з, В, в, Ч, ч, |). Used by the OCR-confusion
    # correction pass to catch tokens the strict digit-only regex misses.
    _OCR_DIGITISH_RE: Final[Pattern[str]] = re.compile(
        r"\b[0-9OoОоIlI|ЗзВвЧч!]{10,15}\b"
    )

    def _validate_identifiers(self, text: str) -> str:
        """Replace catalog-1-edit-matched digit tokens with canonical.

        Scans for digit-runs of length 10/12 (ИНН) and 13/15 (ОГРН).
        For each token:

          1. If the token is already a valid identifier AND appears in
             the catalog → keep as-is (canonical).
          2. If invalid OR valid-but-absent-from-catalog: search the
             catalog for a unique 1-edit neighbour. Replace when
             exactly one match is found.
          3. Otherwise leave the token alone — ambiguous and bare-
             checksum corrections are unsafe to pick without the
             catalog narrowing the space.

        The replacements are logged at INFO so the user can audit
        each fixup.
        """
        from src.core.doc_validators import (
            catalog_assisted_fix,
            validate_inn,
            validate_ogrn,
        )
        from src.shared.requisite_validators import (
            correct_inn as _ocr_fix_inn,
        )
        from src.shared.requisite_validators import (
            correct_ogrn as _ocr_fix_ogrn,
        )

        catalog = self._catalog
        has_catalog = catalog is not None and not catalog.is_empty

        def _pick_catalog(token: str) -> frozenset[str]:
            # Route ИНН-shaped tokens to the INN set, ОГРН-shaped to
            # the OGRN set. ``12``-digit tokens could be ИНН-for-
            # individual; ``13``/``15`` are ОГРН — no overlap.
            if len(token) in (10, 12):
                return catalog.inns
            if len(token) in (13, 15):
                return catalog.ogrns
            # 11 / 14 are only valid as one-edit-deletion inputs;
            # route them to both catalogs so the fix can go either
            # way (deletion of a spurious digit typically restores
            # a 10-digit ИНН or 13-digit ОГРН).
            if len(token) == 11:
                return catalog.inns
            if len(token) == 14:
                return catalog.ogrns
            return frozenset()

        def _ocr_confusion_fallback(token: str) -> str | None:
            """Checksum-based OCR-confusion correction (O↔0, l↔1, З↔3 …).
            Returns the canonical requisite or ``None`` if no valid
            rescue exists within the substitution budget.
            """
            if len(token) in (10, 12):
                return _ocr_fix_inn(token)
            if len(token) in (13, 15):
                return _ocr_fix_ogrn(token)
            return None

        def _replace(match) -> str:
            token = match.group(0)
            # 1. Catalog-based fix (preferred when catalog is available).
            if has_catalog:
                pool = _pick_catalog(token)
                if pool:
                    if token in pool:
                        return token
                    already_valid = (
                        (len(token) in (10, 12) and validate_inn(token))
                        or (len(token) in (13, 15) and validate_ogrn(token))
                    )
                    if already_valid:
                        return token
                    fixed = catalog_assisted_fix(token, known_valid=pool)
                    if fixed is not None:
                        logger.info(
                            "validate_identifiers: %r → %r (catalog match)",
                            token, fixed,
                        )
                        return fixed
            # 2. OCR-confusion fallback — works even without a catalog.
            #    Only replaces when the corrected version passes checksum.
            ocr_fixed = _ocr_confusion_fallback(token)
            if ocr_fixed is not None and ocr_fixed != token:
                logger.info(
                    "validate_identifiers: %r → %r (OCR-confusion fix)",
                    token, ocr_fixed,
                )
                return ocr_fixed
            return token

        # Pass 1: digit-only identifiers — catalog lookup + checksum fix.
        text = self._IDENTIFIER_RE.sub(_replace, text)

        # Pass 2: OCR-garbled identifiers — tokens of length 10/12/13/15
        # made of digits + common OCR lookalikes (O, o, О, о, I, l, З, в …).
        # These don't match the digit-only ``_IDENTIFIER_RE`` but are
        # very likely corrupted requisites. Replace only when the
        # checksum-corrected version differs from the raw token AND
        # validates.
        def _ocr_replace(match) -> str:
            token = match.group(0)
            if token.isdigit():
                return token  # already handled by Pass 1
            fixed = _ocr_confusion_fallback(token)
            if fixed is not None and fixed != token:
                logger.info(
                    "validate_identifiers: %r → %r (OCR-confusion fix, pass 2)",
                    token, fixed,
                )
                return fixed
            return token

        return self._OCR_DIGITISH_RE.sub(_ocr_replace, text)

    # Regex used by ``_validate_entities`` to FIND entity-shaped
    # tokens. Deliberately lenient on separators (dots, commas,
    # spaces, hyphens) so OCR errors on punctuation don't prevent
    # detection. Each match is passed to the entity-specific
    # validator, which decides whether to keep or replace it.
    _DATE_CANDIDATE_RE: Pattern[str] = re.compile(
        r"\b[0-9OoОо|lIZzBSsGT]{1,2}"
        r"[.,\s\-/]"
        r"[0-9OoОо|lIZzBSsGT]{1,2}"
        r"[.,\s\-/]"
        r"[0-9OoОо|lIZzBSsGT]{2,4}\b",
    )
    #: Amounts — integer + optional fractional with currency hint.
    #: ``\b`` on both ends keeps us off digit runs that are part of
    #: longer identifiers (ИНН / phone / account numbers).
    _AMOUNT_CANDIDATE_RE: Pattern[str] = re.compile(
        r"\b\d{1,3}(?:[\s\xa0]\d{3})+(?:[,.]\d{1,2})?"
        r"(?:\s*(?:руб\.?|₽))?\b"
        r"|\b\d+[,.]\d{2}\s*(?:руб\.?|₽)\b",
    )
    _PHONE_CANDIDATE_RE: Pattern[str] = re.compile(
        r"(?:\+7|\b8)[\s\-().]*"
        r"[0-9OoОоlIZzBSsG]{3,4}"
        r"[\s\-().]*"
        r"[0-9OoОоlIZzBSsG][\s\-().0-9OoОоlIZzBSsG]{5,15}",
    )

    def _validate_entities(self, text: str) -> str:
        """Normalise dates / amounts / phones via
        :mod:`src.core.entity_validators`.

        Each category runs as a regex-scoped substitution: we
        FIND entity-shaped candidates (loose patterns that accept
        OCR letter-digit confusion), pass each match to the
        corresponding ``try_fix_*`` validator, and swap in the
        canonical form when the validator returns one. The
        validator returns ``None`` for shapes it can't confidently
        fix, which leaves the OCR text unchanged — preferring a
        missed correction over a false-positive rewrite.

        The three categories run in order: dates → amounts →
        phones. Dates are cheapest to disambiguate (calendar
        constraint is tight) and may consume digit sequences the
        amount / phone matchers would otherwise chase; running
        them first reduces cross-category false positives.
        """
        from src.core.entity_validators import (
            try_fix_amount,
            try_fix_date,
            try_fix_phone,
        )

        def _date_sub(match) -> str:
            fixed = try_fix_date(match.group(0))
            if fixed is None or fixed == match.group(0):
                return match.group(0)
            logger.debug(
                "validate_entities: date %r → %r", match.group(0), fixed,
            )
            return fixed

        def _amount_sub(match) -> str:
            fixed = try_fix_amount(match.group(0))
            if fixed is None or fixed == match.group(0):
                return match.group(0)
            logger.debug(
                "validate_entities: amount %r → %r",
                match.group(0), fixed,
            )
            return fixed

        def _phone_sub(match) -> str:
            fixed = try_fix_phone(match.group(0))
            if fixed is None or fixed == match.group(0):
                return match.group(0)
            logger.debug(
                "validate_entities: phone %r → %r",
                match.group(0), fixed,
            )
            return fixed

        text = self._DATE_CANDIDATE_RE.sub(_date_sub, text)
        text = self._AMOUNT_CANDIDATE_RE.sub(_amount_sub, text)
        text = self._PHONE_CANDIDATE_RE.sub(_phone_sub, text)
        return text

    def _apply_custom_rules(self, text: str, rules: list[RegexRule]) -> str:
        """Apply user-defined rules one by one, in order.

        Each rule is enabled/disabled independently. Invalid regex
        patterns are logged and skipped so a single bad rule cannot
        break the pipeline.

        Every rule runs under a :data:`REGEX_TIMEOUT_SEC` watchdog —
        catastrophic-backtracking patterns (imported from a malicious
        profile, e.g. ``(a+)+b`` applied to ``"a" * 50``) are aborted
        instead of hanging the worker.
        """
        current = text
        for idx, rule in enumerate(rules):
            if not rule.enabled:
                continue
            try:
                if rule.is_regex:
                    flags = 0 if rule.case_sensitive else re.IGNORECASE
                    flags |= re.UNICODE
                    pattern = self._get_or_compile(rule.pattern, flags)
                    current = _substitute_with_timeout(
                        pattern, rule.replacement, current
                    )
                else:
                    if rule.case_sensitive:
                        current = current.replace(rule.pattern, rule.replacement)
                    else:
                        # Case-insensitive literal replace: use a regex with
                        # re.escape for safety.
                        pattern = self._get_or_compile(
                            re.escape(rule.pattern), re.IGNORECASE | re.UNICODE
                        )
                        current = _substitute_with_timeout(
                            pattern, rule.replacement, current
                        )
            except RegexTimeoutError as exc:
                logger.warning(
                    "Правило #%d (%r): %s",
                    idx,
                    rule.pattern,
                    exc,
                )
                continue
            except re.error as exc:
                logger.warning(
                    "Пользовательское правило #%d (%r) некорректно: %s — пропускаем",
                    idx,
                    rule.pattern,
                    exc,
                )
                continue
            except Exception as exc:  # noqa: BLE001 — keep pipeline alive
                logger.warning(
                    "Ошибка при применении правила #%d (%r): %s",
                    idx,
                    rule.pattern,
                    exc,
                )
                continue
        return current


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_rules(
    rules: list[tuple[str, str, str]],
) -> list[tuple[Pattern[str], str]]:
    """Compile a list of ``(pattern, replacement, description)`` tuples."""
    compiled: list[tuple[Pattern[str], str]] = []
    for pattern, replacement, description in rules:
        try:
            compiled.append((re.compile(pattern, re.UNICODE), replacement))
        except re.error as exc:
            logger.error(
                "Не удалось скомпилировать встроенное правило %r (%s): %s",
                pattern,
                description,
                exc,
            )
    return compiled


def _apply_compiled_rules(
    text: str, rules: list[tuple[Pattern[str], str]]
) -> str:
    """Apply a list of pre-compiled ``(Pattern, replacement)`` pairs."""
    current = text
    for pattern, replacement in rules:
        try:
            current = pattern.sub(replacement, current)
        except re.error as exc:
            logger.warning(
                "Ошибка при применении встроенного правила %r: %s",
                pattern.pattern,
                exc,
            )
    return current


# ---------------------------------------------------------------------------
# Universal "max cleanup" preset
# ---------------------------------------------------------------------------


def build_universal_postprocess_config() -> PostprocessConfig:
    """Return the preset used by the ``universal_accurate`` builtin profile.

    Turns on every safe cleanup step:

    * ``normalize_unicode`` — NFC form (canonical composition).
    * ``merge_hyphenated`` — glue end-of-line hyphens back together.
    * ``normalize_whitespace`` — collapse runs of spaces, strip trailing,
      cap blank-line runs at 2.
    * ``remove_artifacts`` — drop lines that are pure punctuation /
      OCR noise.
    * ``autocorrect_russian`` — built-in Russian rule set.
    * ``autocorrect_english`` — built-in English rule set.

    No user custom rules by default — the universal preset is meant to
    be a safe baseline on top of which users can layer their own.
    """
    return PostprocessConfig(
        autocorrect_russian=True,
        autocorrect_english=True,
        merge_hyphenated=True,
        normalize_whitespace=True,
        normalize_unicode=True,
        remove_artifacts=True,
        custom_rules=[],
    )


# Module-level singletons — same rationale as the preprocessor's.
UNIVERSAL_POSTPROCESS_CONFIG: PostprocessConfig = build_universal_postprocess_config()
_UNIVERSAL_POSTPROCESSOR: TextPostprocessor | None = None


def postprocess_universal(text: str) -> str:
    """One-shot: apply every cleanup step to ``text``.

    Equivalent to::

        post = TextPostprocessor()
        post.process(text, UNIVERSAL_POSTPROCESS_CONFIG)

    but with the :class:`TextPostprocessor` cached at module scope.
    """
    global _UNIVERSAL_POSTPROCESSOR
    if _UNIVERSAL_POSTPROCESSOR is None:
        _UNIVERSAL_POSTPROCESSOR = TextPostprocessor()
    return _UNIVERSAL_POSTPROCESSOR.process(text, UNIVERSAL_POSTPROCESS_CONFIG)
