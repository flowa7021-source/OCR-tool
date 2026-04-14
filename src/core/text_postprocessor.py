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
from typing import Final

from src.core.models import PostprocessConfig, RegexRule
from src.shared.validators import ValidationError

logger = logging.getLogger(__name__)


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

    def __init__(self) -> None:
        """Pre-compile the built-in rule sets for efficiency."""
        self._russian_rules: list[tuple[Pattern[str], str]] = _compile_rules(
            DEFAULT_RUSSIAN_RULES
        )
        self._english_rules: list[tuple[Pattern[str], str]] = _compile_rules(
            DEFAULT_ENGLISH_RULES
        )

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

        if config.autocorrect_russian:
            current = self._autocorrect_russian(current)
            logger.debug("Postprocess: Russian autocorrect applied")

        if config.autocorrect_english:
            current = self._autocorrect_english(current)
            logger.debug("Postprocess: English autocorrect applied")

        if config.custom_rules:
            current = self._apply_custom_rules(current, config.custom_rules)
            logger.debug(
                "Postprocess: %d custom rules evaluated", len(config.custom_rules)
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

    @staticmethod
    def _apply_custom_rules(text: str, rules: list[RegexRule]) -> str:
        """Apply user-defined rules one by one, in order.

        Each rule is enabled/disabled independently. Invalid regex patterns
        are logged and skipped so a single bad rule cannot break the
        pipeline.
        """
        current = text
        for idx, rule in enumerate(rules):
            if not rule.enabled:
                continue
            try:
                if rule.is_regex:
                    flags = 0 if rule.case_sensitive else re.IGNORECASE
                    flags |= re.UNICODE
                    pattern = re.compile(rule.pattern, flags)
                    current = pattern.sub(rule.replacement, current)
                else:
                    if rule.case_sensitive:
                        current = current.replace(rule.pattern, rule.replacement)
                    else:
                        # Case-insensitive literal replace: use a regex with
                        # re.escape for safety.
                        pattern = re.compile(
                            re.escape(rule.pattern), re.IGNORECASE | re.UNICODE
                        )
                        current = pattern.sub(rule.replacement, current)
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
