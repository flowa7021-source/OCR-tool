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
    word: str, *, paragraph_majority: str | None = None,
) -> str:
    """Replace Latin ↔ Cyrillic look-alikes inside a single word.

    Skips URLs, emails and Windows-style paths outright (see
    :data:`_SKIP_TOKEN_RE`). For ordinary words, dispatches on
    :func:`_classify_word_script`:

      * ``cyr`` — swap every Latin look-alike → its Cyrillic twin
      * ``lat`` — swap every Cyrillic look-alike → its Latin twin
      * ``mixed`` — when ``paragraph_majority`` is provided, the
        paragraph-wide script wins (fixes short all-look-alike
        tokens like ``Со`` in a Russian document). Without a
        paragraph hint, leaves as-is.
    """
    if _SKIP_TOKEN_RE.search(word):
        return word
    kind = _classify_word_script(word)
    if kind == "mixed" and paragraph_majority is not None:
        kind = paragraph_majority
    if kind == "cyr":
        return "".join(_LATIN_TO_CYRILLIC.get(ch, ch) for ch in word)
    if kind == "lat":
        return "".join(_CYRILLIC_TO_LATIN.get(ch, ch) for ch in word)
    return word


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


def normalize_cyrillic_latin_confusion(text: str) -> str:
    """Tokenise ``text``, normalise Latin/Cyrillic look-alikes per-word.

    Non-letter characters (digits, punctuation, whitespace) pass
    through unchanged. See :func:`_normalize_cyrillic_latin_word` for
    the per-word logic.

    Stage F enhancement: short all-look-alike tokens (which the
    per-word classifier refuses to touch because they carry no
    script-exclusive evidence) inherit the paragraph-wide script
    majority if one exists. This recovers words like ``Со`` and
    ``Оно`` in predominantly-Russian pages that Tesseract split
    with a Latin letter in the middle.
    """
    if not text:
        return text
    paragraph_majority = _paragraph_script_majority(text)
    return _WORD_CHUNK_RE.sub(
        lambda m: _normalize_cyrillic_latin_word(
            m.group(0), paragraph_majority=paragraph_majority,
        ),
        text,
    )


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

        # Business-identifier validation runs LAST — after every other
        # step has settled on its final tokenisation. A digit-run that
        # was corrupted by hyphen-merge or unicode-NFC would get
        # spuriously fixed-up; running at the tail means the number
        # we validate is the number the user will see.
        if (
            getattr(config, "validate_identifiers", False)
            and self._catalog is not None
            and not self._catalog.is_empty
        ):
            current = self._validate_identifiers(current)
            logger.debug("Postprocess: identifiers validated against catalog")

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

        catalog = self._catalog
        if catalog is None or catalog.is_empty:
            return text

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

        def _replace(match) -> str:
            token = match.group(0)
            pool = _pick_catalog(token)
            if not pool:
                return token
            # A token that's already in the catalog is canonical —
            # do not rewrite. A token that VALIDATES but isn't in
            # the catalog is probably a novel counterparty we've
            # never seen before; also leave it alone.
            if token in pool:
                return token
            already_valid = (
                (len(token) in (10, 12) and validate_inn(token))
                or (len(token) in (13, 15) and validate_ogrn(token))
            )
            if already_valid:
                return token
            fixed = catalog_assisted_fix(token, known_valid=pool)
            if fixed is None:
                return token
            logger.info(
                "validate_identifiers: %r → %r (catalog match)",
                token, fixed,
            )
            return fixed

        return self._IDENTIFIER_RE.sub(_replace, text)

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
