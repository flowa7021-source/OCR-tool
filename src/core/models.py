"""Dataclass models used across all application layers.

All models are serializable to/from JSON (via dataclasses.asdict / from_dict
helpers) so they can be persisted as profiles or transferred between processes.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.shared.constants import (
    DEFAULT_ADAPTIVE_BLOCK_SIZE,
    DEFAULT_ADAPTIVE_C,
    DEFAULT_CLAHE_CLIP,
    DEFAULT_CLAHE_TILE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DPI,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_MEDIAN_KSIZE,
    DEFAULT_MORPH_KSIZE,
    DEFAULT_NLM_H,
    DEFAULT_SAUVOLA_K,
    DEFAULT_SAUVOLA_WINDOW,
)
from src.shared.types import (
    BinarizationMethod,
    DenoiseMethod,
    JobStatus,
    OCREngineKind,
)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


@dataclass
class DeskewConfig:
    """Skew correction parameters."""

    enabled: bool = True
    auto_detect: bool = True
    manual_angle: float = 0.0  # used when auto_detect=False
    max_angle: float = 45.0


@dataclass
class AutoRotateConfig:
    """Coarse page-orientation detection (90 / 180 / 270°).

    Runs before :class:`DeskewConfig`. Where deskew fixes the ±5°
    tilt of a scanned page, this fixes the "scanner ate the paper
    sideways" class of errors in 90° increments — a common failure
    mode on landscape documents fed through a portrait-oriented
    sheet feeder, or phone-camera snaps that come out rotated.

    Attributes:
        enabled: Whether to call Tesseract's OSD and rotate. Cheap
            (~50 ms per page) and safe — the orientation module
            only rotates when OSD reports confidence above
            :attr:`min_confidence`; below that it leaves the image
            untouched.
        min_confidence: Tesseract OSD ``Orientation confidence``
            floor. Default ``1.0`` matches
            :data:`src.core.orientation_detector.MIN_ORIENTATION_CONFIDENCE`
            and empirically excludes the random-guess regime for
            logo-only / stamp-only / blank pages.
    """

    enabled: bool = True
    min_confidence: float = 1.0


@dataclass
class DewarpConfig:
    """Page dewarping (cubic sheet model via page-dewarp)."""

    enabled: bool = False
    # page-dewarp internal parameters; only tweak when needed
    text_min_width: int = 15
    text_min_height: int = 2
    text_min_aspect: float = 1.5
    focal_length: float = 1.2


@dataclass
class BinarizationConfig:
    """Binarization parameters."""

    method: BinarizationMethod = BinarizationMethod.OTSU
    adaptive_block_size: int = DEFAULT_ADAPTIVE_BLOCK_SIZE  # must be odd
    adaptive_c: int = DEFAULT_ADAPTIVE_C
    sauvola_window: int = DEFAULT_SAUVOLA_WINDOW
    sauvola_k: float = DEFAULT_SAUVOLA_K


@dataclass
class DenoiseStep:
    """Single denoising step; allows chaining multiple methods."""

    method: DenoiseMethod
    enabled: bool = True
    # generic params; only relevant ones are used per method
    ksize: int = DEFAULT_MEDIAN_KSIZE
    sigma: float = DEFAULT_GAUSSIAN_SIGMA
    h: int = DEFAULT_NLM_H
    morph_ksize: int = DEFAULT_MORPH_KSIZE


@dataclass
class DenoiseConfig:
    """Ordered pipeline of denoising steps."""

    enabled: bool = False
    steps: list[DenoiseStep] = field(default_factory=list)


@dataclass
class ContrastConfig:
    """Contrast/brightness enhancement."""

    clahe_enabled: bool = False
    clahe_clip: float = DEFAULT_CLAHE_CLIP
    clahe_tile: int = DEFAULT_CLAHE_TILE
    manual_enabled: bool = False
    alpha: float = 1.0  # contrast multiplier
    beta: int = 0  # brightness offset


@dataclass
class BackgroundConfig:
    """Shadow/background removal."""

    enabled: bool = False
    blur_kernel: int = 55  # large odd kernel for background estimation


@dataclass
class BorderRemovalConfig:
    """Table-border / ruler-line removal before OCR.

    Erases long horizontal and vertical runs (table borders, form
    rules, underlines) that Tesseract tends to misread as letters,
    fuse into adjacent text, or emit as spurious ``|`` / ``_``
    sequences.
    """

    enabled: bool = False
    #: Minimum run length in pixels. Scale with DPI — 50 is right
    #: for 300 DPI; universal_accurate at 500 DPI uses 75.
    min_line_length: int = 50


@dataclass
class PreprocessConfig:
    """Complete preprocessing pipeline configuration."""

    auto_rotate: AutoRotateConfig = field(default_factory=AutoRotateConfig)
    deskew: DeskewConfig = field(default_factory=DeskewConfig)
    dewarp: DewarpConfig = field(default_factory=DewarpConfig)
    binarization: BinarizationConfig = field(default_factory=BinarizationConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    border_removal: BorderRemovalConfig = field(
        default_factory=BorderRemovalConfig,
    )


# ---------------------------------------------------------------------------
# OCR settings
# ---------------------------------------------------------------------------


@dataclass
class OCRConfig:
    """OCR engine configuration (EasyOCR back-end)."""

    engine: OCREngineKind = OCREngineKind.EASYOCR
    #: EasyOCR language codes: ``"ru"``, ``"en"``. Older profiles with
    #: Tesseract codes (``"rus"``, ``"eng"``) are remapped on load.
    languages: list[str] = field(default_factory=lambda: ["ru", "en"])
    primary_language: str = "ru"
    dpi: int = DEFAULT_DPI
    #: Optional character allowlist (passed to ``readtext(allowlist=)``).
    #: Empty string = accept everything.
    allowlist: str = ""
    #: User-facing confidence threshold (0..100). Used by the
    #: confidence filter for drop / soft-rescue decisions.
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    #: Engine-side minimum confidence (0..1) — boxes below this are
    #: dropped BEFORE they reach the filter. Protects downstream code
    #: from the ``conf<10%`` hallucination blocks EasyOCR emits on
    #: stamps / borders.
    min_keep_confidence: float = 0.1
    #: Use GPU (CUDA) for inference when available. Safe to leave True
    #: on CPU-only hosts — EasyOCR falls back automatically.
    gpu: bool = False
    skip_text: bool = True  # don't re-OCR pages with existing text layer
    #: If >0, only the first N pages of each input are processed. 0 (the
    #: default) disables the limit and means "OCR the whole document".
    max_pages: int = 0
    #: When True, rebuild the per-page extracted text from the engine's
    #: per-word output, keeping ONLY words whose confidence
    #: meets or exceeds :attr:`confidence_threshold`. Dropping sub-
    #: threshold words lifts perceived accuracy: a 51 %-mean document
    #: typically presents as 80–90 % once the noise is gone.
    drop_low_conf_words: bool = False
    #: Soft-rescue layer on top of :attr:`drop_low_conf_words`. Pure-
    #: digit runs (ИНН, КПП, amounts, phones, dates) and clean all-
    #: caps Cyrillic acronyms can report at 50–59 % on noisy forms
    #: even though the read is unambiguous. With soft rescue on,
    #: words in ``[max(threshold-15, 45), threshold)`` are kept when
    #: the token shape passes a lexical-validity check. Mixed-script
    #: short tokens and anything below 45 still drop.
    soft_rescue_dropped_words: bool = False
    #: When True, :attr:`confidence_threshold` becomes a *nominal*
    #: value adapted per page by that page's mean confidence: clean
    #: pages (≥ 90 %) lower to ``min(nominal, 40)``; noisy pages
    #: (< 70 %) raise to ``max(nominal, 70)``.
    adaptive_confidence_threshold: bool = False
    #: Fuzzy-match rescue against ``resources/ru_lexicon.txt``. For
    #: words in the 30–75 confidence band, find the closest
    #: dictionary entry within Levenshtein distance 1–2 and swap in
    #: the canonical spelling. Pure dict + edit-distance lookup.
    user_words_fuzzy_rescue: bool = False


# ---------------------------------------------------------------------------
# Text post-processing
# ---------------------------------------------------------------------------


@dataclass
class RegexRule:
    """A single user-defined find/replace rule.

    Validation is eager: an invalid regex is detected and the rule is
    auto-disabled at construction time (on profile load) rather than
    waiting until the first OCR run to log a ``re.error`` and silently
    skip it. ``invalid_reason`` records the compile error so the UI can
    surface ``Правило #N отключено: <причина>`` next to the rule row
    instead of leaving the user wondering why their substitution has
    no effect.
    """

    pattern: str
    replacement: str
    enabled: bool = True
    description: str = ""
    is_regex: bool = True
    case_sensitive: bool = True
    #: Populated with a non-empty string when ``__post_init__`` rejects
    #: the pattern. A non-empty value always implies ``enabled=False``.
    invalid_reason: str = ""

    def __post_init__(self) -> None:
        """Validate the pattern up front and auto-disable on compile error.

        Only runs the validation when the rule is both marked as a
        regex and currently ``enabled=True`` — a disabled rule with a
        bad pattern is the user's business, not ours to flag. Literal
        (non-regex) rules never go through ``re.compile`` at runtime,
        so they always validate.
        """
        if not self.is_regex or not self.enabled:
            return
        import re as _re

        try:
            _re.compile(self.pattern)
        except _re.error as exc:
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "Правило %r отключено: неверный regex (%s)",
                self.pattern, exc,
            )
            self.enabled = False
            self.invalid_reason = f"{type(exc).__name__}: {exc}"


@dataclass
class PostprocessConfig:
    """Text post-processing configuration."""

    autocorrect_russian: bool = True
    autocorrect_english: bool = True
    merge_hyphenated: bool = True
    normalize_whitespace: bool = True
    normalize_unicode: bool = True
    remove_artifacts: bool = True
    #: Word-level Latin↔Cyrillic look-alike normalisation. Tesseract
    #: mixes ``O/О``, ``A/А``, ``E/Е``, ``K/К``, ``H/Н``, ``P/Р`` etc.
    #: at word boundaries where the in-context regex autocorrect can't
    #: fire. Enabled by default — on pure-Latin / pure-Cyrillic words
    #: the fixup is a no-op, and on genuinely-mixed content (emails,
    #: URLs, product codes) the classifier bails out rather than
    #: corrupt anything.
    fix_cyrillic_latin_confusion: bool = True
    #: Strictness level for :mod:`src.core.garbage_filter`. The filter
    #: drops line-level OCR garbage — symbol walls, ruler lines, and
    #: (on strict) orphan single-letter lines / low-letter-ratio
    #: runs. Stored as a string so JSON profiles round-trip cleanly
    #: without a custom encoder. Accepted values: ``"disabled"``,
    #: ``"lenient"`` (default), ``"strict"``.
    garbage_filter_strictness: str = "lenient"
    #: When True, replace Tesseract blocks whose mean per-word
    #: confidence is below 40 % (and contain ≥ 3 words) with the
    #: marker ``⟨рукописный текст⟩`` in the user-facing text.
    #: Tesseract's Russian LSTM was trained on printed text and
    #: returns long runs of low-confidence noise on handwritten
    #: regions — the word-conf filter correctly drops that noise
    #: but leaves a silent gap the user can't distinguish from
    #: "nothing was there". The marker makes the gap explicit so
    #: the user knows WHERE to type the handwritten content
    #: manually. Off by default for backwards-compatibility;
    #: ``universal_accurate`` opts in.
    mark_suspect_handwritten_blocks: bool = False
    #: When True, scan the OCR output for digit-only tokens that look
    #: like Russian business identifiers (ИНН 10/12-digit, ОГРН 13/15-
    #: digit) and replace a 1-edit-distance typo with the canonical
    #: value from the known-good catalog. Only fires when the
    #: ``TextPostprocessor`` was constructed with a non-empty
    #: :class:`src.core.doc_catalog.DocCatalog` — otherwise it's a
    #: silent no-op. Off by default so JSON-profile migrations from
    #: pre-v3 schemas stay byte-exact; ``universal_accurate`` opts in.
    validate_identifiers: bool = False
    #: When True, normalise dates / amounts / phone numbers in the
    #: OCR output to their canonical Russian business-document
    #: forms: ``DD.MM.YYYY``, ``1 234,56`` (thousand-separated),
    #: and ``+7 (XXX) XXX-XX-XX``. Also repairs single letter-digit
    #: OCR errors on those entities (``12.O1.2O23`` → ``12.01.2023``,
    #: ``+7 (495) 725-8O-62`` → ``+7 (495) 725-80-62``,
    #: ``1 2З4,56`` → ``1 234,56``). Off by default for backwards-
    #: compatibility; ``universal_accurate`` (раньше также ``quick_reliable``, удалён в декабре 2026)
    #: opt in. Implemented in :mod:`src.core.entity_validators`;
    #: skips tokens that aren't entity-shaped so prose isn't affected.
    validate_entities: bool = False
    #: Fuzzy-corrector (~17k ru_lexicon.txt forms). Opt-in, off by
    #: default: применяется ко ВСЕМ русским токенам ≥ 6 chars и
    #: может менять legitimate word-forms (организация↔организации).
    #: Полезен на heavily-mangled scan corpus'е (≥ 30 % CER на OCR),
    #: снижает WER/CER на clean synthetic fixtures из-за form-
    #: mismatch с ground-truth. Включайте явно только для
    #: scan-heavy workflow'ов. Декабрь 2026.
    fuzzy_correction_ru: bool = False
    custom_rules: list[RegexRule] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Post-OCR structured-field extraction (parser: waybill / UPD / invoice)
# ---------------------------------------------------------------------------


@dataclass
class LlmFallbackConfig:
    """Optional Claude-API fallback for fields the regex parser left MISSING.

    Default OFF to preserve the app's offline-first guarantee. When
    enabled, the orchestrator calls
    :func:`src.tn_parser.llm_fallback.improve_row` for rows whose
    overall confidence is below :attr:`min_confidence`. The ANTHROPIC
    API key is NOT stored in the profile (exporting a profile would
    then leak credentials); it lives in the user's settings.json.

    When the ``anthropic`` package is missing or the key is absent,
    :func:`improve_row` returns the row unchanged, so ``enabled=True``
    on an offline install silently degrades to the regex-only path.
    """

    enabled: bool = False
    min_confidence: float = 0.4
    model: str = "claude-opus-4-7"
    max_text_chars: int = 20_000


@dataclass
class ExtractConfig:
    """Post-OCR structured-field extraction configuration.

    When ``enabled=False`` (the default) the pipeline does not invoke
    the parser — profiles authored before schema v11 behave exactly
    as they did. When ``enabled=True`` the orchestrator dispatches on
    :attr:`kind` to pick the parser module:

        ``"tn_upd"`` → :mod:`src.tn_parser` (Russian транспортные
        накладные + УПД)

    Additional parsers register their key here without modifying
    :class:`~src.application.pipeline.OCRPipeline`.
    """

    enabled: bool = False
    #: Dispatcher key. Unused while ``enabled=False``.
    kind: str = "tn_upd"
    #: Allow one PDF to contain several documents (via ``\f`` between
    #: pages). Matches the parser's ``split_documents`` stage.
    multi_document: bool = True
    #: Minimum character count after normalisation; below this the
    #: parser emits a ``LOW_TEXT`` note instead of extracting. Mirrors
    #: ``src.tn_parser.core.LOW_TEXT_THRESHOLD`` so the value can be
    #: tuned per profile without editing parser constants.
    low_text_threshold: int = 200
    #: When True, parser output is cached by SHA1(path|size|mtime).
    #: Off in CI / golden-test runs to force a clean parse.
    cache_enabled: bool = True
    #: Consult :mod:`src.tn_parser.org_lookup` to enrich party fields
    #: with ИНН → name/address data from the bundled catalogue. Pure
    #: local lookup; no network access.
    org_lookup: bool = True
    llm_fallback: LlmFallbackConfig = field(default_factory=LlmFallbackConfig)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


# Bump whenever OCRConfig / PreprocessConfig / PostprocessConfig grow a
# field that would make a newer JSON unreadable by an older binary —
# the reader uses ``_migrate_profile_dict`` to apply compatibility
# shims for every version below the current one.
PROFILE_SCHEMA_VERSION: int = 13


@dataclass
class ProfileData:
    """Complete settings profile: preprocessing + OCR + postprocessing."""

    name: str
    description: str = ""
    schema_version: int = PROFILE_SCHEMA_VERSION
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    #: Post-OCR structured-field extraction. Default ``enabled=False``
    #: means the OCR pipeline behaves exactly as in schema v10 —
    #: existing profiles round-trip unchanged.
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    builtin: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileData:
        """Deserialize from a JSON-loaded dict, applying schema migrations."""
        data = _migrate_profile_dict(dict(data))  # shallow copy, migrations mutate
        return _dict_to_dataclass(cls, data)


def _migrate_profile_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Apply in-place upgrades for older profile schemas.

    Invariant: for every input ``data`` with ``schema_version ≤
    PROFILE_SCHEMA_VERSION``, the returned dict is loadable by the
    current :class:`ProfileData`. Unknown future schemas (version
    greater than current) are returned unchanged with a warning logged
    so the app can still load them best-effort.
    """
    import logging as _logging

    log = _logging.getLogger(__name__)
    version = int(data.get("schema_version", 0))

    if version > PROFILE_SCHEMA_VERSION:
        log.warning(
            "Profile '%s' has schema_version=%d > current %d; loading best-effort",
            data.get("name", "?"),
            version,
            PROFILE_SCHEMA_VERSION,
        )
        data["schema_version"] = PROFILE_SCHEMA_VERSION
        return data

    # v0 → v1: pre-versioning profiles had no `engine` field on OCR.
    if version < 1:
        ocr = data.setdefault("ocr", {})
        ocr.setdefault("engine", "tesseract")
        data["schema_version"] = 1
        version = 1
        log.info("Migrated profile '%s' to schema v1", data.get("name", "?"))

    # v1 → v2: bump tesseract_timeout from 120 → 300. The old default
    # was too low for 600 DPI on complex pages — users hit timeout
    # crashes (FileNotFoundError in OCRmyPDF's graft phase) on every
    # dense Russian contract. Also add ``max_pages: 0`` if absent.
    if version < 2:
        ocr = data.setdefault("ocr", {})
        old_timeout = ocr.get("tesseract_timeout", 0)
        if old_timeout and old_timeout < 300:
            ocr["tesseract_timeout"] = 300
            log.info(
                "Migrated profile '%s' tesseract_timeout %d → 300",
                data.get("name", "?"), old_timeout,
            )
        ocr.setdefault("max_pages", 0)
        data["schema_version"] = 2
        version = 2

    # v2 → v3: add the ``extra_tesseract_params`` dict. Old profiles
    # without it get the empty default (no -c flags), which preserves
    # their exact prior behaviour; the builtin profile builders seed
    # the recommended defaults on freshly-installed machines.
    if version < 3:
        ocr = data.setdefault("ocr", {})
        ocr.setdefault("extra_tesseract_params", {})
        data["schema_version"] = 3
        version = 3

    # v3 → v4: add ``auto_rotate`` to preprocessing AND the
    # ``adaptive_confidence_threshold`` flag to OCR. Old profiles
    # get ``auto_rotate.enabled=True`` (opt-in by default — cheap
    # OSD call, fixes sideways pages) and
    # ``adaptive_confidence_threshold=False`` (opt-OUT by default,
    # preserving the exact filter behaviour pre-v4 profiles saw).
    # Builtin profile builders still set ``True`` on the opinionated
    # presets (``universal_accurate`` — после декабря 2026 единственный builtin).
    if version < 4:
        pre = data.setdefault("preprocess", {})
        pre.setdefault(
            "auto_rotate", {"enabled": True, "min_confidence": 1.0}
        )
        ocr = data.setdefault("ocr", {})
        ocr.setdefault("adaptive_confidence_threshold", False)
        ocr.setdefault("per_word_script_disambiguation", False)
        data["schema_version"] = 4
        version = 4

    # v4 → v5: add ``mark_suspect_handwritten_blocks`` to post-
    # processing. Old profiles get ``False`` (opt-out by default —
    # preserves byte-identical output for pre-v5 profiles); the
    # builtin builders turn it on for the opinionated presets
    # (``universal_accurate`` — после декабря 2026 единственный builtin).
    if version < 5:
        post = data.setdefault("postprocess", {})
        post.setdefault("mark_suspect_handwritten_blocks", False)
        data["schema_version"] = 5
        version = 5

    # v5 → v6: add the per-word image-enhancement rescue flags.
    # Old profiles default both to False to preserve byte-identical
    # behaviour; builtin builders turn them on in
    # ``universal_accurate``.
    if version < 6:
        ocr = data.setdefault("ocr", {})
        ocr.setdefault("per_word_clahe_rescue", False)
        ocr.setdefault("per_word_upscale_rescue", False)
        data["schema_version"] = 6
        version = 6

    # v6 → v7: add ``user_words_fuzzy_rescue`` flag. Default False
    # on migration — rescue is opt-in and universal_accurate's
    # builder turns it on explicitly.
    if version < 7:
        ocr = data.setdefault("ocr", {})
        ocr.setdefault("user_words_fuzzy_rescue", False)
        data["schema_version"] = 7
        version = 7

    # v7 → v8: add ``validate_entities`` flag to postprocess.
    # Default False — preserves byte-identical output for old
    # profiles. Builtin builders turn it on for ``universal_accurate``
    # (``quick_reliable`` удалён в декабре 2026).
    if version < 8:
        post = data.setdefault("postprocess", {})
        post.setdefault("validate_entities", False)
        data["schema_version"] = 8
        version = 8

    # v8 → v9: add ``per_block_psm_retry`` flag. Default False on
    # migration; ``universal_accurate`` opts in.
    if version < 9:
        ocr = data.setdefault("ocr", {})
        ocr.setdefault("per_block_psm_retry", False)
        data["schema_version"] = 9
        version = 9

    # v9 → v10: add ``soft_rescue_dropped_words`` flag. Default False
    # on every existing profile so the filter behaviour is unchanged —
    # ``universal_accurate`` (раньше также ``quick_reliable``, удалён в декабре 2026) flip it to True
    # via their builders, not via a migration rewrite, to keep the
    # migration minimal and reversible.
    if version < 10:
        ocr = data.setdefault("ocr", {})
        ocr.setdefault("soft_rescue_dropped_words", False)
        data["schema_version"] = 10
        version = 10

    # v10 → v11: add ``extract`` section for post-OCR parser
    # (транспортные накладные / УПД). Default ``enabled=False`` keeps
    # the OCR pipeline behaviourally identical for every pre-v11
    # profile; the ``tn_upd`` builtin (added in the same release)
    # ships with ``enabled=True``. The ``llm_fallback`` sub-section
    # is nested so future parser knobs can live alongside it without
    # another schema bump.
    if version < 11:
        extract = data.setdefault("extract", {})
        extract.setdefault("enabled", False)
        extract.setdefault("kind", "tn_upd")
        extract.setdefault("multi_document", True)
        extract.setdefault("low_text_threshold", 200)
        extract.setdefault("cache_enabled", True)
        extract.setdefault("org_lookup", True)
        llm = extract.setdefault("llm_fallback", {})
        llm.setdefault("enabled", False)
        llm.setdefault("min_confidence", 0.4)
        llm.setdefault("model", "claude-opus-4-7")
        llm.setdefault("max_text_chars", 20_000)
        data["schema_version"] = 11
        version = 11

    # v11 → v12 (апрель 2026): add ``postprocess.fuzzy_correction_ru``
    # default False (preserves exact behaviour for старых профилей,
    # builtin'ы перестраивают себя с каждым initialize_builtins).
    if version < 12:
        post = data.setdefault("postprocess", {})
        post.setdefault("fuzzy_correction_ru", False)
        data["schema_version"] = 12
        version = 12

    # v12 → v13 (апрель 2026): Tesseract → EasyOCR migration.
    # Strip Tesseract-specific OCR fields, remap language codes
    # (rus→ru, eng→en), and switch engine default. Unknown fields in
    # the remaining ``ocr`` section are tolerated by ``_convert_value``.
    if version < 13:
        ocr = data.setdefault("ocr", {})
        ocr["engine"] = "easyocr"
        lang_map = {"rus": "ru", "eng": "en"}
        if "languages" in ocr:
            ocr["languages"] = [
                lang_map.get(lang, lang) for lang in ocr["languages"]
            ]
        if "primary_language" in ocr:
            ocr["primary_language"] = lang_map.get(
                ocr["primary_language"], ocr["primary_language"]
            )
        if "char_whitelist" in ocr and "allowlist" not in ocr:
            ocr["allowlist"] = ocr["char_whitelist"]
        for dead in (
            "psm", "oem", "tesseract_timeout", "char_whitelist",
            "char_blacklist", "use_user_dictionaries",
            "extra_tesseract_params", "per_word_script_disambiguation",
            "per_word_clahe_rescue", "per_word_upscale_rescue",
            "per_block_psm_retry", "redact_noisy_blocks",
            "optimize_level",
        ):
            ocr.pop(dead, None)
        ocr.setdefault("gpu", False)
        ocr.setdefault("min_keep_confidence", 0.1)
        data["schema_version"] = 13
        version = 13

    # Future migrations go here: `if version < 14: ...`

    return data


# ---------------------------------------------------------------------------
# Job / Queue
# ---------------------------------------------------------------------------


@dataclass
class OCRJobConfig:
    """Full configuration for a single OCR job (one file)."""

    input_path: str
    output_path: str
    profile: ProfileData
    export_formats: list[str] = field(default_factory=lambda: ["pdf"])
    priority: int = 0

    @property
    def input(self) -> Path:
        return Path(self.input_path)

    @property
    def output(self) -> Path:
        return Path(self.output_path)


@dataclass
class PageResult:
    """Result for a single page."""

    page_number: int  # 1-based
    text: str = ""
    mean_confidence: float = 0.0
    low_confidence_words: list[str] = field(default_factory=list)
    processing_time_sec: float = 0.0
    error: str | None = None
    skew_angle: float = 0.0
    #: Raw ``pytesseract.image_to_data(output_type=DICT)`` на preprocessed
    #: PNG. Используется парсером для layout-aware section detection
    #: (src.tn_parser.layout_anchor) и token-level confidence propagation
    #: (src.tn_parser.token_confidence). Опциональное поле — только если
    #: pipeline.compute_confidence=True. None не означает «OCR failed»,
    #: просто means «TSV не был вычислен» (например cache-hit).
    tsv_data: dict | None = None
    #: Ширина preprocessed страницы в px (того же raster'а что tsv_data).
    #: Нужна для layout_anchor.find_tokens_by_column. 0 = unknown.
    page_width_px: int = 0
    #: Абсолютный путь к preprocessed PNG этой страницы. Pipeline
    #: сохраняет если ``preserve_page_rasters=True`` в профиле (off by
    #: default — большие PNG'и). Используется field_rescue для
    #: targeted re-OCR конкретных bbox'ов. None если raster уже удалён.
    raster_path: str | None = None


@dataclass
class ParsedDocument:
    """Structured fields extracted from a job by the post-OCR parser.

    Populated when the profile's :class:`ExtractConfig` had
    ``enabled=True`` and the parser produced at least one row.
    ``None`` on :attr:`JobResult.parsed` means extraction was
    disabled, the parser was unavailable, or the text was too short
    / produced no rows — distinguishing those cases is the
    orchestrator's job, visible through the application log.

    :attr:`rows` stores rows in the JSON-dict form produced by
    :meth:`src.tn_parser.models.ParsedRow.to_json_dict`, NOT the
    dataclass instances. Storing dicts keeps :mod:`src.core` free
    of a compile-time dependency on :mod:`src.tn_parser` (the parser
    lives in a parallel top-level package and must not become a
    hard dependency of core domain types — the layer boundary
    matters: a user running a pipeline without ``extract.enabled``
    should be able to import :mod:`src.core.models` even if
    ``src.tn_parser`` is absent from a custom build).

    Attributes:
        rows: One dict per extracted document. The average ТН PDF
            contributes one; сводные УПД + реестр may contribute
            several. Each dict has the field names documented on
            :class:`src.tn_parser.models.ParsedRow`.
        overall_confidence: Mean of per-row ``confidence.overall()``
            values in the ``[0.0, 1.0]`` range. Zero when ``rows``
            is empty. Use to colour UI rows and to gate "auto-open
            the editor on finish".
        snapshot_path: Populated by the Excel exporter when the
            ``.xlsx.snapshot.json`` sidecar is written. Used by the
            feedback-loop tooling (``scripts/collect_feedback.py``)
            to diff operator corrections against the original parser
            output. Remains ``None`` when no Excel was exported.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    overall_confidence: float = 0.0
    snapshot_path: str | None = None


@dataclass
class JobResult:
    """Aggregated result for a completed job."""

    job_id: str
    status: JobStatus
    input_path: str
    output_path: str
    pages: list[PageResult] = field(default_factory=list)
    total_time_sec: float = 0.0
    error: str | None = None
    #: Structured-field extraction output from the post-OCR parser,
    #: or ``None`` when extraction was disabled / unavailable / yielded
    #: no rows. Populated by
    #: :func:`src.application.parsers.tn_orchestrator.extract_from_pages`
    #: when the profile's :attr:`ProfileData.extract.enabled` is True.
    parsed: ParsedDocument | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def average_confidence(self) -> float:
        if not self.pages:
            return 0.0
        vals = [p.mean_confidence for p in self.pages if p.mean_confidence > 0]
        return sum(vals) / len(vals) if vals else 0.0


@dataclass
class QueueItem:
    """A single file in the processing queue."""

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    config: OCRJobConfig | None = None
    status: JobStatus = JobStatus.PENDING
    progress_current: int = 0  # current page
    progress_total: int = 0  # total pages
    error_message: str = ""
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def file_name(self) -> str:
        if self.config is None:
            return "(unknown)"
        return Path(self.config.input_path).name

    @property
    def progress_pct(self) -> float:
        if self.progress_total <= 0:
            return 0.0
        return min(100.0, 100.0 * self.progress_current / self.progress_total)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Recursively convert dataclass → dict with enum coercion."""
    raw = asdict(obj)
    return _coerce_enums(raw)


def _coerce_enums(value: Any) -> Any:
    """Convert Enum → value recursively."""
    import enum

    if isinstance(value, dict):
        return {k: _coerce_enums(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_enums(v) for v in value]
    if isinstance(value, enum.Enum):
        return value.value
    return value


# Sentinel returned by ``_convert_value`` when the raw JSON value was
# present but could not be interpreted as the target type (unknown enum
# member, legacy shape, etc.). The caller drops the key from ``kwargs``
# so the dataclass field keeps its own default — equivalent to the JSON
# never having mentioned the field. Using an object() instead of None
# matters because ``None`` is a legitimate value for ``Optional`` fields
# and conflating the two would silently reset unrelated data.
_USE_DATACLASS_DEFAULT: Any = object()


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Reconstruct a dataclass (possibly nested) from a dict.

    Handles nested dataclasses, lists of dataclasses, and Enums via type hints.
    """
    import dataclasses
    import typing

    if not dataclasses.is_dataclass(cls):
        return data

    type_hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    fields_info = {f.name: f for f in dataclasses.fields(cls)}

    for name, f in fields_info.items():
        if name not in data:
            continue
        raw_value = data[name]
        field_type = type_hints.get(name, f.type)
        converted = _convert_value(raw_value, field_type)
        if converted is _USE_DATACLASS_DEFAULT:
            # The stored value was uninterpretable (e.g. enum member
            # removed in a schema change). Skip the kwarg so the
            # dataclass falls back to its default value instead of
            # blowing up on construction.
            continue
        kwargs[name] = converted
    return cls(**kwargs)


def _convert_value(value: Any, target_type: Any) -> Any:
    """Convert a raw JSON value to the target type (dataclass, Enum, or primitive)."""
    import dataclasses
    import enum
    import logging as _logging
    import typing

    if value is None:
        return None

    origin = typing.get_origin(target_type)
    args = typing.get_args(target_type)

    # list[X]
    if origin is list and args:
        elem_type = args[0]
        return [_convert_value(v, elem_type) for v in value]

    # Union / Optional
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _convert_value(value, non_none[0])
        return value

    # Enum. Catch BOTH TypeError (passed a non-hashable) and ValueError
    # (value is hashable but not a member of the enum). A stored profile
    # that references an enum member the current build no longer knows —
    # for example, a downgraded binary or a hand-edited JSON with
    # ``"engine": "some_unknown_engine"`` — would otherwise raise and
    # leave the entire profile unloadable, dragging every user-saved
    # config with it. Falling back to ``None`` lets the surrounding
    # :class:`dataclasses` default kick in (e.g. ``OCRConfig.engine``
    # reverts to ``OCREngineKind.TESSERACT``), which is the behaviour a
    # user expects from "my one odd field got reset" rather than "my
    # whole profile is gone".
    if isinstance(target_type, type) and issubclass(target_type, enum.Enum):
        try:
            return target_type(value)
        except (TypeError, ValueError):
            _logging.getLogger(__name__).warning(
                "Unknown %s value %r in profile JSON; falling back to "
                "dataclass default",
                getattr(target_type, "__name__", target_type),
                value,
            )
            return _USE_DATACLASS_DEFAULT

    # Nested dataclass
    if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
        return _dict_to_dataclass(target_type, value)

    return value
