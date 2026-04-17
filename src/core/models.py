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
    DEFAULT_TESSERACT_TIMEOUT_SEC,
)
from src.shared.types import (
    OEM,
    PSM,
    BinarizationMethod,
    DenoiseMethod,
    JobStatus,
    OCREngineKind,
    OptimizeLevel,
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
class PreprocessConfig:
    """Complete preprocessing pipeline configuration."""

    deskew: DeskewConfig = field(default_factory=DeskewConfig)
    dewarp: DewarpConfig = field(default_factory=DewarpConfig)
    binarization: BinarizationConfig = field(default_factory=BinarizationConfig)
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)


# ---------------------------------------------------------------------------
# OCR settings
# ---------------------------------------------------------------------------


@dataclass
class OCRConfig:
    """OCR engine configuration.

    The ``engine`` field selects the backend; remaining fields are
    primarily interpreted by Tesseract/OCRmyPDF but alternative engines
    reuse the language list, DPI, and confidence threshold where they
    make sense.
    """

    engine: OCREngineKind = OCREngineKind.TESSERACT
    languages: list[str] = field(default_factory=lambda: ["rus", "eng"])
    primary_language: str = "rus"  # determines priority order in OCR string
    psm: PSM = PSM.AUTO
    oem: OEM = OEM.LSTM_ONLY
    dpi: int = DEFAULT_DPI
    char_whitelist: str = ""
    char_blacklist: str = ""
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    tesseract_timeout: int = DEFAULT_TESSERACT_TIMEOUT_SEC
    optimize_level: OptimizeLevel = OptimizeLevel.LOSSLESS
    skip_text: bool = True  # don't re-OCR pages with existing text
    #: If >0, only the first N pages of each input are processed. 0 (the
    #: default) disables the limit and means "OCR the whole document".
    #: Useful for previewing a profile before running a full 500-page job.
    max_pages: int = 0

    @property
    def tesseract_language_string(self) -> str:
        """Return language string in OCRmyPDF/Tesseract format (e.g. 'rus+eng')."""
        ordered = [self.primary_language] + [
            lang for lang in self.languages if lang != self.primary_language
        ]
        return "+".join(ordered)


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
    custom_rules: list[RegexRule] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


# Bump whenever OCRConfig / PreprocessConfig / PostprocessConfig grow a
# field that would make a newer JSON unreadable by an older binary —
# the reader uses ``_migrate_profile_dict`` to apply compatibility
# shims for every version below the current one.
PROFILE_SCHEMA_VERSION: int = 2


@dataclass
class ProfileData:
    """Complete settings profile: preprocessing + OCR + postprocessing."""

    name: str
    description: str = ""
    schema_version: int = PROFILE_SCHEMA_VERSION
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
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

    # Future migrations go here: `if version < 3: ...`

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
    # ``"engine": "got_ocr3"`` — would otherwise raise and leave the
    # entire profile unloadable, dragging every user-saved config with
    # it. Falling back to ``None`` lets the surrounding
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
