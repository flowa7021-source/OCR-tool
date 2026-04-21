"""High-level profile management service.

Wraps :class:`~src.infrastructure.config_storage.ProfileStorage` with in-memory
"current profile" state and builder methods for the two builtin profiles that
ship with the application.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.models import (
    AutoRotateConfig,
    BackgroundConfig,
    BinarizationConfig,
    BorderRemovalConfig,
    ContrastConfig,
    DenoiseConfig,
    DeskewConfig,
    DewarpConfig,
    ExtractConfig,
    LlmFallbackConfig,
    OCRConfig,
    PostprocessConfig,
    PreprocessConfig,
    ProfileData,
)
from src.infrastructure.config_storage import ProfileStorage
from src.shared.types import (
    BinarizationMethod,
    OCREngineKind,
)

logger = logging.getLogger(__name__)


# Два builtin-профиля. ``universal_accurate`` — full postprocess +
# extract.kind=tn_upd для реальных сканов; ``universal_clean`` —
# минимум вмешательства для digital / high-contrast входов.
BUILTIN_NAMES: tuple[str, ...] = ("universal_accurate", "universal_clean")


class ProfileManager:
    """In-memory profile service with on-disk persistence."""

    def __init__(self, storage: ProfileStorage) -> None:
        """Create the manager. Default selection — ``universal_accurate``."""
        self.storage = storage
        self._current_name: str = "universal_accurate"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def list_profiles(self) -> list[ProfileData]:
        """Return all profiles known to the storage layer."""
        return self.storage.list_profiles()

    def load(self, name: str) -> ProfileData:
        """Load a profile by name."""
        return self.storage.load(name)

    def save(self, profile: ProfileData) -> None:
        """Persist ``profile`` to the user profiles directory."""
        self.storage.save(profile)

    def delete(self, name: str) -> None:
        """Delete a user profile by name."""
        self.storage.delete(name)

    # ------------------------------------------------------------------
    # Current profile tracking
    # ------------------------------------------------------------------

    def set_current(self, name: str) -> None:
        """Remember ``name`` as the currently selected profile."""
        _ = self.storage.load(name)  # raises if absent
        self._current_name = name
        logger.info("Current profile set to: %s", name)

    def get_current(self) -> ProfileData:
        """Return the currently selected profile."""
        try:
            return self.storage.load(self._current_name)
        except FileNotFoundError:
            logger.warning(
                "Current profile %s missing; falling back to 'universal_accurate'",
                self._current_name,
            )
            self._current_name = "universal_accurate"
            return self.storage.load("universal_accurate")

    # ------------------------------------------------------------------
    # Duplication / import / export
    # ------------------------------------------------------------------

    def duplicate(self, name: str, new_name: str) -> ProfileData:
        """Duplicate profile ``name`` under ``new_name``."""
        src = self.storage.load(name)
        copy = ProfileData.from_dict(src.to_dict())
        copy.name = new_name
        copy.builtin = False
        copy.description = f"Копия профиля '{name}'" if not src.description else (
            f"{src.description} (копия)"
        )
        self.storage.save(copy)
        return copy

    def export_profile(self, name: str, path: Path) -> None:
        """Write the profile to ``path`` as JSON."""
        profile = self.storage.load(name)
        self.storage.export_to(profile, path)

    def import_profile(self, path: Path) -> ProfileData:
        """Read a profile JSON from ``path`` and save it as a user profile."""
        profile = self.storage.import_from(path)
        self.storage.save(profile)
        return profile

    # ------------------------------------------------------------------
    # Builtin profiles
    # ------------------------------------------------------------------

    def initialize_builtins(self) -> None:
        """Re-seed builtin profiles on disk on every startup.

        Builtin profiles are pere-сидируются on each launch so the user
        always gets canonical settings after an upgrade. User profiles
        (``builtin=False``) are NOT touched — duplicate + edit для
        кастомизации.
        """
        builders = {
            "universal_accurate": self._build_universal_accurate,
            "universal_clean": self._build_universal_clean,
        }
        for name, builder in builders.items():
            profile = builder()
            profile.builtin = True
            try:
                existing = self.storage.load(name)
            except FileNotFoundError:
                existing = None
            if existing is None or existing.builtin:
                self.storage.save(profile)
                logger.info("Re-seeded builtin profile %s.", profile.name)
            else:
                logger.debug(
                    "Profile %s is user-customised (builtin=False); skipping re-seed.",
                    name,
                )

    # -- individual builders ----------------------------------------------

    def _build_universal_accurate(self) -> ProfileData:
        """Full-quality EasyOCR preset for real scans (mirrors profiles/universal_accurate.json)."""
        preprocess = PreprocessConfig(
            auto_rotate=AutoRotateConfig(enabled=True, min_confidence=1.0),
            deskew=DeskewConfig(enabled=True, auto_detect=True, max_angle=45.0),
            dewarp=DewarpConfig(enabled=False),
            binarization=BinarizationConfig(method=BinarizationMethod.OTSU),
            denoise=DenoiseConfig(enabled=False, steps=[]),
            contrast=ContrastConfig(clahe_enabled=False),
            background=BackgroundConfig(enabled=False, blur_kernel=55),
            border_removal=BorderRemovalConfig(enabled=False),
        )
        ocr = OCRConfig(
            engine=OCREngineKind.EASYOCR,
            languages=["ru", "en"],
            primary_language="ru",
            dpi=300,
            allowlist="",
            confidence_threshold=60.0,
            min_keep_confidence=0.1,
            gpu=False,
            skip_text=True,
            max_pages=0,
            drop_low_conf_words=True,
            soft_rescue_dropped_words=True,
            adaptive_confidence_threshold=True,
            user_words_fuzzy_rescue=True,
        )
        postprocess = PostprocessConfig(
            autocorrect_russian=True,
            autocorrect_english=True,
            merge_hyphenated=True,
            normalize_whitespace=True,
            normalize_unicode=True,
            remove_artifacts=True,
            fix_cyrillic_latin_confusion=True,
            garbage_filter_strictness="lenient",
            mark_suspect_handwritten_blocks=True,
            validate_identifiers=True,
            validate_entities=True,
            fuzzy_correction_ru=True,
            custom_rules=[],
        )
        extract = ExtractConfig(
            enabled=True,
            kind="tn_upd",
            multi_document=True,
            low_text_threshold=200,
            cache_enabled=True,
            org_lookup=True,
            llm_fallback=LlmFallbackConfig(enabled=False),
        )
        return ProfileData(
            name="universal_accurate",
            description=(
                "Универсальный default (апрель 2026, EasyOCR): минимальная "
                "предобработка + EasyOCR CRAFT+CRNN ru+en на 300 DPI. "
                "drop_low_conf_words + adaptive threshold + soft_rescue + "
                "user_words_fuzzy_rescue — полный postprocess и tn_upd парсер."
            ),
            preprocess=preprocess,
            ocr=ocr,
            postprocess=postprocess,
            extract=extract,
        )

    def _build_universal_clean(self) -> ProfileData:
        """Minimal-preprocessing EasyOCR preset (mirrors profiles/universal_clean.json)."""
        preprocess = PreprocessConfig(
            auto_rotate=AutoRotateConfig(enabled=True, min_confidence=1.0),
            deskew=DeskewConfig(enabled=True, auto_detect=True),
            dewarp=DewarpConfig(enabled=False),
            binarization=BinarizationConfig(method=BinarizationMethod.OTSU),
            denoise=DenoiseConfig(enabled=False, steps=[]),
            contrast=ContrastConfig(clahe_enabled=False),
            background=BackgroundConfig(enabled=False),
            border_removal=BorderRemovalConfig(enabled=False),
        )
        ocr = OCRConfig(
            engine=OCREngineKind.EASYOCR,
            languages=["ru", "en"],
            primary_language="ru",
            dpi=300,
            allowlist="",
            confidence_threshold=60.0,
            min_keep_confidence=0.1,
            gpu=False,
            skip_text=True,
            max_pages=0,
            drop_low_conf_words=False,
            soft_rescue_dropped_words=False,
            adaptive_confidence_threshold=False,
            user_words_fuzzy_rescue=False,
        )
        postprocess = PostprocessConfig(
            autocorrect_russian=True,
            autocorrect_english=True,
            merge_hyphenated=True,
            normalize_whitespace=True,
            normalize_unicode=True,
            remove_artifacts=True,
            fix_cyrillic_latin_confusion=True,
            garbage_filter_strictness="lenient",
            mark_suspect_handwritten_blocks=False,
            validate_identifiers=False,
            validate_entities=False,
            fuzzy_correction_ru=False,
            custom_rules=[],
        )
        extract = ExtractConfig(
            enabled=True,
            kind="tn_upd",
            multi_document=True,
            low_text_threshold=200,
            cache_enabled=True,
            org_lookup=True,
            llm_fallback=LlmFallbackConfig(enabled=False),
        )
        return ProfileData(
            name="universal_clean",
            description=(
                "Минимальная предобработка для чистых сканов / digital-"
                "экспортов / уже OCR'нутых PDF. Только deskew + OTSU + "
                "EasyOCR. Выгоднее universal_accurate когда на входе "
                "high-contrast скан: предобработка экономит ~10 sec и не "
                "портит исходный raster."
            ),
            preprocess=preprocess,
            ocr=ocr,
            postprocess=postprocess,
            extract=extract,
        )
