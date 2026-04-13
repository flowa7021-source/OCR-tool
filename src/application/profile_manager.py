"""High-level profile management service.

Wraps :class:`~src.infrastructure.config_storage.ProfileStorage` with in-memory
"current profile" state and builder methods for the four builtin profiles that
ship with the application.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.models import (
    BackgroundConfig,
    BinarizationConfig,
    ContrastConfig,
    DenoiseConfig,
    DenoiseStep,
    DeskewConfig,
    DewarpConfig,
    OCRConfig,
    PostprocessConfig,
    PreprocessConfig,
    ProfileData,
)
from src.infrastructure.config_storage import ProfileStorage
from src.shared.types import BinarizationMethod, DenoiseMethod, OEM, OptimizeLevel, PSM

logger = logging.getLogger(__name__)


BUILTIN_NAMES: tuple[str, ...] = (
    "default",
    "low_quality_scan",
    "contracts_ru",
    "english_text",
)


class ProfileManager:
    """In-memory profile service with on-disk persistence.

    Attributes:
        storage: The underlying :class:`ProfileStorage` instance.
    """

    def __init__(self, storage: ProfileStorage) -> None:
        """Create the manager and remember the chosen ``default`` profile."""
        self.storage = storage
        self._current_name: str = "default"

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
        """Remember ``name`` as the currently selected profile.

        Args:
            name: Profile name. The method validates that the profile loads.
        """
        _ = self.storage.load(name)  # raises if absent
        self._current_name = name
        logger.info("Current profile set to: %s", name)

    def get_current(self) -> ProfileData:
        """Return the currently selected profile."""
        try:
            return self.storage.load(self._current_name)
        except FileNotFoundError:
            logger.warning(
                "Current profile %s missing; falling back to 'default'",
                self._current_name,
            )
            self._current_name = "default"
            return self.storage.load("default")

    # ------------------------------------------------------------------
    # Duplication / import / export
    # ------------------------------------------------------------------

    def duplicate(self, name: str, new_name: str) -> ProfileData:
        """Duplicate profile ``name`` under ``new_name``.

        The duplicate is saved as a user profile (``builtin=False``).
        """
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
        """Ensure the four builtin profiles are available on disk.

        Profiles that already exist are left untouched so user modifications
        persist across upgrades.
        """
        builders = {
            "default": self._build_default,
            "low_quality_scan": self._build_low_quality,
            "contracts_ru": self._build_contracts_ru,
            "english_text": self._build_english_text,
        }
        for name, builder in builders.items():
            try:
                existing = self.storage.load(name)
                logger.debug("Builtin profile already present: %s", existing.name)
            except FileNotFoundError:
                profile = builder()
                profile.builtin = True
                self.storage.save(profile)
                logger.info("Seeded builtin profile: %s", profile.name)

    # -- individual builders ----------------------------------------------

    def _build_default(self) -> ProfileData:
        """Balanced defaults suitable for most scans."""
        preprocess = PreprocessConfig(
            deskew=DeskewConfig(enabled=True, auto_detect=True),
            dewarp=DewarpConfig(enabled=False),
            binarization=BinarizationConfig(method=BinarizationMethod.OTSU),
            denoise=DenoiseConfig(
                enabled=True,
                steps=[DenoiseStep(method=DenoiseMethod.MEDIAN, ksize=3)],
            ),
            contrast=ContrastConfig(clahe_enabled=True, clahe_clip=2.0),
            background=BackgroundConfig(enabled=False),
        )
        ocr = OCRConfig(
            languages=["rus", "eng"],
            primary_language="rus",
            psm=PSM.AUTO,
            oem=OEM.LSTM_ONLY,
            dpi=300,
            optimize_level=OptimizeLevel.LOSSLESS,
        )
        return ProfileData(
            name="default",
            description="Сбалансированные настройки по умолчанию (rus+eng, OTSU, CLAHE)",
            preprocess=preprocess,
            ocr=ocr,
            postprocess=PostprocessConfig(),
        )

    def _build_low_quality(self) -> ProfileData:
        """Aggressive cleanup for blurry / noisy / low-contrast scans."""
        preprocess = PreprocessConfig(
            deskew=DeskewConfig(enabled=True, auto_detect=True),
            dewarp=DewarpConfig(enabled=False),
            binarization=BinarizationConfig(
                method=BinarizationMethod.ADAPTIVE_GAUSSIAN,
                adaptive_block_size=31,
                adaptive_c=10,
            ),
            denoise=DenoiseConfig(
                enabled=True,
                steps=[
                    DenoiseStep(method=DenoiseMethod.NLM, h=15),
                    DenoiseStep(method=DenoiseMethod.MEDIAN, ksize=5),
                ],
            ),
            contrast=ContrastConfig(clahe_enabled=True, clahe_clip=4.0),
            background=BackgroundConfig(enabled=True, blur_kernel=55),
        )
        ocr = OCRConfig(
            languages=["rus", "eng"],
            primary_language="rus",
            psm=PSM.AUTO,
            oem=OEM.LSTM_ONLY,
            dpi=400,
            optimize_level=OptimizeLevel.LOSSLESS,
        )
        return ProfileData(
            name="low_quality_scan",
            description="Агрессивная обработка для плохо отсканированных документов",
            preprocess=preprocess,
            ocr=ocr,
            postprocess=PostprocessConfig(),
        )

    def _build_contracts_ru(self) -> ProfileData:
        """Russian contracts: single uniform block, minimal preprocessing."""
        preprocess = PreprocessConfig(
            deskew=DeskewConfig(enabled=True, auto_detect=True),
            dewarp=DewarpConfig(enabled=False),
            binarization=BinarizationConfig(method=BinarizationMethod.OTSU),
            denoise=DenoiseConfig(enabled=False),
            contrast=ContrastConfig(clahe_enabled=False),
            background=BackgroundConfig(enabled=False),
        )
        ocr = OCRConfig(
            languages=["rus", "eng"],
            primary_language="rus",
            psm=PSM.SINGLE_BLOCK,
            oem=OEM.LSTM_ONLY,
            dpi=300,
            optimize_level=OptimizeLevel.LOSSLESS,
        )
        return ProfileData(
            name="contracts_ru",
            description="Русские договоры: единый блок текста, минимальная обработка",
            preprocess=preprocess,
            ocr=ocr,
            postprocess=PostprocessConfig(),
        )

    def _build_english_text(self) -> ProfileData:
        """Clean English documents: light preprocessing, eng language only."""
        preprocess = PreprocessConfig(
            deskew=DeskewConfig(enabled=True, auto_detect=True),
            dewarp=DewarpConfig(enabled=False),
            binarization=BinarizationConfig(method=BinarizationMethod.OTSU),
            denoise=DenoiseConfig(enabled=False),
            contrast=ContrastConfig(clahe_enabled=True, clahe_clip=2.0),
            background=BackgroundConfig(enabled=False),
        )
        ocr = OCRConfig(
            languages=["eng"],
            primary_language="eng",
            psm=PSM.AUTO,
            oem=OEM.LSTM_ONLY,
            dpi=300,
            optimize_level=OptimizeLevel.LOSSLESS,
        )
        return ProfileData(
            name="english_text",
            description="English documents with clean layout (eng, OTSU, light CLAHE)",
            preprocess=preprocess,
            ocr=ocr,
            postprocess=PostprocessConfig(),
        )
