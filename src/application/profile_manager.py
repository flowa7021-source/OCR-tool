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
from src.shared.types import (
    OEM,
    PSM,
    BinarizationMethod,
    DenoiseMethod,
    OCREngineKind,
    OptimizeLevel,
)

logger = logging.getLogger(__name__)


BUILTIN_NAMES: tuple[str, ...] = (
    "universal_accurate",
    "default",
    "quick_reliable",
    "low_quality_scan",
    "contracts_ru",
    "english_text",
    "handwritten_mixed",
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
            "universal_accurate": self._build_universal_accurate,
            "default": self._build_default,
            "quick_reliable": self._build_quick_reliable,
            "low_quality_scan": self._build_low_quality,
            "contracts_ru": self._build_contracts_ru,
            "english_text": self._build_english_text,
            "handwritten_mixed": self._build_handwritten_mixed,
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

    def _build_universal_accurate(self) -> ProfileData:
        """Universal max-accuracy preset.

        The "just give me the best text" profile. Every preprocessing
        step that helps OCR accuracy across a wide range of inputs is
        turned on with conservative parameters:

          * **deskew** — auto-detect; essential, non-destructive.
          * **CLAHE** contrast — ``clip=2.0`` for even lighting without
            over-amplifying noise.
          * Background removal is **off** in the universal preset:
            at 600 DPI a ``blur_kernel=55`` pass costs 2–3 s per A4
            page with only a marginal accuracy gain over CLAHE. Users
            with yellowed or photographed pages can toggle it on;
            ``low_quality_scan`` already bundles it.
          * **Denoise chain** — median ``ksize=3`` then a morphological
            close ``ksize=2`` to repair sub-pixel breaks in thin glyphs
            without swallowing dots of ``ё``, ``ь``, ``ъ``.
          * **Adaptive Gaussian binarisation** (``block=31``, ``C=10``)
            instead of OTSU: better on uneven lighting and safe on
            clean pages too.
          * **OCR** at 300 DPI with PSM=AUTO, OEM=LSTM_ONLY (best quality
            Tesseract mode), rus+eng, LOSSLESS PDF.
          * **Post-processing**: everything enabled — Unicode NFC,
            hyphenation merge, whitespace normalization, artifact line
            removal, Russian + English autocorrect.

        Users who need raw speed should pick ``default`` (OTSU,
        single-step median). Users with awful scans should pick
        ``low_quality_scan`` (NLM denoise + larger CLAHE).
        """
        preprocess = PreprocessConfig(
            deskew=DeskewConfig(enabled=True, auto_detect=True, max_angle=45.0),
            dewarp=DewarpConfig(enabled=False),
            # Sauvola adapts threshold per-pixel based on local mean +
            # standard deviation — handles uneven lighting far better
            # than adaptive Gaussian on real-world scans with shadows
            # or page-edge darkening. Window 25 is the typical sweet
            # spot for 300-400 DPI text; k=0.2 is the paper default
            # for documents (lower than k=0.5 which is better for
            # photos).
            binarization=BinarizationConfig(
                method=BinarizationMethod.SAUVOLA,
                sauvola_window=25,
                sauvola_k=0.2,
            ),
            denoise=DenoiseConfig(
                enabled=True,
                steps=[
                    DenoiseStep(method=DenoiseMethod.MEDIAN, ksize=3),
                    DenoiseStep(method=DenoiseMethod.MORPH_CLOSE, morph_ksize=3),
                ],
            ),
            # CLAHE clip 3.0 (was 2.0) gives a more aggressive local
            # contrast boost without the global over-brightening a
            # straight histogram equalise would cause. Makes a
            # measurable difference on faded photocopies where 2.0
            # leaves the text barely darker than the paper.
            contrast=ContrastConfig(
                clahe_enabled=True, clahe_clip=3.0, clahe_tile=8
            ),
            # Background removal ENABLED. Real scanned contracts
            # almost always have a light gradient (scanner lamp
            # unevenness, off-axis lighting). Removing it before
            # Sauvola + CLAHE gives the binariser a flat, clean
            # input. The ~500 ms per page cost is worth the
            # accuracy gain.
            background=BackgroundConfig(enabled=True, blur_kernel=55),
        )
        ocr = OCRConfig(
            languages=["rus", "eng"],
            primary_language="rus",
            psm=PSM.AUTO,
            oem=OEM.LSTM_ONLY,
            # 400 DPI is the practical max for Tesseract on complex
            # Russian contracts. 600 DPI was the original choice but
            # production logs showed page 4 of a real contract timing
            # out at BOTH 300s and 900s — Tesseract's layout analysis
            # on a 600 DPI A4 page produces an image so large (~5000×
            # 7000 pixels) that even the LSTM engine can't finish one
            # page within any reasonable timeout. 400 DPI gives the
            # same recognition accuracy on typical 10-12pt text while
            # keeping per-page processing under 2 minutes.
            dpi=400,
            optimize_level=OptimizeLevel.LOSSLESS,
            confidence_threshold=60.0,
            skip_text=True,
        )
        postprocess = PostprocessConfig(
            autocorrect_russian=True,
            autocorrect_english=True,
            merge_hyphenated=True,
            normalize_whitespace=True,
            normalize_unicode=True,
            remove_artifacts=True,
            # Critical for Russian documents — Tesseract swaps
            # letter pairs like ``О/O`` at word edges, and the
            # in-context regex autocorrect can't catch those.
            fix_cyrillic_latin_confusion=True,
            custom_rules=[],
        )
        return ProfileData(
            name="universal_accurate",
            description=(
                "Универсальный «максимум точности»: 400 DPI, Sauvola + "
                "CLAHE + удаление фона + deskew, полная постобработка "
                "включая нормализацию кириллицы/латиницы"
            ),
            preprocess=preprocess,
            ocr=ocr,
            postprocess=postprocess,
        )

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

    def _build_quick_reliable(self) -> ProfileData:
        """Low-risk fallback profile: gets OCR output even on hard cases.

        Built for the user who just needs a *result* — not the highest
        accuracy, not the fanciest engine, just a searchable PDF on
        disk. Intentionally conservative on every axis where an
        aggressive choice could fail or hang:

          * **Tesseract**, never GOT-OCR 2.0 — the transformer path
            depends on a ~580 MB optional model download; if any of its
            ``trust_remote_code`` Python modules is missing the job
            dies at load time.
          * **300 DPI**, not 400 / 600 — at 600 DPI the ``universal_accurate``
            profile hit ``tesseract_timeout`` on dense Russian contract
            pages even with the auto-retry escalation.
          * **OTSU** binarisation — single-threshold, deterministic,
            fast; adaptive / Sauvola can produce artefacts that confuse
            Tesseract's layout analysis (``pixClipBoxToForeground``
            warnings in production logs).
          * **Denoise OFF** — one less step that can fail. Text from
            a modern scanner is already clean enough for Tesseract;
            denoise mostly helps on photographed documents, which are
            a different profile's job.
          * **Dewarp / background removal OFF** — expensive and
            optional; their payoff is on phone-camera pages, not flat
            scans.
          * **CLAHE contrast ON** — cheap, never hurts, helps on
            uneven illumination.
          * **tesseract_timeout=300** (matches the new default)
            plus the auto-retry inside ``run_ocrmypdf`` gives two
            chances per page, so even a slow page lands within the
            same job.
          * **Post-processing: everything enabled** — Russian +
            English autocorrect, NFC, hyphen merge, artifact strip.
            These are pure-Python and cannot fail the job.

        Marketed as "use this when anything else breaks" — documented
        explicitly in the profile description so UI users see it.
        """
        preprocess = PreprocessConfig(
            deskew=DeskewConfig(enabled=True, auto_detect=True, max_angle=45.0),
            dewarp=DewarpConfig(enabled=False),
            binarization=BinarizationConfig(method=BinarizationMethod.OTSU),
            denoise=DenoiseConfig(enabled=False, steps=[]),
            contrast=ContrastConfig(clahe_enabled=True, clahe_clip=2.0),
            background=BackgroundConfig(enabled=False),
        )
        ocr = OCRConfig(
            engine=OCREngineKind.TESSERACT,
            languages=["rus", "eng"],
            primary_language="rus",
            psm=PSM.AUTO,
            oem=OEM.LSTM_ONLY,
            dpi=300,
            confidence_threshold=50.0,
            # Explicit 300s even though the constant default is already
            # 300 — spelling it out future-proofs the profile against
            # another default-constant tweak.
            tesseract_timeout=300,
            optimize_level=OptimizeLevel.LOSSLESS,
            skip_text=True,
        )
        return ProfileData(
            name="quick_reliable",
            description=(
                "Быстрый и надёжный. Рекомендуется, если другие профили "
                "падают с ошибкой (таймаут, не хватает памяти). 300 DPI, "
                "Tesseract, минимум шагов."
            ),
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

    def _build_handwritten_mixed(self) -> ProfileData:
        """GOT-OCR 2.0 для рукописного и печатного текста.

        Требует отдельного скачивания модели через меню «Движок OCR».
        Бинаризация отключена — modern transformer-OCR работает лучше
        на серых полутонах. Лёгкий CLAHE сохраняем для контраста.
        """
        preprocess = PreprocessConfig(
            deskew=DeskewConfig(enabled=True, auto_detect=True),
            dewarp=DewarpConfig(enabled=False),
            binarization=BinarizationConfig(method=BinarizationMethod.NONE),
            denoise=DenoiseConfig(enabled=False),
            contrast=ContrastConfig(clahe_enabled=True, clahe_clip=2.5),
            background=BackgroundConfig(enabled=False),
        )
        ocr = OCRConfig(
            engine=OCREngineKind.GOT_OCR2,
            languages=["rus", "eng"],
            primary_language="rus",
            dpi=300,
            confidence_threshold=50.0,
            optimize_level=OptimizeLevel.LOSSLESS,
        )
        return ProfileData(
            name="handwritten_mixed",
            description=(
                "GOT-OCR 2.0 для рукописного и печатного текста "
                "(требует отдельной модели)"
            ),
            preprocess=preprocess,
            ocr=ocr,
            postprocess=PostprocessConfig(),
        )
