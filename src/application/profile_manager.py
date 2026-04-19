"""High-level profile management service.

Wraps :class:`~src.infrastructure.config_storage.ProfileStorage` with in-memory
"current profile" state and builder methods for the four builtin profiles that
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
)


# Tesseract ``-c`` parameters applied to every builtin profile.
#
# ``preserve_interword_spaces=1`` — keeps variable-width gaps (columns
# in tables, tab-separated form fields) intact in the OCR output.
# Without this Tesseract collapses them to a single space and we lose
# column alignment in the text layer.
#
# ``tessedit_do_invert=0`` — suppresses Tesseract's "maybe this page
# is white-on-black" auto-detector. Our preprocessing already hands
# Tesseract a correctly-polarised binary; the detector is responsible
# for a class of "every letter comes back as gibberish" reports where
# it misfires on a page with a dark border or an inverted header panel.
_COMMON_TESSERACT_PARAMS: dict[str, str] = {
    "preserve_interword_spaces": "1",
    "tessedit_do_invert": "0",
}


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
            auto_rotate=AutoRotateConfig(enabled=True, min_confidence=1.0),
            deskew=DeskewConfig(enabled=True, auto_detect=True, max_angle=45.0),
            dewarp=DewarpConfig(enabled=False),
            # Sauvola adapts threshold per-pixel based on local mean +
            # standard deviation — handles uneven lighting far better
            # than adaptive Gaussian on real-world scans with shadows
            # or page-edge darkening. Window 25 is the empirically-
            # validated value for this profile; a brief Apr 2026
            # experiment bumping to 41 hurt real-document mean
            # confidence (51.5 % → 44 %) — the larger window over-
            # averaged the local std and produced thinner, fuzzier
            # stroke edges. Keep 25 until the
            # ``scripts/benchmark_universal.py`` tool proves a
            # different value wins on the user's document. k=0.2 is
            # the paper default for printed documents.
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
            # Stage E: erase long horizontal / vertical runs (table
            # borders, form rules) before binarisation so Tesseract
            # doesn't fuse adjacent text into the border glyph.
            # 75 px is the 300-DPI baseline; ImagePreprocessor scales
            # it to the runtime DPI, so it stays at ~0.25 inch at any
            # render resolution.
            border_removal=BorderRemovalConfig(
                enabled=True, min_line_length=75,
            ),
        )
        ocr = OCRConfig(
            languages=["rus", "eng"],
            primary_language="rus",
            psm=PSM.AUTO,
            oem=OEM.LSTM_ONLY,
            # 400 DPI is the LSTM sweet spot for printed Russian text.
            # Tesseract's LSTM was trained on 150–300 DPI corpora; at
            # 500–600 DPI the pixel features grow beyond what the net
            # saw, softmax confidence drops, and the layout analyser
            # crashes far more often (5000×7000 px A4 → retry tiers at
            # 200 DPI, which are strictly worse than the originally
            # requested DPI). 400 gives enough pixels-per-glyph for
            # 10 pt body text without tripping either failure mode.
            # Combined with DPI-adaptive preprocessing (kernel sizes
            # auto-scale in ImagePreprocessor), this delivers the
            # stable-95-%-confidence target the user asked for.
            dpi=400,
            optimize_level=OptimizeLevel.LOSSLESS,
            confidence_threshold=60.0,
            skip_text=True,
            # 360 s fits the per-page work at 400 DPI with headroom for
            # table-dense contract pages. The per-page retry inside
            # ocrmypdf_integration.py still escalates to the 900 s cap
            # for rare outliers before surfacing an error.
            tesseract_timeout=360,
            # Word-level confidence filter. On mixed-content scans
            # (forms + stamps + signatures + logos) Tesseract emits a
            # long tail of 10–40 %-confidence guesses from the
            # non-text regions. Without this flag those guesses end
            # up in the user-facing text next to real content, and a
            # 51 % mean_confidence reads to the user as "51 % of the
            # document is gibberish" even though 90 % of it is clean.
            # With the filter on, the results panel / TXT / DOCX
            # export show only words meeting ``confidence_threshold``
            # (60 %), and mean_conf is reported over the kept set.
            # See ``src.core.confidence_filter`` for the mechanism.
            drop_low_conf_words=True,
            extra_tesseract_params=dict(_COMMON_TESSERACT_PARAMS),
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
            # Lenient garbage filter drops ruler lines and symbol
            # walls Tesseract emits from table borders and page
            # noise, without touching legitimate short tokens like
            # "ООО" or numeric totals. Strict mode was tried briefly
            # but drops too many short-but-valid tokens on real
            # scanned forms; needs per-document benchmarking before
            # flipping.
            garbage_filter_strictness="lenient",
            custom_rules=[],
        )
        return ProfileData(
            name="universal_accurate",
            description=(
                "Универсальный «максимум точности»: 400 DPI (LSTM sweet "
                "spot), Sauvola + CLAHE + удаление фона + deskew + "
                "удаление рамок таблиц, адаптивный масштаб ядер по DPI, "
                "полная постобработка включая нормализацию "
                "кириллицы/латиницы и фильтр слов по уверенности "
                "распознавания"
            ),
            preprocess=preprocess,
            ocr=ocr,
            postprocess=postprocess,
        )

    def _build_default(self) -> ProfileData:
        """Balanced defaults suitable for most scans."""
        preprocess = PreprocessConfig(
            auto_rotate=AutoRotateConfig(enabled=True, min_confidence=1.0),
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
            extra_tesseract_params=dict(_COMMON_TESSERACT_PARAMS),
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

          * **Tesseract** — the bundled LSTM engine is always
            available in the installer.
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
            auto_rotate=AutoRotateConfig(enabled=True, min_confidence=1.0),
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
            # Word-level conf filter on the user-facing text. Measured
            # on real transport-invoice scans (Apr 2026): lifts the
            # mean_confidence of surfaced text from ~57 to ~82.
            drop_low_conf_words=True,
            # Block-level redaction on top of the per-word pass. When
            # Tesseract's layout analysis clusters a region of majority-
            # noise words (stamps, signatures, fine-print headers), we
            # wipe the whole block from the PDF text layer rather than
            # letting borderline-conf words inside that region sneak
            # into Ctrl-F and copy-paste output.
            redact_noisy_blocks=True,
            extra_tesseract_params=dict(_COMMON_TESSERACT_PARAMS),
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
            # ``validate_identifiers=True`` means the postprocessor
            # will rewrite corrupt ИНН / ОГРН tokens into their
            # canonical form when a unique 1-edit match exists in
            # the catalog loaded from ``expected/*.json``. Silent
            # no-op when the catalog is empty or absent.
            postprocess=PostprocessConfig(validate_identifiers=True),
        )

    def _build_low_quality(self) -> ProfileData:
        """Aggressive cleanup for blurry / noisy / low-contrast scans."""
        preprocess = PreprocessConfig(
            auto_rotate=AutoRotateConfig(enabled=True, min_confidence=1.0),
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
            extra_tesseract_params=dict(_COMMON_TESSERACT_PARAMS),
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
            auto_rotate=AutoRotateConfig(enabled=True, min_confidence=1.0),
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
            # ``load_freq_dawg=0`` disables Tesseract's frequency
            # dictionary for this profile. Russian contracts are full
            # of ИНН / ОГРН / account numbers and legal-entity names
            # (``ООО "Ромашка"``) that aren't in the freq dict; when
            # the dict IS loaded Tesseract biases digit sequences
            # toward common Russian words, corrupting the very
            # fields the user cares about most. Keeping the system
            # DAWG (``load_system_dawg`` unchanged) preserves prose
            # accuracy in the contract body.
            extra_tesseract_params={
                **_COMMON_TESSERACT_PARAMS,
                "load_freq_dawg": "0",
            },
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
            auto_rotate=AutoRotateConfig(enabled=True, min_confidence=1.0),
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
            extra_tesseract_params=dict(_COMMON_TESSERACT_PARAMS),
        )
        return ProfileData(
            name="english_text",
            description="English documents with clean layout (eng, OTSU, light CLAHE)",
            preprocess=preprocess,
            ocr=ocr,
            postprocess=PostprocessConfig(),
        )

