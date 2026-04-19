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
            # 75 px at 500 DPI is the empirically-validated value; a
            # brief Apr 2026 experiment raising it to 125 hurt real-
            # document OCR (more table rules made it through to
            # Tesseract's segmentation → more ``|||`` / ``===`` noise
            # fragments in the output). Keep 75 until a benchmarked
            # change shows otherwise.
            border_removal=BorderRemovalConfig(
                enabled=True, min_line_length=75,
            ),
        )
        ocr = OCRConfig(
            languages=["rus", "eng"],
            primary_language="rus",
            psm=PSM.AUTO,
            oem=OEM.LSTM_ONLY,
            # 500 DPI is the sweet spot for the "maximum accuracy"
            # preset after the parallel-per-page engine lifted the
            # per-page timeout ceiling. 600 DPI was tried first and
            # still blows past 900 s on A4 Russian contracts
            # (5000×7000 pixels crashes Tesseract's layout analyser).
            # 400 worked but left ``ru_dense_small`` CER at ~25 % —
            # small 10pt body text genuinely needed more pixel density.
            # 500 DPI gives the LSTM 25 % more pixels per character
            # with ~1.56× image area vs 400; combined with the
            # raised timeout below, real-world contracts complete
            # without hitting retry tiers.
            dpi=500,
            optimize_level=OptimizeLevel.LOSSLESS,
            confidence_threshold=60.0,
            skip_text=True,
            # Timeout raised 300 → 450 s to match the ~1.56× per-page
            # work at 500 DPI. Still well under the per-page retry
            # escalation ceiling in ocrmypdf_integration.py
            # (``_MAX_RETRY_TESSERACT_TIMEOUT_SEC = 900``), so a rare
            # dense page that exceeds 450 s still gets one retry at
            # the 900 s cap before falling back to the simplified-
            # settings tier.
            tesseract_timeout=450,
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
                "Универсальный «максимум точности»: 500 DPI, Sauvola + "
                "CLAHE + удаление фона + deskew + удаление рамок таблиц, "
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
        """GOT-OCR 2.0 for scans with handwriting, stamps, and mixed fonts.

        When to use (vs. ``quick_reliable``):

          * Documents with **handwritten** notes, signatures, or
            stamp text — Tesseract's LSTM is printed-font-only and
            returns 5-25 % conf on handwriting. GOT-OCR 2.0 is a
            transformer trained on mixed print + handwriting; it
            routinely reads handwriting at 70-90 %.
          * **Non-standard fonts** — stylised invoice headers, fancy
            company logos, decorative stamps. Tesseract guesses at
            10-40 %; GOT-OCR handles them directly.
          * **Acceptable trade-offs**: ~580 MB model download, CPU
            inference 3-5× slower than Tesseract at 300 DPI, GPU
            (when available) comparable-to-faster.

        When NOT to use:

          * Pure printed-text scans at 300 DPI or better —
            Tesseract via ``quick_reliable`` is faster and the
            accuracy delta is not worth the wall-time cost.
          * When the bundled model file isn't downloaded yet — the
            first run surfaces a clear "model missing" error and
            bails. Use the in-app "Download GOT-OCR 2.0" action
            from the engine menu.

        Preprocessing is deliberately MINIMAL — transformer OCR
        prefers greyscale input over binarised. CLAHE stays on
        (cheap, lifts contrast on faded pages). Deskew stays on
        (angled input wastes the model's positional encoding).
        Everything else is off.

        Postprocess shares the ``quick_reliable`` identifier-fixup
        path: even though GOT-OCR reads ИНН / ОГРН more cleanly than
        Tesseract, a single-digit miss still happens on faded print,
        and the catalog-assisted rewrite fixes it transparently.
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
            # The word-level / block-level filters (Steps 1 + 2) are
            # Tesseract-only — they read the pytesseract TSV that
            # GOT-OCR doesn't produce. Leaving the flags off here
            # makes the intent explicit; the pipeline already gates
            # those steps on ``engine == TESSERACT``, so setting
            # them to True here would be a silent no-op anyway.
            drop_low_conf_words=False,
            redact_noisy_blocks=False,
        )
        # Postprocess IS engine-agnostic — it operates on the
        # extracted text after OCR is done, regardless of which
        # engine produced it. Sharing the ИНН/ОГРН catalog fixup
        # between quick_reliable and handwritten_mixed means a
        # corrupt ИНН in GOT-OCR output also gets rewritten to the
        # canonical form.
        postprocess = PostprocessConfig(validate_identifiers=True)
        return ProfileData(
            name="handwritten_mixed",
            description=(
                "GOT-OCR 2.0 для рукописного и печатного текста. "
                "Требует скачивания модели (~580 МБ) через меню "
                "«Движок OCR». Рекомендуется когда в документе "
                "есть рукописные заметки, штампы с текстом или "
                "нестандартные шрифты — Tesseract на них сдаётся. "
                "Для чистых печатных сканов используйте quick_reliable."
            ),
            preprocess=preprocess,
            ocr=ocr,
            postprocess=postprocess,
        )
