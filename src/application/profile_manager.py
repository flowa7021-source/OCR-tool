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
    ExtractConfig,
    LlmFallbackConfig,
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
    OptimizeLevel,
)

logger = logging.getLogger(__name__)


# Один-единственный builtin-профиль. С декабря 2026 «default»,
# «quick_reliable», «low_quality_scan», «contracts_ru», «english_text»
# и «tn_upd» удалены — пользователь попросил собрать «лучшее со всех»
# в единый ``universal_accurate``. Конкретно перенесено:
#   * Sauvola + CLAHE + deskew + border_removal + 400 DPI + полный
#     post-processing (universal_accurate base);
#   * ``load_freq_dawg=0`` + ``validate_identifiers=True`` (contracts_ru,
#     tn_upd, quick_reliable — лучше для ИНН/КПП/ОГРН);
#   * ``skip_text=True`` + ``extract.enabled=True`` + ``kind="tn_upd"`` +
#     ``multi_document=True`` (tn_upd — парсер ТН/УПД встроен).
# Пользовательские профили (``builtin=False``) продолжают работать
# через ProfileStorage. Документ-специфичные настройки теперь
# конфигурируются duplicate'ом + ручной правкой.
BUILTIN_NAMES: tuple[str, ...] = ("universal_accurate",)


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
        """Create the manager. Default selection — единственный
        builtin-профиль ``universal_accurate``."""
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
                "Current profile %s missing; falling back to 'universal_accurate'",
                self._current_name,
            )
            self._current_name = "universal_accurate"
            return self.storage.load("universal_accurate")

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
        """Re-seed the single builtin profile ``universal_accurate`` on disk.

        **Ключевое отличие от версии до декабря 2026:** builtin-профиль
        пере-сидируется на каждом запуске — даже если файл уже есть.
        Это гарантирует, что пользователь всегда получает канонические
        «best-of-all» настройки после апгрейда (Sauvola, validate_*,
        extract.kind=tn_upd, ...), без ручного удаления старого JSON.

        Пользовательские правки нужно вести в КОПИЯХ (``duplicate``):
        profile с ``builtin=False`` этим методом не трогается. Именно
        это отражает ``builtin=True`` — «этот профиль принадлежит
        приложению и обновляется вместе с ним».

        Удалённые builtin'ы (``default``, ``quick_reliable``,
        ``low_quality_scan``, ``contracts_ru``, ``english_text``,
        ``tn_upd``) остаются на диске у апгрейднувшихся пользователей,
        но больше не пере-сидятся. При желании их можно удалить через
        ``ProfileManager.delete(name)``.
        """
        builders = {"universal_accurate": self._build_universal_accurate}
        for name, builder in builders.items():
            profile = builder()
            profile.builtin = True
            try:
                existing = self.storage.load(name)
            except FileNotFoundError:
                existing = None
            # Если пользователь конвертировал builtin в свой (builtin=False
            # через duplicate + rename), НЕ переписываем — уважаем
            # кастомизацию. В остальных случаях (файла нет / builtin=True)
            # перезаписываем каноническими настройками.
            if existing is None or existing.builtin:
                self.storage.save(profile)
                logger.info(
                    "Re-seeded builtin profile %s (merged best-of-all settings).",
                    profile.name,
                )
            else:
                logger.debug(
                    "Profile %s is user-customised (builtin=False); skipping re-seed.",
                    name,
                )

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
            # Sauvola (re-enabled after real-document benchmark).
            # The first Apr 2026 benchmark round ran on synthetic
            # clean-paper fixtures where OTSU's global threshold
            # hits 0 % CER; that measurement led us to switch off
            # Sauvola after its diacritic artefacts on thin strokes
            # regressed the clean-text numbers. A follow-up round
            # on the user's ACTUAL transport-invoice corpus
            # (``inputs/TN_k_UPD_36_ot_02.09.2022.pdf``, 4 pages)
            # reversed the finding:
            #
            #     method                avg_conf  time
            #     OTSU                    85.9 %  442 s
            #     NONE (Tesseract own)    85.9 %  470 s
            #     SAUVOLA                 88.5 %  423 s  ← winner
            #     ADAPTIVE_GAUSSIAN       88.2 %  407 s
            #
            # Sauvola's +2.6-point lift over OTSU on real scanned
            # invoices comes from its locally-adaptive threshold —
            # scanner-lamp gradients and stamp-overlay shadows
            # break OTSU's single global cutoff, while Sauvola
            # computes per-window mean + std so each image region
            # gets its own threshold. The synthetic regression
            # (clean-paper text) wasn't representative of the
            # real workload. Keep Sauvola here; users who want
            # the old behaviour can pick ``default`` (OTSU).
            binarization=BinarizationConfig(
                method=BinarizationMethod.SAUVOLA,
                sauvola_window=25,
                sauvola_k=0.2,
            ),
            denoise=DenoiseConfig(
                enabled=True,
                steps=[
                    DenoiseStep(method=DenoiseMethod.MEDIAN, ksize=3),
                ],
            ),
            # CLAHE clip 2.0 (reverted from 3.0). The 3.0 bump pushed
            # the same Apr 2026 benchmark into the diacritic-artifact
            # regime: high clip + adaptive binarisation + background
            # division amplified subpixel noise into fake glyphs.
            # 2.0 is the conservative value that behaved correctly
            # across every profile we benchmarked.
            contrast=ContrastConfig(
                clahe_enabled=True, clahe_clip=2.0, clahe_tile=8
            ),
            # Background removal OFF (reverted from on). Measurement
            # showed it added ~5 % CER on clean synthetic scans
            # without any compensating gain on the noisy ones
            # (``low_quality_scan`` already has its own
            # blur_kernel=55 background path for those). Keep the
            # code path intact so users with dark-gradient phone
            # snaps can toggle it on via the UI — just don't default
            # it to true for the universal preset.
            background=BackgroundConfig(enabled=False, blur_kernel=55),
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
            # Soft-rescue band [45, 60) for lexically-clean tokens:
            # Tesseract underweights confidence on short digit runs
            # (ИНН, суммы, даты) and all-caps Cyrillic acronyms. A
            # straight 60%-cut discards real content along with the
            # stamp noise; soft-rescue keeps only the shape-credible
            # borderline tokens (single-script letters len≥3 OR
            # digit/separator tokens). Mixed-script tokens like
            # ``нe``, ``Taw`` and anything below 45% still drop, so
            # the stamp / signature garbage stays out. Layers on top
            # of the CAPS-company preservation already baked into
            # ``confidence_filter._should_keep_despite_low_conf``.
            soft_rescue_dropped_words=True,
            # Adaptive threshold: clean pages (mean ≥ 90%) lower the
            # bar to 40 to keep borderline-but-correct words; noisy
            # pages (mean < 70%) raise the bar to 70 to filter harder.
            # Turns the nominal ``confidence_threshold`` into a
            # reasonable default instead of a per-document tuning knob.
            adaptive_confidence_threshold=True,
            # Per-word script re-OCR for mixed tokens. Catches the
            # "ИНV-12345" class of errors where a single glyph went
            # Latin when the line went Cyrillic (or vice versa).
            # Adds ~10-20 % per-page latency on Russian documents
            # with Latin islands; zero cost when there are no mixed
            # tokens.
            per_word_script_disambiguation=True,
            # Per-word image rescue for borderline-conf tokens —
            # CLAHE + sharpen first (cheap), upscale second (for
            # tiny-glyph text the CLAHE pass couldn't lift). Both
            # only fire on words in the 30-70 conf band so rescues
            # don't churn already-correct text.
            per_word_clahe_rescue=True,
            per_word_upscale_rescue=True,
            # Dictionary-backed fuzzy rescue. For each borderline-
            # conf token (30-75 band), find the closest entry in
            # ``user-words.rus`` within edit distance 1 (short) or
            # 2 (≥ 6 chars) and swap in the canonical spelling.
            # Covers the failure mode where the crop was readable
            # but the LSTM's DAWG bias at primary OCR time was too
            # soft (``ИНЦ`` → ``ИНН``, ``Скаnia`` → ``Scania``).
            user_words_fuzzy_rescue=True,
            # Re-OCR fragmented layout blocks with PSM=SINGLE_BLOCK
            # when the primary AUTO pass collapsed their per-word
            # confidence below 60 %. Targets invoice / transport-doc
            # tables — the user's corpus is ТН / УПД forms with
            # grid layouts where AUTO regularly mis-segments.
            per_block_psm_retry=True,
            extra_tesseract_params={
                **_COMMON_TESSERACT_PARAMS,
                # ``load_freq_dawg=0`` — отключаем частотный
                # словарь Tesseract. ИНН / КПП / ОГРН / суммы —
                # длинные цифровые последовательности; freq-DAWG
                # систематически биасит цифры в сторону похожих
                # русских слов («7707820890» → «777..820...»).
                # Перенесено из удалённых contracts_ru / tn_upd —
                # обе профиля доказали на проде что DAWG для
                # документов с ИД-полями вреден.
                "load_freq_dawg": "0",
            },
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
            # Surface ``⟨рукописный текст⟩`` markers for blocks
            # Tesseract can't read reliably (handwritten regions
            # the Russian LSTM wasn't trained on). The word-conf
            # filter would otherwise drop them silently, hiding
            # from the user that there was content to type
            # manually. 40 % mean-conf + 3-word minimum gates
            # keep the marker off legitimate faded-but-printed
            # blocks.
            mark_suspect_handwritten_blocks=True,
            # ``validate_identifiers=True`` — переписываем
            # повреждённые ИНН / ОГРН в их каноническую форму,
            # когда уникальное 1-edit-совпадение есть в каталоге
            # ``expected/*.json``. Перенесено из tn_upd /
            # quick_reliable — рабочий способ восстановить
            # 1-2-символьные OCR-опечатки в ИД-полях ДО парсера.
            validate_identifiers=True,
            # Normalise dates / amounts / phones to the canonical
            # Russian business-document shapes (``DD.MM.YYYY``,
            # ``1 234,56``, ``+7 (XXX) XXX-XX-XX``). Repairs
            # common letter-digit OCR errors (``12.O1.2O23`` →
            # ``12.01.2023``) and makes downstream accounting
            # imports byte-stable across OCR runs.
            validate_entities=True,
            custom_rules=[],
        )
        # ``extract.enabled=True``, ``kind="tn_upd"`` — встроенный
        # пост-OCR парсер транспортных накладных и УПД. Перенесено
        # из удалённого tn_upd профиля. Multi-document on:
        # сводные УПД часто содержат несколько ТН в одном PDF.
        # LLM-fallback по умолчанию выключен — offline-first;
        # включается через preferences (ANTHROPIC_API_KEY).
        # Если документ — НЕ ТН/УПД, ``low_text_threshold=200`` и
        # парсер просто вернёт пустые rows; orchestrator это видит
        # и пропускает (см. tn_orchestrator.extract_from_pages).
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
                "Универсальный builtin максимальной точности (декабрь 2026): "
                "собрал best-of-all из удалённых default / quick_reliable / "
                "low_quality_scan / contracts_ru / english_text / tn_upd. "
                "400 DPI (LSTM sweet spot), Sauvola + мягкий CLAHE + "
                "deskew + удаление рамок таблиц, полный postprocessing с "
                "validate_identifiers / validate_entities, "
                "load_freq_dawg=0 для ИНН/КПП/ОГРН, skip_text для уже "
                "OCR'нутых PDF, встроенный парсер ТН/УПД (extract.kind=tn_upd) "
                "с multi-document support."
            ),
            preprocess=preprocess,
            ocr=ocr,
            postprocess=postprocess,
            extract=extract,
        )

