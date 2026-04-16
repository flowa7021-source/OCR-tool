"""E2E tests that exercise *settings variation* through the full pipeline.

The existing E2E suites (``test_e2e.py``, ``test_e2e_real_ocr.py``,
``test_pipeline_mocked.py``) all run with one hard-coded profile. They
prove the pipeline glue works once. They do **not** prove that:

  * switching ``BinarizationMethod`` actually changes the bitmap that
    reaches the OCR engine;
  * each individual ``PostprocessConfig`` flag (``merge_hyphenated``,
    ``normalize_unicode``, ``autocorrect_russian``, ``remove_artifacts``)
    actually transforms ``JobResult.pages[*].text`` end-to-end;
  * the bundled built-in profiles in ``profiles/*.json`` actually load
    via :class:`ProfileStorage` and run cleanly through the pipeline.

Those are the "I changed the setting in the UI but nothing happened"
bugs — invisible to the existing suites. This module closes that gap
with mocked OCR (fast, deterministic): the OCR engine is stubbed but
preprocessing, assemble, postprocess, and result extraction are all
real.

Cache hygiene: every test gets its own ``OCR_CACHE_DIR`` under
``tmp_path`` so cache hits cannot short-circuit a second run with the
same input bytes + profile.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fitz")
pytest.importorskip("cv2")

from src.application.engines.base import OCREngine, PageOCRResult  # noqa: E402
from src.application.engines.registry import reset_cache  # noqa: E402
from src.application.pipeline import OCRPipeline  # noqa: E402
from src.application.profile_manager import BUILTIN_NAMES  # noqa: E402
from src.core.image_preprocessor import ImagePreprocessor  # noqa: E402
from src.core.models import (  # noqa: E402
    OCRConfig,
    OCRJobConfig,
    PostprocessConfig,
    PreprocessConfig,
    ProfileData,
)
from src.core.text_postprocessor import TextPostprocessor  # noqa: E402
from src.infrastructure.config_storage import ProfileStorage  # noqa: E402
from src.infrastructure.tesseract_wrapper import TesseractWrapper  # noqa: E402
from src.shared.types import (  # noqa: E402
    BinarizationMethod,
    JobStatus,
    OCREngineKind,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_test_pdf(path: Path, page_count: int = 1) -> Path:
    """Synthesize a tiny multi-page PDF with PyMuPDF.

    Same shape as the one in ``test_pipeline_mocked.py`` — small enough
    that the parametrized tests all fit in a few seconds even on a
    cold CI runner.
    """
    import fitz

    doc = fitz.open()
    try:
        for i in range(page_count):
            page = doc.new_page(width=300, height=200)
            page.insert_text((50, 100), f"Page {i + 1} content")
        doc.save(str(path))
    finally:
        doc.close()
    return path


def _profile(
    *,
    binarization: BinarizationMethod = BinarizationMethod.NONE,
    deskew: bool = False,
    postprocess: PostprocessConfig | None = None,
    engine: OCREngineKind = OCREngineKind.TESSERACT,
) -> ProfileData:
    """Build a deterministic, fast profile with selectable knobs."""
    pre = PreprocessConfig()
    pre.binarization.method = binarization
    pre.deskew.enabled = deskew

    ocr = OCRConfig(engine=engine, dpi=150)
    return ProfileData(
        name=f"variation-{binarization.value}-{engine.value}",
        ocr=ocr,
        preprocess=pre,
        postprocess=postprocess if postprocess is not None else PostprocessConfig(),
    )


class _CapturingEngine(OCREngine):
    """Stub that copies the preprocessed PDF *and* records its bytes.

    The recorded bytes let us assert that two profiles with different
    preprocessing settings produced different bitmaps reaching OCR —
    closing the loop that "the user toggled binarization, and the
    bytes that reached Tesseract genuinely differed".

    The engine returns one ``PageOCRResult`` per page in the PDF, with
    ``text`` taken from ``page_texts`` (or a fallback). This mirrors
    GOT-OCR2's behaviour where the engine pre-fills the text and the
    pipeline then runs postprocess on it.
    """

    kind = OCREngineKind.TESSERACT

    def __init__(self, page_texts: list[str] | None = None) -> None:
        self.page_texts = page_texts
        self.captured_bytes: bytes | None = None
        self.run_called = 0

    @property
    def name(self) -> str:
        return "capturing-stub"

    @property
    def description(self) -> str:
        return "for tests"

    def is_available(self) -> tuple[bool, str]:
        return True, ""

    def run(self, preprocessed_pdf, output_pdf, config, progress_callback=None):
        self.run_called += 1
        # Snapshot what the pipeline actually built. This is the bitmap
        # (wrapped in a PDF) Tesseract would see — including every
        # preprocessing transformation the profile asked for.
        self.captured_bytes = Path(preprocessed_pdf).read_bytes()

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preprocessed_pdf, output_pdf)

        import fitz

        with fitz.open(str(output_pdf)) as doc:
            results = []
            for i in range(doc.page_count):
                if self.page_texts and i < len(self.page_texts):
                    text = self.page_texts[i]
                else:
                    text = f"stub page {i + 1}"
                results.append(
                    PageOCRResult(
                        page_number=i + 1,
                        text=text,
                        mean_confidence=85.0,
                    )
                )
        if progress_callback is not None:
            progress_callback(len(results), len(results), "ocr")
        return results


def _make_pipeline() -> OCRPipeline:
    return OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        tesseract=TesseractWrapper(),
        compute_confidence=False,
    )


@pytest.fixture(autouse=True)
def _reset_engines() -> None:
    reset_cache()
    yield
    reset_cache()


@pytest.fixture(autouse=True)
def _isolated_ocr_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect the persistent OCR cache to a per-test tmp directory.

    Without this, the second test that processes identical bytes with
    the same profile would *cache-hit* the first test's run and skip
    the part of the pipeline we want to assert on. Pointing the cache
    at ``tmp_path / ocr-cache`` keeps each test hermetic.
    """
    cache_dir = tmp_path / "ocr-cache"
    cache_dir.mkdir()
    monkeypatch.setattr(
        "src.infrastructure.ocr_cache.OCR_CACHE_DIR", cache_dir
    )
    return cache_dir


# ---------------------------------------------------------------------------
# 1. Binarization choice actually reaches the OCR engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        BinarizationMethod.NONE,
        BinarizationMethod.OTSU,
        BinarizationMethod.ADAPTIVE_GAUSSIAN,
        BinarizationMethod.ADAPTIVE_MEAN,
        BinarizationMethod.SAUVOLA,
    ],
)
def test_pipeline_completes_for_every_binarization_method(
    method: BinarizationMethod, tmp_path: Path
) -> None:
    """Each documented binarization method runs cleanly through the pipeline.

    Catches regressions where a new OpenCV / scikit-image release breaks
    one of the binarization branches in :class:`ImagePreprocessor` —
    the unit suite already covers each in isolation, this proves the
    *integration* (preprocess → assemble → engine handoff) survives.
    """
    input_pdf = _build_test_pdf(tmp_path / "input.pdf", page_count=1)
    output_pdf = tmp_path / "out.pdf"

    stub = _CapturingEngine()
    with patch("src.application.engines.get_engine", return_value=stub):
        result = _make_pipeline().run(
            OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(output_pdf),
                profile=_profile(binarization=method),
            )
        )

    assert result.status is JobStatus.COMPLETED, (
        f"binarization={method!r} failed: {result.error!r}"
    )
    assert stub.captured_bytes is not None
    # Sanity: the preprocessed PDF actually contains content (a 1-page
    # image-only PDF wrapping our preprocessed PNG is at least a few KB).
    assert len(stub.captured_bytes) > 500


def test_different_binarization_methods_produce_different_preprocessed_bitmaps(
    tmp_path: Path,
) -> None:
    """OTSU vs ADAPTIVE_GAUSSIAN vs SAUVOLA each yield distinct preprocessed PDFs.

    This is the assertion that closes "I changed binarization in the
    Settings dialog but nothing happened in the recognised text". If
    the pipeline were ignoring ``profile.preprocess.binarization.method``
    and always running the default, all three captured PDFs would be
    byte-identical.
    """
    input_pdf = _build_test_pdf(tmp_path / "input.pdf", page_count=1)
    output_pdf = tmp_path / "out.pdf"

    captured: dict[BinarizationMethod, bytes] = {}
    for method in (
        BinarizationMethod.OTSU,
        BinarizationMethod.ADAPTIVE_GAUSSIAN,
        BinarizationMethod.SAUVOLA,
    ):
        stub = _CapturingEngine()
        with patch("src.application.engines.get_engine", return_value=stub):
            result = _make_pipeline().run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_profile(binarization=method),
                )
            )
        assert result.status is JobStatus.COMPLETED, result.error
        assert stub.captured_bytes is not None
        captured[method] = stub.captured_bytes

    # Pairwise distinct: every pair must differ. PDFs that wrap
    # different bitmaps will differ in the embedded image stream
    # length, in the content-stream md5, and in the xref offsets — so
    # bytewise inequality is a robust proxy for "preprocessing
    # actually changed".
    pairs = [
        (BinarizationMethod.OTSU, BinarizationMethod.ADAPTIVE_GAUSSIAN),
        (BinarizationMethod.OTSU, BinarizationMethod.SAUVOLA),
        (BinarizationMethod.ADAPTIVE_GAUSSIAN, BinarizationMethod.SAUVOLA),
    ]
    for a, b in pairs:
        assert captured[a] != captured[b], (
            f"Preprocessed PDFs identical for {a.value} vs {b.value} — "
            "binarization setting is not reaching the engine."
        )


# ---------------------------------------------------------------------------
# 2. Each PostprocessConfig flag is honoured end-to-end
# ---------------------------------------------------------------------------


def _run_with_postprocess(
    tmp_path: Path,
    *,
    page_texts: list[str],
    postprocess: PostprocessConfig,
) -> str:
    """Run the pipeline with a stub engine returning ``page_texts``.

    Returns the postprocessed text for page 1 — what the user sees in
    the Results panel.
    """
    input_pdf = _build_test_pdf(tmp_path / "input.pdf", page_count=len(page_texts))
    output_pdf = tmp_path / "out.pdf"

    stub = _CapturingEngine(page_texts=page_texts)
    with patch("src.application.engines.get_engine", return_value=stub):
        result = _make_pipeline().run(
            OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(output_pdf),
                profile=_profile(postprocess=postprocess),
            )
        )
    assert result.status is JobStatus.COMPLETED, result.error
    assert result.pages, "no pages in JobResult"
    return result.pages[0].text


@pytest.mark.parametrize("merge_hyphenated", [True, False])
def test_merge_hyphenated_flag_changes_pipeline_output(
    merge_hyphenated: bool, tmp_path: Path
) -> None:
    """``merge_hyphenated`` toggle must actually affect ``JobResult.pages[].text``."""
    text = _run_with_postprocess(
        tmp_path,
        page_texts=["приме-\nры текста"],
        postprocess=PostprocessConfig(
            merge_hyphenated=merge_hyphenated,
            # Disable everything else so the assertion isolates this flag.
            autocorrect_russian=False,
            autocorrect_english=False,
            normalize_whitespace=False,
            normalize_unicode=False,
            remove_artifacts=False,
        ),
    )
    if merge_hyphenated:
        assert "примеры" in text, (
            f"merge_hyphenated=True did not glue 'приме-\\nры' → 'примеры'; "
            f"got: {text!r}"
        )
    else:
        # The hyphen + newline must survive when the flag is off.
        assert "приме-" in text and "\n" in text, (
            f"merge_hyphenated=False unexpectedly altered text: {text!r}"
        )


def test_normalize_unicode_flag_composes_nfd_to_nfc(tmp_path: Path) -> None:
    """NFD-form Cyrillic gets composed to NFC when the flag is on.

    ``Й`` can be encoded as a single codepoint U+0419 (NFC) or as
    ``И + combining breve`` (NFD). Tesseract has been seen to emit NFD
    on some platforms; if ``normalize_unicode`` were silently disabled,
    downstream text comparison and search-in-PDF would break.
    """
    nfd_text = "И\u0306"  # 'И' + combining breve = NFD form of 'Й'

    text_on = _run_with_postprocess(
        tmp_path,
        page_texts=[nfd_text],
        postprocess=PostprocessConfig(
            normalize_unicode=True,
            autocorrect_russian=False,
            autocorrect_english=False,
            merge_hyphenated=False,
            normalize_whitespace=False,
            remove_artifacts=False,
        ),
    )
    text_off = _run_with_postprocess(
        tmp_path,
        page_texts=[nfd_text],
        postprocess=PostprocessConfig(
            normalize_unicode=False,
            autocorrect_russian=False,
            autocorrect_english=False,
            merge_hyphenated=False,
            normalize_whitespace=False,
            remove_artifacts=False,
        ),
    )

    assert text_on == "Й", f"expected NFC 'Й', got {text_on!r}"
    # NFD must survive untouched when the flag is off.
    assert text_off == nfd_text, f"NFD text was mutated: {text_off!r}"


@pytest.mark.parametrize("autocorrect", [True, False])
def test_autocorrect_russian_flag_changes_pipeline_output(
    autocorrect: bool, tmp_path: Path
) -> None:
    """Digit-in-cyrillic-word fix only fires when the flag is on.

    The built-in rule rewrites ``0`` → ``о`` *only* when surrounded by
    Cyrillic letters (so legitimate numbers like ``2024`` stay intact).
    """
    # 'д0м' — Tesseract-style misread of 'дом' where 'о' came back as '0'.
    text = _run_with_postprocess(
        tmp_path,
        page_texts=["д0м и сад 2024"],
        postprocess=PostprocessConfig(
            autocorrect_russian=autocorrect,
            autocorrect_english=False,
            merge_hyphenated=False,
            normalize_whitespace=False,
            normalize_unicode=False,
            remove_artifacts=False,
        ),
    )
    if autocorrect:
        assert "дом" in text, (
            f"autocorrect_russian=True did not fix 'д0м' → 'дом'; got: {text!r}"
        )
        # The legitimate number 2024 must NOT be rewritten.
        assert "2024" in text, (
            f"autocorrect_russian=True clobbered the legitimate number "
            f"'2024'; got: {text!r}"
        )
    else:
        assert "д0м" in text, (
            f"autocorrect_russian=False unexpectedly corrected text: {text!r}"
        )


def test_remove_artifacts_flag_drops_pure_punctuation_lines(tmp_path: Path) -> None:
    """``remove_artifacts`` strips lines that contain no letters/digits."""
    page = "Реальная строка\n~~~~~\nЕщё одна строка\n|||\n"

    text_on = _run_with_postprocess(
        tmp_path,
        page_texts=[page],
        postprocess=PostprocessConfig(
            remove_artifacts=True,
            autocorrect_russian=False,
            autocorrect_english=False,
            merge_hyphenated=False,
            normalize_whitespace=False,
            normalize_unicode=False,
        ),
    )
    text_off = _run_with_postprocess(
        tmp_path,
        page_texts=[page],
        postprocess=PostprocessConfig(
            remove_artifacts=False,
            autocorrect_russian=False,
            autocorrect_english=False,
            merge_hyphenated=False,
            normalize_whitespace=False,
            normalize_unicode=False,
        ),
    )

    # Real content survives in both cases.
    assert "Реальная строка" in text_on
    assert "Ещё одна строка" in text_on
    # Artefact lines are gone when the flag is on.
    assert "~~~~~" not in text_on, f"artefact line not stripped: {text_on!r}"
    assert "|||" not in text_on, f"artefact line not stripped: {text_on!r}"
    # Artefact lines stay when the flag is off.
    assert "~~~~~" in text_off and "|||" in text_off, (
        f"remove_artifacts=False unexpectedly dropped lines: {text_off!r}"
    )


# ---------------------------------------------------------------------------
# 3. Bundled built-in profiles load via storage and run the pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile_name",
    # Skip universal_accurate (only built in code, no JSON in repo) and
    # handwritten_mixed (GOT-OCR2 engine — covered by the engine-dispatch
    # test in test_e2e.py). The five below ship as JSON in /profiles/.
    ["default", "quick_reliable", "low_quality_scan", "contracts_ru", "english_text"],
)
def test_bundled_builtin_profile_loads_and_runs_through_pipeline(
    profile_name: str, tmp_path: Path
) -> None:
    """Every bundled JSON profile loads via :class:`ProfileStorage` and runs.

    Catches regressions where a new field is added to ``OCRConfig``/
    ``PreprocessConfig`` but the bundled JSON wasn't migrated — the
    user's first launch with that profile would crash deep in
    ``ProfileData.from_dict``. By running each profile end-to-end we
    also prove the chosen settings combination doesn't blow up
    preprocessing (e.g. NLM denoise at 400 DPI in ``low_quality_scan``).
    """
    # Sanity: profile_name is in the master list — guards against a
    # bundled JSON disappearing without us noticing.
    assert profile_name in BUILTIN_NAMES, (
        f"{profile_name} is not in BUILTIN_NAMES — profile registry drift"
    )

    # Use an empty user profiles dir so ProfileStorage seeds from the
    # bundled JSONs in /profiles/. This is exactly the path the app
    # takes on a brand-new install.
    storage = ProfileStorage(profiles_dir=tmp_path / "user-profiles")
    profile = storage.load(profile_name)
    assert profile.name == profile_name
    assert profile.builtin is True

    input_pdf = _build_test_pdf(tmp_path / "input.pdf", page_count=1)
    output_pdf = tmp_path / "out.pdf"

    stub = _CapturingEngine()
    with patch("src.application.engines.get_engine", return_value=stub):
        result = _make_pipeline().run(
            OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(output_pdf),
                profile=profile,
            )
        )

    assert result.status is JobStatus.COMPLETED, (
        f"profile={profile_name!r} failed end-to-end: {result.error!r}"
    )
    assert output_pdf.exists()
    assert result.page_count == 1
    assert stub.run_called == 1


# ---------------------------------------------------------------------------
# 4. quick_reliable profile contract
# ---------------------------------------------------------------------------


class TestQuickReliableProfile:
    """The low-risk fallback profile must stay genuinely low-risk.

    ``quick_reliable`` is the profile users are told to switch to when
    ``universal_accurate`` fails. It MUST avoid every config choice that
    was in any of the production failure logs:

      * Tesseract engine (not GOT-OCR 2.0 — optional model, separate
        install failure mode);
      * DPI strictly below 600 (the DPI that produced the timeout-
        then-graft-crash chain);
      * tesseract_timeout at least 300s (matches the new default and
        leaves headroom for the auto-retry);
      * No dewarp / no background removal (heaviest optional steps).
    """

    def test_profile_loads_from_bundled_json(self, tmp_path: Path) -> None:
        """The new profile ships as a JSON under /profiles/ so existing
        user installs see it after upgrade without needing any code
        migration."""
        storage = ProfileStorage(profiles_dir=tmp_path / "user-profiles")
        profile = storage.load("quick_reliable")
        assert profile.name == "quick_reliable"
        assert profile.builtin is True

    def test_profile_uses_tesseract_not_got_ocr2(self, tmp_path: Path) -> None:
        storage = ProfileStorage(profiles_dir=tmp_path / "user-profiles")
        profile = storage.load("quick_reliable")
        assert profile.ocr.engine is OCREngineKind.TESSERACT, (
            "quick_reliable must use Tesseract — GOT-OCR 2.0 depends on "
            "a separately-downloaded model, which is exactly the failure "
            "mode this profile exists to route around."
        )

    def test_profile_uses_moderate_dpi_and_generous_timeout(
        self, tmp_path: Path
    ) -> None:
        storage = ProfileStorage(profiles_dir=tmp_path / "user-profiles")
        profile = storage.load("quick_reliable")
        assert profile.ocr.dpi < 600, (
            f"quick_reliable at DPI={profile.ocr.dpi} recreates the "
            "600 DPI timeout failure mode it's meant to avoid."
        )
        assert profile.ocr.tesseract_timeout >= 300, (
            f"quick_reliable at tesseract_timeout="
            f"{profile.ocr.tesseract_timeout}s is tighter than the new "
            "global default and leaves no room for the auto-retry."
        )

    def test_profile_disables_heavy_optional_steps(
        self, tmp_path: Path
    ) -> None:
        """Dewarp + background removal are the slowest optional steps;
        for a fallback profile we keep them off."""
        storage = ProfileStorage(profiles_dir=tmp_path / "user-profiles")
        profile = storage.load("quick_reliable")
        assert profile.preprocess.dewarp.enabled is False
        assert profile.preprocess.background.enabled is False


# ---------------------------------------------------------------------------
# 5. Pipeline preflight: fail fast when the engine is broken
# ---------------------------------------------------------------------------


class TestPipelinePreflight:
    """The pipeline must refuse obviously-broken configurations BEFORE
    doing any expensive work.

    In a pre-fix build, a user with a stale GOT-OCR 2.0 model spent
    ~25 seconds rasterising + preprocessing 4 pages before the engine
    load finally crashed with ``OSError``. Preflight now calls
    ``engine.is_available()`` up front — when False, the job returns
    FAILED within roughly a second, so the user can fix the config
    and re-run without waiting.
    """

    def _make_stub_unavailable_engine(self) -> OCREngine:
        class _Unavailable(OCREngine):
            kind = OCREngineKind.TESSERACT

            @property
            def name(self) -> str:
                return "unavailable-stub"

            @property
            def description(self) -> str:
                return "stub"

            def is_available(self) -> tuple[bool, str]:
                return False, (
                    "Файлы модели GOT-OCR 2.0 устарели — "
                    "откройте Настройки → Скачать модель."
                )

            def run(self, *a, **kw):  # pragma: no cover — preflight skips run
                raise AssertionError(
                    "run() must not be called when is_available() is False"
                )

        return _Unavailable()

    def test_preflight_rejects_unavailable_engine_before_preprocess(
        self, tmp_path: Path
    ) -> None:
        """FAILED + engine's is_available message surfaces verbatim, AND
        ``engine.run()`` is never reached."""
        input_pdf = _build_test_pdf(tmp_path / "input.pdf", page_count=2)
        output_pdf = tmp_path / "out.pdf"

        stub = self._make_stub_unavailable_engine()
        with patch("src.application.engines.get_engine", return_value=stub):
            result = _make_pipeline().run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_profile(),
                )
            )

        assert result.status is JobStatus.FAILED
        assert result.error is not None
        assert "Скачать модель" in result.error, (
            f"engine.is_available() message should be preserved verbatim; "
            f"got {result.error!r}"
        )
        # No output PDF produced — preflight stopped the job before the
        # assemble stage.
        assert not output_pdf.exists()

    def test_preflight_progress_event_fires(self, tmp_path: Path) -> None:
        """Before the first slow step, a ``preflight`` progress event
        must arrive so the UI can move the bar off 0%."""
        input_pdf = _build_test_pdf(tmp_path / "input.pdf", page_count=1)
        output_pdf = tmp_path / "out.pdf"

        events: list[tuple[int, int, str]] = []
        stub = _CapturingEngine()
        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=TesseractWrapper(),
            progress_callback=lambda c, t, s: events.append((c, t, s)),
            compute_confidence=False,
        )
        with patch("src.application.engines.get_engine", return_value=stub):
            pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_profile(),
                )
            )

        stages = [s for _, _, s in events]
        assert "preflight" in stages, (
            f"preflight must emit a progress event; saw stages={stages}"
        )
        # And it must be the FIRST stage so the UI shows responsiveness
        # before the expensive analyze/preprocess stages run.
        assert stages[0] == "preflight", (
            f"preflight must be first stage; saw {stages}"
        )
