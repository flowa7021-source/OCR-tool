"""Real end-to-end test: actual Tesseract + OCRmyPDF + Ghostscript.

Unlike ``test_e2e.py`` (which stubs the OCR engine) and
``test_pipeline_mocked.py`` (which stubs just the engine but runs
real preprocessing), this module exercises the **entire** stack:

    OCRJobConfig
        → OCRPipeline.run
            → fitz.open + analyze_pdf             (real PyMuPDF)
            → ImagePreprocessor                   (real OpenCV)
            → temp PDF assembly                   (real PyMuPDF)
            → TesseractEngine.run
                → run_ocrmypdf                    (real ocrmypdf.ocr)
                    → Tesseract binary            (subprocess)
                    → Ghostscript                 (subprocess)
            → _extract_and_postprocess            (real fitz + postprocess)
        → JobResult (COMPLETED)
            → searchable PDF with a real invisible text layer

Skipped unless ``tesseract`` AND ``gs`` are actually installed on
``PATH``. Local developers on Windows/macOS can pull them in via
Chocolatey / Homebrew; the CI runners have them pre-installed.

The PDF we feed in is rendered from a known string at high contrast
so Tesseract genuinely recognises it. We then assert:

  1. The pipeline reaches ``JobStatus.COMPLETED``.
  2. The output searchable PDF exists on disk.
  3. PyMuPDF can extract the recognised text from it (so a text
     layer was actually stamped — this is the single biggest
     functional assertion of the whole suite).
  4. The recognised text contains the source string (Tesseract isn't
     byte-perfect on low-res synthetic renders, so we tolerate OCR
     noise and assert a stable substring).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# Hard dependencies for a "real" run. importorskip is a soft guard —
# if any of these are missing we'd just produce false negatives.
pytest.importorskip("fitz")
pytest.importorskip("cv2")
pytest.importorskip("ocrmypdf")
pytest.importorskip("pytesseract")

# External subprocess tools — skip cleanly if the host doesn't have them.
# On Windows, Artifex's Ghostscript distribution ships ``gswin64c.exe`` /
# ``gswin32c.exe`` — never a plain ``gs.exe`` — so ``shutil.which("gs")``
# returns None on every Windows host even when Ghostscript is correctly
# installed. Probing both POSIX and Windows names matches the production
# logic in ``src.infrastructure.external_tools`` and keeps the whole
# real-OCR suite from being silently skipped on the Windows CI leg.
def _find_ghostscript_binary() -> str | None:
    for name in ("gs", "gswin64c", "gswin32c"):
        p = shutil.which(name)
        if p:
            return p
    return None


_REAL_OCR_AVAILABLE = bool(shutil.which("tesseract")) and bool(
    _find_ghostscript_binary()
)
pytestmark = [
    pytest.mark.skipif(
        not _REAL_OCR_AVAILABLE,
        reason=(
            "Real OCR E2E requires tesseract + ghostscript on PATH "
            "(gs on POSIX, gswin64c/gswin32c on Windows)"
        ),
    ),
    # Bypass autouse pre-flight mock: we want the REAL
    # verify_required_for_ocrmypdf to confirm tesseract/gs discovery.
    pytest.mark.exercise_preflight,
]


_CYRILLIC_CAPABLE_FONTS: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Ubuntu / Debian
    "/usr/share/fonts/TTF/DejaVuSans.ttf",               # Arch
    "/Library/Fonts/Arial Unicode.ttf",                  # macOS
    "C:/Windows/Fonts/arial.ttf",                        # Windows
    "C:/Windows/Fonts/segoeui.ttf",                      # Windows fallback
)


def _find_cyrillic_font() -> Path | None:
    """Return a system TTF path that supports Cyrillic, or None.

    PyMuPDF's built-in ``helv`` is Latin-only — rendering ``ПРИВЕТ``
    through it produces unreadable glyphs that Tesseract can't OCR,
    masquerading as a pipeline bug. Tests that render Cyrillic text
    should look up a system Cyrillic-capable TTF via this helper and
    skip gracefully when none is available.
    """
    for candidate in _CYRILLIC_CAPABLE_FONTS:
        p = Path(candidate)
        if p.is_file():
            return p
    return None


def _render_text_pdf(path: Path, text: str, page_count: int = 1) -> Path:
    """Render ``text`` into a rasterised PDF that Tesseract can OCR.

    The pipeline rasterises pages via PyMuPDF and feeds PNGs to
    OCRmyPDF — so for a realistic test we need a scan-like input
    rather than text-selectable PDF. We render ``text`` at a large
    font size, print it to a page, then rasterise + re-embed as an
    image so there's no existing selectable text layer for OCRmyPDF
    to short-circuit on. This mimics what a user scanning a printed
    document would actually feed in.

    When ``text`` contains non-ASCII characters (e.g. Cyrillic), the
    built-in ``helv`` font produces unreadable glyphs. Auto-upgrade
    to a system Cyrillic-capable TTF in that case. Pure-ASCII text
    keeps the original ``helv`` path so the test works on minimal
    systems without DejaVu etc.
    """
    import fitz

    needs_unicode_font = any(ord(c) > 127 for c in text)
    font_file = _find_cyrillic_font() if needs_unicode_font else None
    if needs_unicode_font and font_file is None:
        raise RuntimeError(
            "Cyrillic/unicode text requested but no system Cyrillic font "
            "found. Install fonts-dejavu-core (Linux) or use ASCII-only "
            "text in the test."
        )

    # Step 1: write a plain text PDF.
    txt_doc = fitz.open()
    try:
        for i in range(page_count):
            page = txt_doc.new_page(width=612, height=792)  # US Letter
            # Big font so 150 DPI rasterisation captures sharp glyphs.
            kwargs: dict[str, object] = {
                "fontsize": 40,
            }
            if font_file is not None:
                kwargs["fontfile"] = str(font_file)
                kwargs["fontname"] = "UserUnicode"
            else:
                kwargs["fontname"] = "helv"
            page.insert_text(
                (72, 200),
                f"{text}\nPage {i + 1}",
                **kwargs,
            )
        raw = txt_doc.tobytes()
    finally:
        txt_doc.close()

    # Step 2: rasterise it to images and repack as an image-only PDF.
    scan_doc = fitz.open()
    src_doc = fitz.open(stream=raw, filetype="pdf")
    try:
        for src_page in src_doc:
            pix = src_page.get_pixmap(dpi=200, alpha=False)
            new = scan_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
            new.insert_image(new.rect, stream=pix.tobytes("png"))
        scan_doc.save(str(path))
    finally:
        scan_doc.close()
        src_doc.close()
    return path


def _tesseract_tessdata_dir() -> Path | None:
    """Locate tessdata — try several standard Linux locations + env var.

    ``TesseractWrapper.find_tessdata_dir`` expects to see either an
    app-bundled directory (absent in a pip-only dev install) or
    ``TESSDATA_PREFIX``. On CI we set the env var to the system-wide
    location. Returning ``None`` lets the caller decide whether to
    skip or warn.
    """
    import os

    env = os.environ.get("TESSDATA_PREFIX")
    if env and (Path(env) / "eng.traineddata").is_file():
        return Path(env)
    for cand in (
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tessdata",
        "/usr/local/share/tessdata",
        "/opt/homebrew/share/tessdata",
    ):
        p = Path(cand)
        if (p / "eng.traineddata").is_file():
            return p
    return None


@pytest.fixture
def _real_tesseract_wrapper(tmp_path: Path):
    """TesseractWrapper patched to use the system Tesseract, not ours.

    The production ``find_tesseract_binary`` prefers the bundled
    ``resources/tesseract/tesseract.exe`` first — absent in a
    pip-only dev checkout. We override its class-level caches so
    the wrapper points at whatever the system actually has.
    """
    from src.infrastructure.tesseract_wrapper import TesseractWrapper

    system_tesseract = shutil.which("tesseract")
    if not system_tesseract:
        pytest.skip("tesseract not on PATH")

    tessdata = _tesseract_tessdata_dir()
    if tessdata is None:
        pytest.skip("tessdata not found (missing eng.traineddata)")

    # Inject the real paths into the class-level cache.
    TesseractWrapper.reset()
    TesseractWrapper._binary_path = Path(system_tesseract)
    TesseractWrapper._tessdata_path = tessdata
    TesseractWrapper._configured = False
    wrapper = TesseractWrapper()
    wrapper.configure_pytesseract()
    yield wrapper
    TesseractWrapper.reset()


@pytest.fixture
def _fast_profile():
    """Profile tuned to keep the E2E short without gutting accuracy."""
    from src.core.models import (
        OCRConfig,
        PostprocessConfig,
        PreprocessConfig,
        ProfileData,
    )
    from src.shared.types import (
        OEM,
        PSM,
        BinarizationMethod,
        OCREngineKind,
        OptimizeLevel,
    )

    pre = PreprocessConfig()
    # Keep preprocessing minimal; we're testing orchestration + OCR
    # wiring, not preprocessor correctness (that's covered elsewhere).
    pre.binarization.method = BinarizationMethod.OTSU
    pre.deskew.enabled = False

    ocr = OCRConfig(
        engine=OCREngineKind.TESSERACT,
        # Russian+English for the assert below, but also so we
        # exercise the multi-language code path.
        languages=["eng"],
        primary_language="eng",
        psm=PSM.AUTO,
        oem=OEM.LSTM_ONLY,
        dpi=200,
        # optimize=0 — skip Ghostscript's multi-pass optimize stage
        # to keep the test under ~30 s. optimize>=1 is what the user
        # profiles default to, but it's not what we're testing here.
        optimize_level=OptimizeLevel.NONE,
        skip_text=False,
        confidence_threshold=50.0,
        tesseract_timeout=60,
    )
    post = PostprocessConfig(
        autocorrect_russian=False,
        autocorrect_english=False,
        merge_hyphenated=False,
        normalize_whitespace=True,
        normalize_unicode=True,
    )
    return ProfileData(
        name="e2e-real-ocr",
        ocr=ocr,
        preprocess=pre,
        postprocess=post,
    )


class TestRealOCREndToEnd:
    """The one test that actually proves OCR works."""

    def test_pipeline_runs_real_tesseract_and_emits_searchable_pdf(
        self,
        tmp_path: Path,
        _real_tesseract_wrapper,
        _fast_profile,
    ) -> None:
        """Full real-OCR pipeline on a synthetic scan of known English text."""
        from src.application.engines.registry import reset_cache
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import OCRJobConfig
        from src.core.text_postprocessor import TextPostprocessor
        from src.shared.types import JobStatus

        reset_cache()

        # The string we expect Tesseract to recognise. All-caps + all-
        # ASCII so the assertion is robust to minor OCR noise like
        # "WORLD" -> "WORLD," etc.
        known_text = "HELLO WORLD OCR"
        input_pdf = _render_text_pdf(
            tmp_path / "scan.pdf", text=known_text, page_count=1
        )
        output_pdf = tmp_path / "out" / "scan_ocr.pdf"

        progress_events: list[tuple[str, int, int]] = []

        def on_progress(current: int, total: int, stage: str) -> None:
            progress_events.append((stage, current, total))

        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=_real_tesseract_wrapper,
            progress_callback=on_progress,
            compute_confidence=False,
        )
        job = OCRJobConfig(
            input_path=str(input_pdf),
            output_path=str(output_pdf),
            profile=_fast_profile,
        )

        result = pipeline.run(job)

        # 1. Pipeline reached COMPLETED.
        assert result.status is JobStatus.COMPLETED, (
            f"Job FAILED: error={result.error!r}\n"
            f"Stages reached: {[s for s,_,_ in progress_events]}"
        )
        assert result.error is None

        # 2. Output PDF exists and is non-trivial in size (a valid
        #    searchable PDF from OCRmyPDF is usually >5 KB).
        assert output_pdf.exists(), "OCRmyPDF did not produce output PDF"
        assert output_pdf.stat().st_size > 1000, (
            f"output PDF suspiciously small: {output_pdf.stat().st_size} bytes"
        )

        # 3. A real text layer was stamped — extract via fitz and check.
        import fitz

        with fitz.open(str(output_pdf)) as doc:
            assert doc.page_count == 1
            page_text = doc.load_page(0).get_text("text") or ""

        # Tesseract isn't perfect on synthetic renders. Tolerate case /
        # extra whitespace and assert at least one expected word survives.
        norm = page_text.upper().replace("\n", " ")
        assert any(word in norm for word in known_text.split()), (
            f"None of {known_text.split()!r} found in extracted OCR "
            f"text.\nRaw extracted: {page_text!r}"
        )

        # 4. Progress events covered every pipeline stage.
        stages_seen = {s for s, _, _ in progress_events}
        assert {"analyze", "preprocess", "assemble", "ocr", "postprocess"} <= stages_seen, (
            f"Missing stages: {{analyze, preprocess, assemble, ocr, postprocess}} - "
            f"{stages_seen}"
        )

        # 5. JobResult.pages reflects actual OCR output.
        assert len(result.pages) == 1
        # The text we extracted from the PDF should match what the
        # JobResult's per-page text says (or be a superset of it —
        # postprocess may normalise whitespace).
        stored_upper = result.pages[0].text.upper()
        assert any(word in stored_upper for word in known_text.split())

    def test_multi_page_real_ocr(
        self,
        tmp_path: Path,
        _real_tesseract_wrapper,
        _fast_profile,
    ) -> None:
        """Two pages → two entries in JobResult.pages, both searchable."""
        from src.application.engines.registry import reset_cache
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import OCRJobConfig
        from src.core.text_postprocessor import TextPostprocessor
        from src.shared.types import JobStatus

        reset_cache()

        known_text = "ALPHA BRAVO CHARLIE"
        input_pdf = _render_text_pdf(
            tmp_path / "multi.pdf", text=known_text, page_count=2
        )
        output_pdf = tmp_path / "multi_ocr.pdf"

        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=_real_tesseract_wrapper,
            compute_confidence=False,
        )
        result = pipeline.run(
            OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(output_pdf),
                profile=_fast_profile,
            )
        )

        assert result.status is JobStatus.COMPLETED, result.error
        assert len(result.pages) == 2
        import fitz

        with fitz.open(str(output_pdf)) as doc:
            assert doc.page_count == 2
            for i in range(2):
                page_text = doc.load_page(i).get_text("text").upper()
                assert any(w in page_text for w in known_text.split()), (
                    f"Page {i + 1} text has no expected word: "
                    f"{page_text!r}"
                )


class TestRealOCRFailures:
    """Failure surfaces: corrupt input, empty PDF, missing binary."""

    def test_corrupt_pdf_surfaces_as_failed_job(
        self, tmp_path: Path, _real_tesseract_wrapper, _fast_profile
    ) -> None:
        """A garbage input must yield FAILED, not crash the worker."""
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import OCRJobConfig
        from src.core.text_postprocessor import TextPostprocessor
        from src.shared.types import JobStatus

        bad = tmp_path / "not_a_pdf.pdf"
        bad.write_bytes(b"\x00\x00\x00 definitely not a PDF")

        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=_real_tesseract_wrapper,
            compute_confidence=False,
        )
        result = pipeline.run(
            OCRJobConfig(
                input_path=str(bad),
                output_path=str(tmp_path / "out.pdf"),
                profile=_fast_profile,
            )
        )
        assert result.status is JobStatus.FAILED
        assert result.error
        assert not (tmp_path / "out.pdf").exists()


class TestRealOCROnUnicodePath:
    """Regression: the original user bug. End-to-end on a cyrillic path."""

    def test_full_pipeline_with_cyrillic_tmp_dir(
        self,
        tmp_path: Path,
        _real_tesseract_wrapper,
        _fast_profile,
    ) -> None:
        """Output under ``Т.Н. 020/...`` must survive the full OCR run.

        This is the *integration* version of test_save_png's unit
        regression. If ``cv2.imwrite`` ever gets re-introduced into
        the preprocess path, THIS is the test that catches it end to
        end, even on systems where the Windows fopen-ANSI bug doesn't
        trigger (like Linux) — because it's real I/O through the
        whole stack.
        """
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import OCRJobConfig
        from src.core.text_postprocessor import TextPostprocessor
        from src.shared.types import JobStatus

        unicode_dir = tmp_path / "Т.Н. 020" / "Рабочий стол"
        unicode_dir.mkdir(parents=True)

        known_text = "UNICODE PATH TEST"
        input_pdf = _render_text_pdf(
            unicode_dir / "скан.pdf", text=known_text
        )
        output_pdf = unicode_dir / "скан_ocr.pdf"

        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=_real_tesseract_wrapper,
            compute_confidence=False,
        )
        result = pipeline.run(
            OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(output_pdf),
                profile=_fast_profile,
            )
        )

        assert result.status is JobStatus.COMPLETED, (
            f"Unicode-path E2E FAILED: {result.error!r}"
        )
        assert output_pdf.exists()
        assert output_pdf.stat().st_size > 1000


# ---------------------------------------------------------------------------
# Russian language + autocorrect post-processing in a single real-OCR run.
# Closes the gap between the existing real-OCR tests (English only,
# postprocess flags all OFF) and the user-facing default profile, which
# ships with rus+eng + autocorrect_russian=True. A regression where the
# postprocess chain was silently disabled for the real-Tesseract code
# path would slip past every other E2E test in this module.
# ---------------------------------------------------------------------------


class TestRealOCRWithRussianAutocorrect:
    """Real Tesseract on Russian text, with autocorrect_russian=True."""

    def test_russian_text_is_recognised_and_postprocessed(
        self, tmp_path: Path, _real_tesseract_wrapper
    ) -> None:
        """Render Cyrillic text → real OCR → Russian autocorrect → readable text.

        Skips cleanly if ``rus.traineddata`` is missing on the host
        (some CI minimal-Tesseract installs only ship ``eng``).

        We deliberately render *clean* Russian (no digit-as-letter
        misreads), so the autocorrect rules should be idempotent — but
        the postprocess chain still has to execute through every step
        (Unicode NFC, hyphen merge, whitespace normalisation, regex
        rule application). A bug that crashed *any* of them on
        non-empty Cyrillic would flip the job to FAILED.
        """
        from src.application.engines.registry import reset_cache
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import (
            OCRConfig,
            OCRJobConfig,
            PostprocessConfig,
            PreprocessConfig,
            ProfileData,
        )
        from src.core.text_postprocessor import TextPostprocessor
        from src.shared.types import (
            OEM,
            PSM,
            BinarizationMethod,
            JobStatus,
            OCREngineKind,
            OptimizeLevel,
        )

        # Hard skip if the Russian model is missing — without it,
        # Tesseract returns gibberish for Cyrillic glyphs and the
        # assertion below would be a false negative.
        tessdata = _tesseract_tessdata_dir()
        if tessdata is None or not (tessdata / "rus.traineddata").is_file():
            pytest.skip("rus.traineddata not found — install tesseract-ocr-rus")

        # Same story for a Cyrillic-capable system font: without one,
        # ``_render_text_pdf`` would draw garbled glyphs through the
        # Latin-only ``helv``, not a pipeline bug. Skip cleanly on
        # minimal CI images that don't carry DejaVu / Arial Unicode.
        if _find_cyrillic_font() is None:
            pytest.skip(
                "No Cyrillic-capable system font found — install "
                "fonts-dejavu-core or equivalent"
            )

        reset_cache()

        # Short, distinctive Cyrillic word that Tesseract handles well
        # at 200 DPI. Avoids look-alikes (no Е/E, no О/O) so the
        # post-processor's autocorrect rules genuinely do nothing.
        known_text = "ПРИВЕТ"
        input_pdf = _render_text_pdf(
            tmp_path / "ru.pdf", text=known_text, page_count=1
        )
        output_pdf = tmp_path / "ru_ocr.pdf"

        pre = PreprocessConfig()
        pre.binarization.method = BinarizationMethod.OTSU
        pre.deskew.enabled = False
        ocr = OCRConfig(
            engine=OCREngineKind.TESSERACT,
            languages=["rus", "eng"],
            primary_language="rus",
            psm=PSM.AUTO,
            oem=OEM.LSTM_ONLY,
            dpi=200,
            optimize_level=OptimizeLevel.NONE,
            skip_text=False,
            tesseract_timeout=60,
        )
        # Every postprocess flag the default profile ships with — so
        # this test exercises the *exact* user-facing chain.
        post = PostprocessConfig(
            autocorrect_russian=True,
            autocorrect_english=True,
            merge_hyphenated=True,
            normalize_whitespace=True,
            normalize_unicode=True,
            remove_artifacts=True,
        )
        profile = ProfileData(
            name="e2e-real-ru-autocorrect",
            ocr=ocr,
            preprocess=pre,
            postprocess=post,
        )

        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=_real_tesseract_wrapper,
            compute_confidence=False,
        )
        result = pipeline.run(
            OCRJobConfig(
                input_path=str(input_pdf),
                output_path=str(output_pdf),
                profile=profile,
            )
        )

        assert result.status is JobStatus.COMPLETED, (
            f"rus+autocorrect E2E FAILED: {result.error!r}"
        )
        assert output_pdf.exists()
        assert len(result.pages) == 1

        recognised = result.pages[0].text
        # Tesseract isn't byte-perfect; tolerate noise but require at
        # least one of the expected tri-grams to survive both OCR and
        # the full post-process chain. A regression that disabled
        # postprocess on the real path would still pass this — a
        # regression that crashed it would not.
        norm = recognised.upper()
        assert any(stem in norm for stem in ("ПРИ", "ИВЕ", "ВЕТ")), (
            f"None of expected tri-grams found in recognised text — "
            f"either OCR or postprocess silently dropped Cyrillic "
            f"content. Got: {recognised!r}"
        )

        # Postprocess invariant: NFC normalisation must have run, so
        # the recognised text must equal its own NFC form (no NFD
        # decompositions sneaking through).
        import unicodedata

        assert recognised == unicodedata.normalize("NFC", recognised), (
            "normalize_unicode=True did not run on the real-OCR path; "
            "recognised text contains non-NFC sequences."
        )


# ---------------------------------------------------------------------------
# Smoke test: does the packaged external_tools registry actually resolve
# Tesseract / Ghostscript on THIS runner? Doesn't hit the pipeline —
# just checks our discovery layer against the real system.
# ---------------------------------------------------------------------------


class TestExternalToolsOnRealHost:
    def test_verify_required_finds_system_tools(self) -> None:
        """On a host with tesseract + gs on PATH, no required tool is missing.

        This is the integration counterpart to the unit test with a
        fake registry. It confirms our fallback (``shutil.which`` in
        ``locate()``) correctly detects system-installed binaries
        when the bundle dir is absent — which is exactly the dev /
        CI environment.

        Runs on both POSIX and Windows: ``locate()`` probes platform-
        specific candidates (``gs`` vs ``gswin64c.exe``), so as long
        as the CI workflow's install step put the binaries on PATH
        this test exercises the discovery layer identically on both
        operating systems.
        """
        from src.infrastructure.external_tools import (
            verify_required_for_ocrmypdf,
        )

        missing = verify_required_for_ocrmypdf()
        assert missing == [], (
            f"Required external tools missing on CI host: {missing}. "
            "Install tesseract and/or ghostscript via apt/brew/choco."
        )
