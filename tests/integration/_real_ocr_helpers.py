"""Shared helpers for real-OCR integration tests.

Every test under ``tests/integration/test_e2e_*.py`` that runs the
full pipeline with live Tesseract + Ghostscript imports from here.
Keeping the generators + fixtures in one place means:

* The same "render a scan-like PDF" invariant is used everywhere —
  a change to how we simulate skew / noise / contrast reshapes every
  test that depends on it.
* Platform-specific skip logic (tesseract binary, tessdata language
  files, Cyrillic-capable system font) is written once. Any new test
  that wants real Cyrillic OCR just uses ``requires_real_russian_ocr``.

Leading underscore in the filename keeps pytest from trying to
collect this module as a test.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Environment probes (tesseract, ghostscript, tessdata, fonts)
# ---------------------------------------------------------------------------


_CYRILLIC_FONTS: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",     # Debian/Ubuntu
    "/usr/share/fonts/TTF/DejaVuSans.ttf",                 # Arch
    "/Library/Fonts/Arial Unicode.ttf",                    # macOS (pre-Catalina)
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", # macOS (post-Catalina)
    "C:/Windows/Fonts/arial.ttf",                          # Windows
    "C:/Windows/Fonts/segoeui.ttf",                        # Windows fallback
)


def find_cyrillic_font() -> Path | None:
    """Return a system TTF path that supports Cyrillic, or None.

    PyMuPDF's built-in ``helv`` is Latin-only — rendering Cyrillic
    through it produces unreadable glyphs that look like a pipeline
    bug but are actually a font-coverage bug in the test harness.
    Callers that need Cyrillic text should look up a Unicode-capable
    TTF via this helper and skip gracefully when none is available.
    """
    for candidate in _CYRILLIC_FONTS:
        p = Path(candidate)
        if p.is_file():
            return p
    return None


def tesseract_tessdata_dir() -> Path | None:
    """Locate a tessdata directory that at least contains eng.traineddata.

    Looks at standard POSIX locations + the TESSDATA_PREFIX env var.
    Returns None if nothing usable is found — tests using this should
    ``pytest.skip`` rather than fail.
    """
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


def _find_ghostscript() -> str | None:
    """Return a Ghostscript executable path, or None.

    The console binary is called ``gs`` on POSIX but ``gswin64c.exe`` /
    ``gswin32c.exe`` on Windows — Artifex's Windows build never ships a
    ``gs.exe``. ``shutil.which("gs")`` therefore returns ``None`` on
    every Windows host with a correctly-installed Ghostscript, and the
    ``REAL_OCR_AVAILABLE`` probe below would silently skip every
    real-OCR test under ``tests/integration/``. Scan the platform-
    specific names so Windows CI actually runs the suite.
    """
    for name in ("gs", "gswin64c", "gswin32c"):
        path = shutil.which(name)
        if path:
            return path
    return None


REAL_OCR_AVAILABLE: bool = bool(
    shutil.which("tesseract") and _find_ghostscript()
)
"""``True`` when the host has both Tesseract and Ghostscript on PATH."""


#: Import-time flag: do we have rus.traineddata available?
_td = tesseract_tessdata_dir()
RUSSIAN_OCR_AVAILABLE: bool = bool(
    _td is not None
    and (_td / "rus.traineddata").is_file()
    and find_cyrillic_font() is not None
)
"""``True`` when both rus.traineddata AND a Cyrillic-capable system font
are available. Tests rendering Russian text must skip when this is
False — rendering through a Latin-only font produces glyphs Tesseract
can't read, and that's the test harness's fault not the pipeline's."""


# pytest decorators callers can slap on any test
requires_real_ocr = pytest.mark.skipif(
    not REAL_OCR_AVAILABLE,
    reason="Tesseract + Ghostscript must be on PATH (apt install "
    "tesseract-ocr ghostscript; choco install tesseract)",
)
requires_real_russian_ocr = pytest.mark.skipif(
    not RUSSIAN_OCR_AVAILABLE,
    reason="Real Russian OCR needs both rus.traineddata AND a "
    "Cyrillic-capable system font (apt install tesseract-ocr-rus "
    "fonts-dejavu-core)",
)


def _got_ocr2_available() -> tuple[bool, str]:
    """Return ``(is_available, reason_if_not)`` for the GOT-OCR 2.0 path.

    End-to-end GOT-OCR tests need EVERY piece of the stack present on
    the host. Probe in order from cheapest to heaviest so the reason
    string reports the FIRST missing dep — most actionable for
    whoever's reading the skip message in CI logs.
    """
    for dep in (
        "torch", "transformers", "einops", "accelerate",
        "torchvision", "verovio", "tiktoken", "safetensors",
    ):
        try:
            __import__(dep)
        except ImportError:
            return False, f"Python package '{dep}' not installed"

    try:
        from src.infrastructure.model_manager import GOT_OCR2_SPEC, ModelManager
    except ImportError as exc:
        return False, f"ModelManager import failed: {exc}"
    mgr = ModelManager()
    if not mgr.is_available(GOT_OCR2_SPEC.model_id):
        return False, (
            "GOT-OCR 2.0 model weights missing — run "
            ".github/scripts/download_got_ocr2.py"
        )
    return True, ""


_GOT_OK, _GOT_REASON = _got_ocr2_available()

#: Import-time flag: is every piece of the GOT-OCR 2.0 stack installed?
REAL_GOT_OCR2_AVAILABLE: bool = _GOT_OK
"""``True`` when the HTR deps AND the 580 MB model weights are
both present on the host. End-to-end GOT-OCR tests depend on this.
A False value carries the reason in :data:`_GOT_REASON` for the
skip-message."""

requires_real_got_ocr2 = pytest.mark.skipif(
    not REAL_GOT_OCR2_AVAILABLE,
    reason=(
        f"Real GOT-OCR 2.0 unavailable: {_GOT_REASON}. "
        "Install via `pip install -e '.[htr]'` and run "
        "`python .github/scripts/download_got_ocr2.py`."
    ),
)


# ---------------------------------------------------------------------------
# Tesseract wrapper fixture — points at the SYSTEM tesseract, not our bundle
# ---------------------------------------------------------------------------


@pytest.fixture
def real_tesseract_wrapper() -> Any:
    """TesseractWrapper patched to use the system Tesseract, not the bundle.

    Our production ``find_tesseract_binary`` prefers
    ``resources/tesseract/tesseract.exe`` (PyInstaller bundle) first —
    absent in a pip-only dev checkout. This fixture overrides the
    class-level caches so the wrapper points at whatever the system
    actually has. Every real-OCR test uses this fixture, not
    ``TesseractWrapper()`` directly.
    """
    from src.infrastructure.tesseract_wrapper import TesseractWrapper

    system_tesseract = shutil.which("tesseract")
    if not system_tesseract:
        pytest.skip("tesseract not on PATH")
    tessdata = tesseract_tessdata_dir()
    if tessdata is None:
        pytest.skip("tessdata not found (missing eng.traineddata)")

    TesseractWrapper.reset()
    TesseractWrapper._binary_path = Path(system_tesseract)
    TesseractWrapper._tessdata_path = tessdata
    TesseractWrapper._configured = False
    wrapper = TesseractWrapper()
    wrapper.configure_pytesseract()
    yield wrapper
    TesseractWrapper.reset()


# ---------------------------------------------------------------------------
# Clean-cache fixture: ocr_cache HITs would otherwise short-circuit the
# second real-OCR run with the same (input bytes, profile) pair
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_ocr_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect the persistent OCR cache to a per-test tmp dir.

    Without this, re-running a test (or two tests with identical
    fixture output) hits ``OCR cache HIT`` and skips the pipeline
    we wanted to exercise. Pointing the cache at ``tmp_path`` keeps
    each test hermetic.
    """
    cache_dir = tmp_path / "ocr-cache"
    cache_dir.mkdir()
    monkeypatch.setattr(
        "src.infrastructure.ocr_cache.OCR_CACHE_DIR", cache_dir
    )
    return cache_dir


# ---------------------------------------------------------------------------
# PDF generation — baseline "scan-like" rasterised PDF
# ---------------------------------------------------------------------------


def _rasterise_as_image_only_pdf(
    raw_text_pdf: bytes, dest: Path, dpi: int = 200
) -> Path:
    """Rasterise every page of ``raw_text_pdf`` and re-embed as images.

    This is the simulation of a scan: no selectable text layer in the
    resulting PDF, just raster pages. Real-OCR tests need this because
    OCRmyPDF's ``skip_text=True`` would otherwise short-circuit on a
    text-selectable input and never call Tesseract at all.
    """
    import fitz

    scan_doc = fitz.open()
    src_doc = fitz.open(stream=raw_text_pdf, filetype="pdf")
    try:
        for src_page in src_doc:
            pix = src_page.get_pixmap(dpi=dpi, alpha=False)
            new = scan_doc.new_page(
                width=src_page.rect.width, height=src_page.rect.height
            )
            new.insert_image(new.rect, stream=pix.tobytes("png"))
        scan_doc.save(str(dest))
    finally:
        scan_doc.close()
        src_doc.close()
    return dest


def _text_pdf_bytes(
    text: str, pages: int = 1, fontsize: int = 40, cyrillic: bool = False
) -> bytes:
    """Build a selectable-text PDF with ``text`` (+ page number) per page.

    Returns raw bytes so callers can choose where to put them (pass
    through ``_rasterise_as_image_only_pdf`` or apply image
    manipulations first).
    """
    import fitz

    font_file: Path | None = None
    if cyrillic:
        font_file = find_cyrillic_font()
        if font_file is None:
            raise RuntimeError(
                "Cyrillic text requested but no system font with "
                "Cyrillic coverage found"
            )

    doc = fitz.open()
    try:
        for i in range(pages):
            page = doc.new_page(width=612, height=792)  # US Letter
            kwargs: dict[str, Any] = {"fontsize": fontsize}
            if font_file is not None:
                kwargs["fontfile"] = str(font_file)
                kwargs["fontname"] = "Unicode"
            else:
                kwargs["fontname"] = "helv"
            page.insert_text(
                (72, 200), f"{text}\nPage {i + 1}", **kwargs
            )
        return doc.tobytes()
    finally:
        doc.close()


def render_clean_text_pdf(
    path: Path,
    text: str,
    pages: int = 1,
    cyrillic: bool = False,
    dpi: int = 200,
    fontsize: int = 40,
) -> Path:
    """Baseline: render ``text`` to a rasterised, scan-like PDF.

    The rendered image is crisp (no skew, no noise, no contrast issues).
    Tests for individual preprocessing steps start from this baseline
    and add a single challenge (rotation, noise, etc.) at a time.
    """
    raw = _text_pdf_bytes(text, pages=pages, fontsize=fontsize, cyrillic=cyrillic)
    return _rasterise_as_image_only_pdf(raw, path, dpi=dpi)


def render_rotated_text_pdf(
    path: Path,
    text: str,
    angle_degrees: float,
    cyrillic: bool = False,
    dpi: int = 200,
) -> Path:
    """Rasterise text into a PDF rotated by ``angle_degrees``.

    Simulates a page that was scanned slightly crooked. Deskew tests
    use ~5°; any more and Tesseract's own OSD may auto-correct before
    the pipeline's deskew kicks in, muddying the test signal.
    """
    import cv2
    import fitz

    raw = _text_pdf_bytes(text, cyrillic=cyrillic)
    src = fitz.open(stream=raw, filetype="pdf")
    try:
        pix = src.load_page(0).get_pixmap(dpi=dpi, alpha=False)
        # RGB bytes → np array → rotate → encode back to PNG.
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        h, w = arr.shape[:2]
        center = (w / 2, h / 2)
        matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        rotated = cv2.warpAffine(
            arr, matrix, (w, h),
            borderValue=(255, 255, 255),
            flags=cv2.INTER_LINEAR,
        )
        ok, buf = cv2.imencode(".png", rotated)
        assert ok
        png_bytes = buf.tobytes()
    finally:
        src.close()

    out = fitz.open()
    try:
        page = out.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=png_bytes)
        out.save(str(path))
    finally:
        out.close()
    return path


def render_noisy_text_pdf(
    path: Path,
    text: str,
    salt_pepper_ratio: float = 0.03,
    cyrillic: bool = False,
    dpi: int = 200,
) -> Path:
    """Add salt-and-pepper noise to a rasterised text image.

    Each pixel has ``salt_pepper_ratio`` probability of being flipped
    to pure black or pure white — simulates photocopy speckle or
    degraded scan quality.
    """
    import cv2
    import fitz

    raw = _text_pdf_bytes(text, cyrillic=cyrillic)
    src = fitz.open(stream=raw, filetype="pdf")
    try:
        pix = src.load_page(0).get_pixmap(dpi=dpi, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    finally:
        src.close()

    rng = np.random.default_rng(seed=42)  # deterministic noise
    noise_mask = rng.random(arr.shape[:2])
    arr[noise_mask < salt_pepper_ratio / 2] = 0
    arr[noise_mask > 1 - salt_pepper_ratio / 2] = 255

    ok, buf = cv2.imencode(".png", arr)
    assert ok
    out = fitz.open()
    try:
        h, w = arr.shape[:2]
        page = out.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=buf.tobytes())
        out.save(str(path))
    finally:
        out.close()
    return path


def render_low_contrast_text_pdf(
    path: Path,
    text: str,
    foreground_level: int = 128,
    background_level: int = 210,
    cyrillic: bool = False,
    dpi: int = 200,
) -> Path:
    """Render text with reduced foreground/background contrast.

    Real-world equivalents: faded receipts, dim photocopies, pages
    scanned with too-low exposure. CLAHE preprocessing should lift
    the contrast enough for Tesseract to recognise the glyphs.
    """
    import cv2
    import fitz

    raw = _text_pdf_bytes(text, cyrillic=cyrillic)
    src = fitz.open(stream=raw, filetype="pdf")
    try:
        pix = src.load_page(0).get_pixmap(dpi=dpi, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    finally:
        src.close()

    # Map [0, 255] to [foreground_level, background_level]. Dark text
    # (previously 0) becomes ``foreground_level``; white background
    # (previously 255) becomes ``background_level``.
    span = background_level - foreground_level
    arr = (foreground_level + arr.astype(np.float32) / 255.0 * span).astype(
        np.uint8
    )

    ok, buf = cv2.imencode(".png", arr)
    assert ok
    out = fitz.open()
    try:
        h, w = arr.shape[:2]
        page = out.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=buf.tobytes())
        out.save(str(path))
    finally:
        out.close()
    return path


# ---------------------------------------------------------------------------
# Pipeline runner — stamps out the boilerplate every real-OCR test writes
# ---------------------------------------------------------------------------


def make_realistic_profile(
    *,
    name: str = "e2e-real",
    languages: list[str] | None = None,
    binarization: str = "otsu",
    deskew: bool = False,
    denoise_steps: list[tuple[str, dict[str, int | float]]] | None = None,
    clahe: bool = False,
    background_removal: bool = False,
    dpi: int = 200,
    optimize_level: int = 0,
    autocorrect_russian: bool = False,
    autocorrect_english: bool = False,
    merge_hyphenated: bool = False,
    normalize_whitespace: bool = True,
    normalize_unicode: bool = True,
    remove_artifacts: bool = False,
    custom_rules: list[dict[str, Any]] | None = None,
) -> Any:
    """Build a :class:`ProfileData` with sensible real-OCR defaults.

    Most test knobs default to OFF so each test can turn ON exactly
    the one preprocess/postprocess step it wants to exercise — the
    rest should not interfere.
    """
    from src.core.models import (
        BackgroundConfig,
        BinarizationConfig,
        ContrastConfig,
        DenoiseConfig,
        DenoiseStep,
        DeskewConfig,
        OCRConfig,
        PostprocessConfig,
        PreprocessConfig,
        ProfileData,
        RegexRule,
    )
    from src.shared.types import (
        OEM,
        PSM,
        BinarizationMethod,
        DenoiseMethod,
        OCREngineKind,
        OptimizeLevel,
    )

    pre = PreprocessConfig(
        deskew=DeskewConfig(enabled=deskew, auto_detect=True, max_angle=45.0),
        binarization=BinarizationConfig(method=BinarizationMethod(binarization)),
        denoise=DenoiseConfig(
            enabled=bool(denoise_steps),
            steps=[
                DenoiseStep(method=DenoiseMethod(m), **params)
                for (m, params) in (denoise_steps or [])
            ],
        ),
        contrast=ContrastConfig(clahe_enabled=clahe),
        background=BackgroundConfig(enabled=background_removal),
    )
    ocr = OCRConfig(
        engine=OCREngineKind.TESSERACT,
        languages=languages or ["eng"],
        primary_language=(languages or ["eng"])[0],
        psm=PSM.AUTO,
        oem=OEM.LSTM_ONLY,
        dpi=dpi,
        optimize_level=OptimizeLevel(optimize_level),
        skip_text=False,
        tesseract_timeout=120,
    )
    post = PostprocessConfig(
        autocorrect_russian=autocorrect_russian,
        autocorrect_english=autocorrect_english,
        merge_hyphenated=merge_hyphenated,
        normalize_whitespace=normalize_whitespace,
        normalize_unicode=normalize_unicode,
        remove_artifacts=remove_artifacts,
        custom_rules=[
            RegexRule(**r) for r in (custom_rules or [])
        ],
    )
    return ProfileData(name=name, ocr=ocr, preprocess=pre, postprocess=post)


RunPipeline = Callable[[Path, Path, Any], Any]
"""Type alias: ``run_pipeline(input_pdf, output_pdf, profile) -> JobResult``."""


def run_pipeline(
    input_pdf: Path, output_pdf: Path, profile: Any, wrapper: Any
) -> Any:
    """Invoke the full pipeline with the given profile + wrapper.

    Returns the :class:`JobResult`. Callers still need to assert on
    ``.status``, ``.pages``, etc. — this just centralises the boilerplate
    of constructing preprocessor/postprocessor + OCRPipeline.
    """
    from src.application.engines.registry import reset_cache
    from src.application.pipeline import OCRPipeline
    from src.core.image_preprocessor import ImagePreprocessor
    from src.core.models import OCRJobConfig
    from src.core.text_postprocessor import TextPostprocessor

    reset_cache()
    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        tesseract=wrapper,
        compute_confidence=False,
    )
    return pipeline.run(
        OCRJobConfig(
            input_path=str(input_pdf),
            output_path=str(output_pdf),
            profile=profile,
        )
    )


def assert_ocr_recognised(result: Any, expected_fragments: list[str]) -> None:
    """Assert the recognised text contains at least one expected fragment.

    Tesseract isn't byte-perfect on synthetic renders. We tolerate
    noise but require ≥1 expected substring (case-insensitive) —
    that's enough signal that the OCR path ran to completion without
    silently returning empty.

    Raises:
        AssertionError: When neither the job completed nor a fragment
            was found. The message includes the raw recognised text
            to make diagnosis possible from a single failing test log.
    """
    from src.shared.types import JobStatus

    assert result.status is JobStatus.COMPLETED, (
        f"Job FAILED: {result.error!r}"
    )
    assert result.pages, "Job COMPLETED but no pages"
    text = result.pages[0].text or ""
    upper = text.upper()
    matches = [frag for frag in expected_fragments if frag.upper() in upper]
    assert matches, (
        f"No expected fragment found in OCR output. "
        f"Expected any of: {expected_fragments!r}. "
        f"Got: {text!r}"
    )
