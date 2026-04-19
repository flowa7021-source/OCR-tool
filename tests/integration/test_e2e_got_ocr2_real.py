"""End-to-end test for the REAL GOT-OCR 2.0 engine.

The existing ``test_e2e_got_ocr2.py`` stubs out torch / transformers
so it can run on any CI runner. That catches wiring regressions
(profile → engine dispatch, text extraction, postprocess routing)
but cannot catch:

  * Bundle defects — missing ``torchvision/_C.*.pyd``, corrupt
    ``model.safetensors``, HuggingFace ``trust_remote_code``
    scripts pulling in a package the installer dropped.
  * Inference regressions — a refactor of the engine's
    ``_recognize_page`` that subtly breaks tokenization or
    positional encoding.
  * Preprocess / postprocess chain incompatibilities — GOT
    takes a different input shape than Tesseract, and the
    ``handwritten_mixed`` profile exists precisely to tune the
    preprocess for it. Silent breakage here (e.g. a binarisation
    step left on) would halve accuracy.

This module is the ONLY place that runs actual torch inference
against the bundled weights. It's skipped by default (the fixture
probes both the HTR deps AND the 580 MB model weights being on
disk) so it's zero-cost in environments where either is missing.

Gated for the nightly real-OCR workflow — the one place in CI
that pays the ~30-60 s/page CPU inference cost.
"""

from __future__ import annotations

import pytest

from tests.integration._real_ocr_helpers import requires_real_got_ocr2

# ``exercise_preflight`` tells conftest's autouse bypass that we want
# the real ``verify_required_for_ocrmypdf`` to run — we DO care about
# tesseract / gs being discoverable for the preprocess + assembly
# stages that come BEFORE the GOT-OCR engine call.
pytestmark = [requires_real_got_ocr2, pytest.mark.exercise_preflight]


@pytest.fixture(scope="module")
def _got_engine():
    """Construct + load the GOT-OCR engine once per test module.

    The 580 MB ``model.safetensors`` mmap + transformers init takes
    ~10-20 s on CPU; doing it per-test would add a minute to every
    new test case. Module-scope caches the loaded engine and
    unloads it on teardown so leftover RAM doesn't leak into the
    next integration test module.
    """
    from src.application.engines.got_ocr_engine import GOTOCREngine

    eng = GOTOCREngine()
    ok, msg = eng.is_available()
    assert ok, f"GOT-OCR should be available here but got: {msg}"
    # Pre-load so the first recognise_page call isn't cold-starting.
    eng._load_model()
    yield eng
    eng.unload()


# ---------------------------------------------------------------------------
# Real model: load + single-page inference + handwritten_mixed profile
# ---------------------------------------------------------------------------


class TestRealGOTOCR2:
    """Verify the bundled GOT-OCR 2.0 weights actually infer."""

    def test_model_load_succeeds(self, _got_engine) -> None:
        """If ``_got_engine`` fixture constructed, the model loaded.
        This assertion is redundant with the fixture's own check but
        makes the regression the user would see explicit — if this
        test shows up as FAILED in CI output, the bundled model is
        broken."""
        assert _got_engine._model is not None
        assert _got_engine._tokenizer is not None
        assert _got_engine._device in ("cpu", "cuda")

    def test_cpu_inference_produces_non_empty_text(
        self, _got_engine, tmp_path,
    ) -> None:
        """Render a simple Russian text PDF, run it through the
        REAL GOT-OCR 2.0 engine, assert SOME text comes back.

        Deliberately NOT asserting specific content — a transformer
        on 12pt synthetic text might hallucinate occasional
        wrong characters. The regression-signal we care about is
        "inference ran and produced text", NOT accuracy (which is
        what the accuracy benchmark is for and is separately
        measurable).
        """
        import fitz

        from src.core.models import OCRConfig
        from src.shared.types import OCREngineKind

        # Minimal 1-page image-only PDF containing "ТЕКСТ".
        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"

        doc = fitz.open()
        try:
            page = doc.new_page(width=612, height=792)
            page.insert_text(
                (100, 200), "ТЕКСТ ОБРАЗЕЦ",
                fontsize=40, fontname="helv",
            )
            doc.save(str(inp))
        finally:
            doc.close()

        cfg = OCRConfig(
            engine=OCREngineKind.GOT_OCR2,
            languages=["rus", "eng"],
            primary_language="rus",
            dpi=300,
        )
        results = _got_engine.run(
            preprocessed_pdf=inp, output_pdf=out, config=cfg,
        )

        assert len(results) == 1
        text = (results[0].text or "").strip()
        assert text, (
            "GOT-OCR 2.0 returned empty text on a 1-page synthetic "
            "Russian input. Either the bundled weights are corrupt "
            "or a recent refactor broke ``_recognize_page``."
        )
        # Also verify a searchable PDF was emitted.
        assert out.exists()
        assert out.stat().st_size > 1024

    def test_handwritten_mixed_profile_end_to_end(
        self, _got_engine, tmp_path,
    ) -> None:
        """Full pipeline (preprocess + GOT-OCR + postprocess) driven
        by the ``handwritten_mixed`` profile. This catches two
        classes of wiring bug:

          1. Preprocess chain incompatible with transformer input
             (e.g. binarisation left on in the profile — GOT-OCR
             expects greyscale).
          2. Postprocess flags wired to GOT output fail (e.g.
             ``validate_identifiers`` raising because catalog
             loading misfired in the worker).
        """
        import fitz

        from src.application.pipeline import OCRPipeline
        from src.application.profile_manager import ProfileManager
        from src.core.doc_catalog import load_default_catalog
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import OCRJobConfig
        from src.core.text_postprocessor import TextPostprocessor
        from src.infrastructure.config_storage import ProfileStorage
        from src.infrastructure.tesseract_wrapper import TesseractWrapper
        from src.shared.types import JobStatus

        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = manager.load("handwritten_mixed")

        inp = tmp_path / "in.pdf"
        out = tmp_path / "out.pdf"

        doc = fitz.open()
        try:
            page = doc.new_page(width=612, height=792)
            page.insert_text(
                (100, 200), "ТЕСТ 7813266190",
                fontsize=40, fontname="helv",
            )
            doc.save(str(inp))
        finally:
            doc.close()

        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(catalog=load_default_catalog()),
            tesseract=TesseractWrapper(),
            compute_confidence=False,
        )
        result = pipeline.run(
            OCRJobConfig(
                input_path=str(inp),
                output_path=str(out),
                profile=profile,
            )
        )

        assert result.status is JobStatus.COMPLETED, result.error
        assert result.pages, "No pages in JobResult"
        assert result.pages[0].text, (
            "handwritten_mixed → GOT-OCR path returned empty text"
        )
