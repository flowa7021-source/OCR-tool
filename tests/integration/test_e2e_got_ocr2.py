"""End-to-end tests for the GOT-OCR 2.0 engine path.

Unlike Tesseract (which we drive via OCRmyPDF on the host), GOT-OCR 2.0
is a transformer model loaded into the worker process via
``transformers.AutoModel.from_pretrained(..., trust_remote_code=True)``.
The ~580 MB model weights can't reasonably ship with a test suite,
so most tests here **stub torch + transformers + the model.chat
call** and verify the pipeline wires the engine up correctly end-to-end.

What these tests actually guard against:
  * A regression in engine-dispatch lets Tesseract run when the user
    picked GOT-OCR 2.0.
  * A regression in ``_extract_and_postprocess`` overwrites the text
    GOT-OCR2 returned (Tesseract-engine branch reads from the PDF;
    GOT-OCR2 branch trusts the engine's in-memory text).
  * The stale-model OSError mapping regresses and users again see
    ``huggingface.co/C:\\Users\\...`` garbage instead of the
    download-model hint.
  * Missing ``trust_remote_code`` files in the manifest break the
    engine silently.

Running a REAL GOT-OCR 2.0 inference needs:
  * pip install torch transformers einops accelerate pillow
  * ``resources/models/got_ocr2/`` populated with the manifest files
  * ~4 GB RAM for the load + inference

Tests that need that are marked ``requires_real_got_ocr2`` and skip
otherwise.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.application.engines.got_ocr_engine import GOTOCREngine
from src.application.pipeline import OCRPipeline
from src.core.image_preprocessor import ImagePreprocessor
from src.core.models import OCRConfig, OCRJobConfig, ProfileData
from src.core.text_postprocessor import TextPostprocessor
from src.infrastructure.model_manager import GOT_OCR2_SPEC, ModelManager
from src.infrastructure.tesseract_wrapper import TesseractWrapper
from src.shared.types import JobStatus, OCREngineKind
from tests.integration._real_ocr_helpers import (
    render_clean_text_pdf,
)

# ---------------------------------------------------------------------------
# Fixtures: fake torch + transformers, fake model manager with weights seeded
# ---------------------------------------------------------------------------


@pytest.fixture
def _fake_got_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Install fake torch + transformers modules + seeded model dir.

    The ``GOTOCREngine`` does lazy imports of torch/transformers/einops/
    accelerate. Injecting these into ``sys.modules`` lets the engine
    import cleanly without the real 2 GB of ML wheels.
    """
    # torch — just the two attributes GOT engine touches.
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        zeros=lambda *a, **kw: object(),
        float16=object(),
        float32=object(),
    )
    fake_torch.__name__ = "torch"

    # transformers — the two from_pretrained's we call.
    fake_transformers = types.SimpleNamespace(
        AutoModel=types.SimpleNamespace(
            from_pretrained=MagicMock(return_value=_build_fake_model()),
        ),
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=MagicMock(return_value=MagicMock(name="tokenizer")),
        ),
    )
    fake_transformers.__name__ = "transformers"

    # Stub every transitive dep that is_available probes for. Keep
    # in sync with the probe list in got_ocr_engine.is_available().
    for dep_name in (
        "einops", "accelerate", "torchvision", "verovio",
        "tiktoken", "safetensors",
    ):
        fake = types.SimpleNamespace()
        fake.__name__ = dep_name
        monkeypatch.setitem(sys.modules, dep_name, fake)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    # Seed the model manager so ``is_available`` returns True.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    target = models_dir / GOT_OCR2_SPEC.model_id
    target.mkdir()
    for f in GOT_OCR2_SPEC.files:
        (target / f.name).write_bytes(
            b"x" * max(f.size_bytes or 2048, f.min_size_bytes or 2048)
        )

    manager = ModelManager(models_dir=models_dir)
    return fake_transformers, manager


def _build_fake_model() -> MagicMock:
    """Build a fake model whose ``.chat(tok, img, ocr_type=...)`` returns
    a deterministic string we can assert on.
    """
    model = MagicMock(name="model")
    model.chat = MagicMock(return_value="RECOGNISED FROM GOT")
    model.eval = MagicMock()
    return model


@pytest.fixture(autouse=True)
def _isolated_ocr_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Per-test OCR cache dir — prevents cross-test HIT pollution."""
    cache_dir = tmp_path / "ocr-cache"
    cache_dir.mkdir()
    monkeypatch.setattr(
        "src.infrastructure.ocr_cache.OCR_CACHE_DIR", cache_dir
    )
    return cache_dir


# ---------------------------------------------------------------------------
# Engine-dispatch tests — profile.ocr.engine routes to the right implementation
# ---------------------------------------------------------------------------


class TestGOTEngineDispatch:
    """When the user picks the GOT-OCR2 profile, the pipeline must
    route to GOTOCREngine, not TesseractEngine."""

    def test_got_ocr2_profile_uses_got_engine(
        self, tmp_path: Path, _fake_got_stack
    ) -> None:
        _, manager = _fake_got_stack
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="anything"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = _got_profile()

        engine = GOTOCREngine(model_manager=manager)
        with patch("src.application.engines.get_engine", return_value=engine):
            pipeline = _pipeline()
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=profile,
                )
            )

        assert result.status is JobStatus.COMPLETED, result.error
        # The engine's in-memory text made it to the page result
        # (not Tesseract output, which wouldn't say "GOT").
        assert any(
            "GOT" in (p.text or "").upper() for p in result.pages
        ), f"pages={[p.text for p in result.pages]!r}"


# ---------------------------------------------------------------------------
# Postprocess on engine-returned text — the subtle branch in pipeline
# ---------------------------------------------------------------------------


class TestGOTPostprocess:
    """Unlike Tesseract (where text comes out of the output PDF layer),
    GOT-OCR2 returns text directly from ``model.chat()``. The pipeline's
    ``_extract_and_postprocess`` must apply postprocess to THAT text,
    not overwrite it by re-reading the PDF (which GOT-OCR2 embeds as
    invisible-text-layer only after postprocess)."""

    def test_postprocess_applied_to_got_engine_text(
        self, tmp_path: Path, _fake_got_stack
    ) -> None:
        """Fake model returns text with extra whitespace. With
        ``normalize_whitespace=True`` the result should be collapsed."""
        fake_transformers, manager = _fake_got_stack

        # Override model.chat to produce text that postprocess will
        # transform in an observable way.
        returned = fake_transformers.AutoModel.from_pretrained(
            "unused", trust_remote_code=True
        )
        returned.chat = MagicMock(return_value="MANY    SPACES\n\n\n\nHERE")

        # Rebuild the AutoModel stub so the engine's _load_model picks
        # up the new chat behaviour.
        fake_transformers.AutoModel.from_pretrained = MagicMock(
            return_value=returned
        )

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="ignored"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = _got_profile(normalize_whitespace=True)
        engine = GOTOCREngine(model_manager=manager)
        with patch("src.application.engines.get_engine", return_value=engine):
            pipeline = _pipeline()
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=profile,
                )
            )

        assert result.status is JobStatus.COMPLETED, result.error
        text = result.pages[0].text
        # Postprocess must have fired — no runs of 4+ spaces / 3+ newlines.
        assert "    " not in text, f"normalize_whitespace didn't run: {text!r}"
        assert "\n\n\n" not in text, f"normalize_whitespace didn't run: {text!r}"
        # Core content survived.
        assert "MANY" in text and "HERE" in text

    def test_got_engine_text_is_not_overwritten_by_pdf_extraction(
        self, tmp_path: Path, _fake_got_stack
    ) -> None:
        """Regression: a bug that re-read text from the output PDF
        would clobber GOT-OCR2's in-memory result with whatever
        OCRmyPDF-style extraction produced (empty, since GOT engine
        draws invisible text only after postprocess)."""
        _, manager = _fake_got_stack

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="ignored"
        )
        output_pdf = tmp_path / "out.pdf"

        profile = _got_profile()
        engine = GOTOCREngine(model_manager=manager)
        with patch("src.application.engines.get_engine", return_value=engine):
            pipeline = _pipeline()
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=profile,
                )
            )

        assert result.status is JobStatus.COMPLETED
        # The engine's string survived through postprocess.
        assert "RECOGNISED FROM GOT" in result.pages[0].text.upper() or \
               "GOT" in result.pages[0].text.upper()


# ---------------------------------------------------------------------------
# Model-missing + stale-model error paths — actionable user-facing errors
# ---------------------------------------------------------------------------


class TestGOTMissingModel:
    """What happens when the user picks GOT-OCR2 but the weights
    aren't downloaded (or are stale after an app upgrade that added
    new trust_remote_code modules to the manifest)."""

    def test_missing_model_fails_preflight_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model manager reports not-available → preflight returns
        FAILED with the "download from Settings" hint, NOT a crash
        deep inside ``AutoTokenizer.from_pretrained``."""
        # Install the minimal fake torch/transformers/einops so the
        # engine's is_available passes its ML-stack probes — we want
        # the MODEL-missing message, not the torch-missing one.
        for dep in (
            "torch", "transformers", "einops", "accelerate",
            "torchvision", "verovio", "tiktoken", "safetensors",
        ):
            fake = types.SimpleNamespace(
                __name__=dep,
                cuda=types.SimpleNamespace(is_available=lambda: False),
                zeros=lambda *a, **kw: object(),
            )
            monkeypatch.setitem(sys.modules, dep, fake)

        # Empty model dir — ``is_available`` should return False.
        empty_manager = ModelManager(models_dir=tmp_path / "empty")
        engine = GOTOCREngine(model_manager=empty_manager)

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="anything"
        )
        output_pdf = tmp_path / "out.pdf"

        with patch("src.application.engines.get_engine", return_value=engine):
            pipeline = _pipeline()
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_got_profile(),
                )
            )

        assert result.status is JobStatus.FAILED
        assert result.error
        # Actionable UI-facing text.
        assert "Скачать модель" in result.error or "не скачана" in (
            result.error or ""
        ).lower()

    def test_stale_model_oserror_becomes_download_hint(
        self, tmp_path: Path, _fake_got_stack
    ) -> None:
        """Files on disk match the old manifest but are missing a new
        trust_remote_code module → HF OSError → our wrapper maps to
        EngineNotAvailableError with a clear message."""
        fake_transformers, manager = _fake_got_stack

        # AutoTokenizer.from_pretrained raises the HF "file not found"
        # OSError shape.
        hf_error = OSError(
            "got_ocr2 does not appear to have a file named "
            "tokenization_qwen.py. Checkout "
            "'https://huggingface.co/got_ocr2/tree/main'"
        )
        fake_transformers.AutoTokenizer.from_pretrained = MagicMock(
            side_effect=hf_error
        )

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="anything"
        )
        output_pdf = tmp_path / "out.pdf"

        engine = GOTOCREngine(model_manager=manager)
        with patch("src.application.engines.get_engine", return_value=engine):
            pipeline = _pipeline()
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_got_profile(),
                )
            )

        assert result.status is JobStatus.FAILED
        # Clean Russian hint, not a raw OSError.
        assert "Скачать модель" in (result.error or "")
        # The raw HF error about "huggingface.co/..." must NOT leak
        # to the user.
        assert "huggingface.co" not in (result.error or "")


# ---------------------------------------------------------------------------
# Multi-page GOT-OCR2 run — sanity that page iteration works
# ---------------------------------------------------------------------------


class TestGOTMultipage:
    def test_multi_page_got_returns_per_page_text(
        self, tmp_path: Path, _fake_got_stack
    ) -> None:
        _, manager = _fake_got_stack
        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="irrelevant", pages=3
        )
        output_pdf = tmp_path / "out.pdf"

        engine = GOTOCREngine(model_manager=manager)
        with patch("src.application.engines.get_engine", return_value=engine):
            pipeline = _pipeline()
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_got_profile(),
                )
            )

        assert result.status is JobStatus.COMPLETED
        assert result.page_count == 3
        # Every page has non-empty text (from the fake model).
        for p in result.pages:
            assert p.text, f"page {p.page_number} has empty text"


# ---------------------------------------------------------------------------
# GOT-OCR2 + Cyrillic path — the user's ultimate scenario
# ---------------------------------------------------------------------------


class TestGOTCyrillicPath:
    def test_got_engine_on_cyrillic_input_output_paths(
        self, tmp_path: Path, _fake_got_stack
    ) -> None:
        """Cyrillic dir + cyrillic filename + GOT-OCR2 — exercises the
        8.3-short-path fallback in ``_safe_model_path`` on Windows,
        and the cv2.imencode-to-Path.write_bytes trick in
        ``_save_png`` for Unicode preprocessing paths."""
        _, manager = _fake_got_stack
        cyrillic = tmp_path / "Т.Н. 020" / "документы"
        cyrillic.mkdir(parents=True)
        input_pdf = render_clean_text_pdf(
            cyrillic / "скан.pdf", text="unused"
        )
        output_pdf = cyrillic / "скан_ocr.pdf"

        engine = GOTOCREngine(model_manager=manager)
        with patch("src.application.engines.get_engine", return_value=engine):
            pipeline = _pipeline()
            result = pipeline.run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=_got_profile(),
                )
            )

        assert result.status is JobStatus.COMPLETED, result.error
        assert output_pdf.exists()


# ---------------------------------------------------------------------------
# Built-in handwritten_mixed profile → GOT-OCR2 engine wiring
# ---------------------------------------------------------------------------


class TestHandwrittenMixedProfile:
    def test_handwritten_mixed_profile_uses_got_engine(
        self, tmp_path: Path, _fake_got_stack
    ) -> None:
        """The bundled ``handwritten_mixed`` profile has
        ``engine=GOT_OCR2``. Loading it through ProfileStorage and
        running the pipeline must end up in GOTOCREngine.run, not
        TesseractEngine.run."""
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        _, manager = _fake_got_stack
        storage = ProfileStorage(profiles_dir=tmp_path / "profiles")
        pmgr = ProfileManager(storage)
        pmgr.initialize_builtins()
        profile = pmgr.load("handwritten_mixed")
        assert profile.ocr.engine is OCREngineKind.GOT_OCR2, (
            "regression: handwritten_mixed bundled profile must use "
            "GOT-OCR 2.0 engine"
        )

        input_pdf = render_clean_text_pdf(
            tmp_path / "in.pdf", text="ignored"
        )
        output_pdf = tmp_path / "out.pdf"

        engine = GOTOCREngine(model_manager=manager)
        with patch("src.application.engines.get_engine", return_value=engine):
            result = _pipeline().run(
                OCRJobConfig(
                    input_path=str(input_pdf),
                    output_path=str(output_pdf),
                    profile=profile,
                )
            )

        assert result.status is JobStatus.COMPLETED, result.error
        assert any("GOT" in (p.text or "").upper() for p in result.pages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pipeline() -> OCRPipeline:
    return OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        tesseract=TesseractWrapper(),
        compute_confidence=False,
    )


def _got_profile(
    *, normalize_whitespace: bool = False, normalize_unicode: bool = False
) -> ProfileData:
    from src.core.models import (
        BinarizationConfig,
        PostprocessConfig,
        PreprocessConfig,
    )
    from src.shared.types import BinarizationMethod

    pre = PreprocessConfig(
        binarization=BinarizationConfig(method=BinarizationMethod.NONE),
    )
    pre.deskew.enabled = False
    ocr = OCRConfig(
        engine=OCREngineKind.GOT_OCR2,
        languages=["rus", "eng"],
        primary_language="rus",
        dpi=150,
    )
    post = PostprocessConfig(
        autocorrect_russian=False,
        autocorrect_english=False,
        merge_hyphenated=False,
        normalize_whitespace=normalize_whitespace,
        normalize_unicode=normalize_unicode,
        remove_artifacts=False,
    )
    return ProfileData(name="got-e2e", ocr=ocr, preprocess=pre, postprocess=post)
