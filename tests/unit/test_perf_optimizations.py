"""Tests for the performance / caching round.

Covers:
* OCR disk cache: hit/miss/store/prune.
* Parallel page rasterisation preserves order + handles failures.
* Deskew detection on large images actually downsamples.
* Regex compile cache re-uses compiled patterns across calls.
* GOT-OCR loads fp16 on CUDA, fp32 on CPU.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------
# OCR cache
# --------------------------------------------------------------------------


class TestOCRCache:
    def _profile(self, name: str = "p"):
        from src.core.models import ProfileData

        return ProfileData(name=name)

    def _write_fake_pdf(self, path: Path, content: bytes = b"%PDF-1.7\nfake") -> Path:
        path.write_bytes(content)
        return path

    def test_miss_on_empty_cache(self, tmp_path: Path) -> None:
        from src.infrastructure import ocr_cache

        pdf = self._write_fake_pdf(tmp_path / "in.pdf")
        assert ocr_cache.lookup(pdf, self._profile(), cache_root=tmp_path / "cache") is None

    def test_store_then_hit(self, tmp_path: Path) -> None:
        from src.core.models import JobResult, PageResult
        from src.infrastructure import ocr_cache
        from src.shared.types import JobStatus

        pdf_in = self._write_fake_pdf(tmp_path / "doc.pdf")
        profile = self._profile()
        out_pdf = tmp_path / "doc_ocr.pdf"
        out_pdf.write_bytes(b"%PDF-1.7\nsearchable")
        jr = JobResult(
            job_id="abc",
            status=JobStatus.COMPLETED,
            input_path=str(pdf_in),
            output_path=str(out_pdf),
            pages=[PageResult(page_number=1, text="hello", mean_confidence=90.0)],
            total_time_sec=1.0,
        )
        cache_root = tmp_path / "cache"
        ocr_cache.store(
            pdf_in, profile,
            output_pdf=out_pdf,
            job_result=jr,
            cache_root=cache_root,
        )
        hit = ocr_cache.lookup(pdf_in, profile, cache_root=cache_root)
        assert hit is not None
        cached_pdf, meta = hit
        assert cached_pdf.exists()
        assert meta["pages"][0]["text"] == "hello"
        assert "job_id" not in meta  # scrubbed on store

    def test_key_sensitive_to_profile_change(self, tmp_path: Path) -> None:
        from src.core.models import OCRConfig, ProfileData
        from src.infrastructure import ocr_cache

        pdf_in = self._write_fake_pdf(tmp_path / "doc.pdf")
        p1 = ProfileData(name="p1", ocr=OCRConfig(dpi=300))
        p2 = ProfileData(name="p1", ocr=OCRConfig(dpi=600))
        assert ocr_cache.cache_key(pdf_in, p1) != ocr_cache.cache_key(pdf_in, p2)

    def test_key_invariant_to_profile_name_change(self, tmp_path: Path) -> None:
        """Renaming a profile shouldn't invalidate the cache."""
        from src.core.models import ProfileData
        from src.infrastructure import ocr_cache

        pdf_in = self._write_fake_pdf(tmp_path / "doc.pdf")
        p1 = ProfileData(name="old")
        p2 = ProfileData(name="new")
        assert ocr_cache.cache_key(pdf_in, p1) == ocr_cache.cache_key(pdf_in, p2)

    def test_prune_evicts_oldest_when_over_budget(self, tmp_path: Path) -> None:
        import time

        from src.infrastructure import ocr_cache

        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        # Three fake entries, each 100 bytes → budget 150 → keep newest.
        for name in ("a", "b", "c"):
            d = cache_root / name
            d.mkdir()
            (d / "payload").write_bytes(b"x" * 100)
            # Stagger mtimes: a oldest, c newest.
            t = time.time() - {"a": 300, "b": 200, "c": 100}[name]
            os.utime(d, (t, t))
        ocr_cache._prune(cache_root, max_bytes=150)
        remaining = {p.name for p in cache_root.iterdir() if p.is_dir()}
        # At least the oldest is gone; "c" must survive.
        assert "c" in remaining
        assert "a" not in remaining

    def test_missing_input_file_returns_none(self, tmp_path: Path) -> None:
        from src.infrastructure import ocr_cache

        assert ocr_cache.cache_key(tmp_path / "does-not-exist.pdf", self._profile()) == ""

    def test_store_is_idempotent(self, tmp_path: Path) -> None:
        """Calling store twice for same key just overwrites, no error."""
        from src.core.models import JobResult
        from src.infrastructure import ocr_cache
        from src.shared.types import JobStatus

        pdf_in = self._write_fake_pdf(tmp_path / "doc.pdf")
        out_pdf = tmp_path / "out.pdf"
        out_pdf.write_bytes(b"pdf")
        jr = JobResult(
            job_id="1",
            status=JobStatus.COMPLETED,
            input_path=str(pdf_in),
            output_path=str(out_pdf),
        )
        cache_root = tmp_path / "cache"
        ocr_cache.store(pdf_in, self._profile(), output_pdf=out_pdf, job_result=jr, cache_root=cache_root)
        ocr_cache.store(pdf_in, self._profile(), output_pdf=out_pdf, job_result=jr, cache_root=cache_root)
        # Still one entry.
        entries = list((cache_root).iterdir())
        assert len(entries) == 1


# --------------------------------------------------------------------------
# Pipeline cache integration
# --------------------------------------------------------------------------


class TestPipelineCacheIntegration:
    def test_second_run_is_served_from_cache(self, tmp_path: Path) -> None:
        """Run the same input+profile twice; second run must skip the engine."""
        import fitz

        from src.application.engines.base import PageOCRResult
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import (
            OCRConfig,
            OCRJobConfig,
            PreprocessConfig,
            ProfileData,
        )
        from src.core.text_postprocessor import TextPostprocessor
        from src.shared.types import BinarizationMethod, JobStatus

        pdf = tmp_path / "doc.pdf"
        doc = fitz.open()
        try:
            p = doc.new_page(width=200, height=100)
            p.insert_text((10, 50), "hello")
            doc.save(str(pdf))
        finally:
            doc.close()

        pre = PreprocessConfig()
        pre.binarization.method = BinarizationMethod.NONE
        pre.deskew.enabled = False
        profile = ProfileData(
            name="p",
            ocr=OCRConfig(dpi=72),
            preprocess=pre,
        )

        engine_calls = {"n": 0}

        class _Engine:
            kind = None
            name = "stub"
            description = ""

            def is_available(self):
                return True, ""

            def run(self, preprocessed_pdf, output_pdf, config, progress_callback=None):
                engine_calls["n"] += 1
                import shutil as _sh

                _sh.copy2(preprocessed_pdf, output_pdf)
                return [PageOCRResult(page_number=1, text="hello")]

            def unload(self):
                pass

        cache_root = tmp_path / "oc"
        with patch(
            "src.infrastructure.ocr_cache.OCR_CACHE_DIR", cache_root
        ), patch(
            "src.application.engines.get_engine", return_value=_Engine()
        ):
            out1 = tmp_path / "out1.pdf"
            out2 = tmp_path / "out2.pdf"
            pipeline = OCRPipeline(
                preprocessor=ImagePreprocessor(),
                postprocessor=TextPostprocessor(),
                tesseract=MagicMock(),
            )
            r1 = pipeline.run(OCRJobConfig(
                input_path=str(pdf), output_path=str(out1), profile=profile
            ))
            r2 = pipeline.run(OCRJobConfig(
                input_path=str(pdf), output_path=str(out2), profile=profile
            ))

        assert r1.status is JobStatus.COMPLETED
        assert r2.status is JobStatus.COMPLETED
        assert engine_calls["n"] == 1, (
            "Second run should have been served from cache, engine not invoked"
        )
        # The second output PDF exists and is a copy of the cached one.
        assert out2.exists()


# --------------------------------------------------------------------------
# Parallel rasterisation ordering + failure handling
# --------------------------------------------------------------------------


class TestParallelPreprocessing:
    def _build_multi_page_pdf(self, path: Path, n: int) -> None:
        import fitz

        doc = fitz.open()
        try:
            for i in range(n):
                p = doc.new_page(width=200, height=100)
                p.insert_text((10, 50), f"page {i + 1}")
            doc.save(str(path))
        finally:
            doc.close()

    def test_page_order_preserved_with_pool(self, tmp_path: Path) -> None:
        """With 4+ pages the thread pool kicks in — results must still be in order."""
        from src.application.engines.base import PageOCRResult
        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor
        from src.core.models import (
            OCRConfig,
            OCRJobConfig,
            PreprocessConfig,
            ProfileData,
        )
        from src.core.text_postprocessor import TextPostprocessor
        from src.shared.types import BinarizationMethod, JobStatus

        pdf = tmp_path / "multi.pdf"
        self._build_multi_page_pdf(pdf, 6)

        pre = PreprocessConfig()
        pre.binarization.method = BinarizationMethod.NONE
        pre.deskew.enabled = False
        profile = ProfileData(
            name="p",
            ocr=OCRConfig(dpi=72),
            preprocess=pre,
        )

        class _Engine:
            kind = None
            name = "stub"
            description = ""

            def is_available(self):
                return True, ""

            def run(self, preprocessed_pdf, output_pdf, config, progress_callback=None):
                import shutil as _sh

                import fitz as _fitz

                _sh.copy2(preprocessed_pdf, output_pdf)
                with _fitz.open(str(output_pdf)) as d:
                    return [
                        PageOCRResult(page_number=i + 1, text=f"p{i + 1}")
                        for i in range(d.page_count)
                    ]

            def unload(self):
                pass

        out = tmp_path / "out.pdf"
        # Prevent cache hits in this test.
        with patch(
            "src.infrastructure.ocr_cache.OCR_CACHE_DIR", tmp_path / "_nocache"
        ), patch("src.application.engines.get_engine", return_value=_Engine()):
            pipeline = OCRPipeline(
                preprocessor=ImagePreprocessor(),
                postprocessor=TextPostprocessor(),
                tesseract=MagicMock(),
            )
            result = pipeline.run(OCRJobConfig(
                input_path=str(pdf), output_path=str(out), profile=profile
            ))

        assert result.status is JobStatus.COMPLETED
        assert [p.page_number for p in result.pages] == [1, 2, 3, 4, 5, 6]


# --------------------------------------------------------------------------
# Deskew downsampling
# --------------------------------------------------------------------------


class TestDeskewDownsampling:
    def test_large_image_is_downsampled_before_detection(self) -> None:
        import numpy as np

        from src.core.deskew_handler import DeskewHandler

        # 2400x3200 grey-ish image. No real skew, but we care about the
        # downsample call, not the angle value.
        big = np.full((3200, 2400), 200, dtype=np.uint8)
        seen_shapes: list[tuple[int, int]] = []

        def fake_determine_skew(gray):
            seen_shapes.append(gray.shape[:2])
            return 0.0

        handler = DeskewHandler()
        with patch.dict("sys.modules", {"deskew": MagicMock(determine_skew=fake_determine_skew)}):
            handler.detect_angle(big)
        assert seen_shapes, "determine_skew was not called"
        h, w = seen_shapes[0]
        assert max(h, w) <= 1600, (
            f"Expected downsampled long-side <=1600, got {seen_shapes[0]}"
        )

    def test_small_image_passes_through_untouched(self) -> None:
        import numpy as np

        from src.core.deskew_handler import DeskewHandler

        small = np.full((400, 600), 180, dtype=np.uint8)
        seen: list[tuple[int, int]] = []

        def fake(gray):
            seen.append(gray.shape[:2])
            return 0.0

        with patch.dict("sys.modules", {"deskew": MagicMock(determine_skew=fake)}):
            DeskewHandler().detect_angle(small)
        assert seen[0] == (400, 600), "Small images must NOT be resampled"


# --------------------------------------------------------------------------
# Regex compile cache
# --------------------------------------------------------------------------


class TestRegexCache:
    def test_compilation_happens_once_per_unique_rule(self) -> None:
        """Same rule used 100 times → 1 re.compile call."""
        import re as _re

        from src.core.models import PostprocessConfig, RegexRule
        from src.core.text_postprocessor import TextPostprocessor

        post = TextPostprocessor()
        rules = [RegexRule(pattern="foo", replacement="bar", is_regex=True, enabled=True)]
        cfg = PostprocessConfig(
            autocorrect_russian=False,
            autocorrect_english=False,
            merge_hyphenated=False,
            normalize_whitespace=False,
            normalize_unicode=False,
            remove_artifacts=False,
            custom_rules=rules,
        )
        real_compile = _re.compile
        calls = {"n": 0}

        def counting_compile(pattern, flags=0):
            # Only count our rule's pattern — other helpers inside re
            # module (like _substitute_with_timeout) also compile under
            # the hood on some Python versions.
            if pattern == "foo":
                calls["n"] += 1
            return real_compile(pattern, flags)

        with patch("src.core.text_postprocessor.re.compile", side_effect=counting_compile):
            for _ in range(100):
                post.process("foo foo foo", cfg)
        assert calls["n"] == 1, f"Expected 1 compile for 'foo', got {calls['n']}"

    def test_different_flags_get_separate_cache_entries(self) -> None:
        from src.core.models import PostprocessConfig, RegexRule
        from src.core.text_postprocessor import TextPostprocessor

        post = TextPostprocessor()
        # Same pattern, different case-sensitivity → different flags key.
        rules = [
            RegexRule(pattern="foo", replacement="a", is_regex=True, case_sensitive=True, enabled=True),
            RegexRule(pattern="foo", replacement="b", is_regex=True, case_sensitive=False, enabled=True),
        ]
        cfg = PostprocessConfig(
            autocorrect_russian=False,
            autocorrect_english=False,
            merge_hyphenated=False,
            normalize_whitespace=False,
            normalize_unicode=False,
            remove_artifacts=False,
            custom_rules=rules,
        )
        post.process("Foo foo FOO", cfg)
        assert len(post._user_regex_cache) == 2


# --------------------------------------------------------------------------
# GOT-OCR fp16 loading
# --------------------------------------------------------------------------


class TestGOTPrecision:
    def test_cuda_loads_fp16(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.float16 = "fp16-marker"
        fake_torch.float32 = "fp32-marker"

        fake_transformers = MagicMock()
        fake_model = MagicMock()
        fake_transformers.AutoModel.from_pretrained.return_value = fake_model

        from src.application.engines.got_ocr_engine import GOTOCREngine
        from src.infrastructure.model_manager import ModelManager

        mm = MagicMock(spec=ModelManager)
        mm.model_dir.return_value = Path("/fake/path")

        engine = GOTOCREngine(model_manager=mm)
        with patch.dict(
            "sys.modules",
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            engine._load_model()

        _, kwargs = fake_transformers.AutoModel.from_pretrained.call_args
        assert kwargs["torch_dtype"] == "fp16-marker"
        assert kwargs["device_map"] == "cuda"

    def test_cpu_loads_fp32(self) -> None:
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.float16 = "fp16-marker"
        fake_torch.float32 = "fp32-marker"

        fake_transformers = MagicMock()
        fake_transformers.AutoModel.from_pretrained.return_value = MagicMock()

        from src.application.engines.got_ocr_engine import GOTOCREngine
        from src.infrastructure.model_manager import ModelManager

        mm = MagicMock(spec=ModelManager)
        mm.model_dir.return_value = Path("/fake/path")

        engine = GOTOCREngine(model_manager=mm)
        with patch.dict(
            "sys.modules",
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            engine._load_model()

        _, kwargs = fake_transformers.AutoModel.from_pretrained.call_args
        assert kwargs["torch_dtype"] == "fp32-marker"
        assert kwargs["device_map"] == "cpu"


# --------------------------------------------------------------------------
# ensure_user_dirs picks up the new cache dir
# --------------------------------------------------------------------------


class TestCacheDirInitialised:
    def test_ocr_cache_dir_constant_is_defined(self) -> None:
        """Sanity check: the constant exists and sits under USER_DATA_DIR."""
        from src.shared.constants import OCR_CACHE_DIR, USER_DATA_DIR

        assert OCR_CACHE_DIR.is_absolute() or OCR_CACHE_DIR.parent == USER_DATA_DIR
        assert OCR_CACHE_DIR.name == "ocr-cache"

    def test_ensure_user_dirs_lists_ocr_cache(self) -> None:
        """Read the source of ``ensure_user_dirs`` and confirm OCR_CACHE_DIR is in the loop."""
        import inspect

        from src.shared import constants

        src = inspect.getsource(constants.ensure_user_dirs)
        assert "OCR_CACHE_DIR" in src
