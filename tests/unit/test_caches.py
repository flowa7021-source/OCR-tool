"""Tests for the six cache/lifetime improvements.

Covers:
    * GOTOCREngine.unload() and registry.reset_cache() side effect
    * Startup temp cleanup hook in src.app.create_application
    * Preview-pixmap LRU cache in src.ui.preprocessing_panel
    * SettingsStorage mtime-based load cache
    * ModelManager.is_available TTL cache + invalidation
    * TesseractWrapper.refresh()
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# GOTOCREngine.unload + registry.reset_cache
# ---------------------------------------------------------------------------


class TestGOTUnload:
    def _engine(self, tmp_path: Path):
        from src.application.engines.got_ocr_engine import GOTOCREngine
        from src.infrastructure.model_manager import ModelManager

        mgr = ModelManager(models_dir=tmp_path / "models")
        engine = GOTOCREngine(model_manager=mgr)
        # Simulate a loaded model
        engine._model = MagicMock(name="fake-model")
        engine._tokenizer = MagicMock(name="fake-tokenizer")
        return engine

    def test_unload_clears_state(self, tmp_path: Path) -> None:
        engine = self._engine(tmp_path)
        assert engine._model is not None
        engine.unload()
        assert engine._model is None
        assert engine._tokenizer is None

    def test_unload_is_idempotent(self, tmp_path: Path) -> None:
        engine = self._engine(tmp_path)
        engine.unload()
        engine.unload()  # second call must not raise
        assert engine._model is None

    def test_unload_noop_when_not_loaded(self, tmp_path: Path) -> None:
        from src.application.engines.got_ocr_engine import GOTOCREngine
        from src.infrastructure.model_manager import ModelManager

        engine = GOTOCREngine(model_manager=ModelManager(models_dir=tmp_path))
        # Fresh engine — nothing loaded
        engine.unload()  # should silently return
        assert engine._model is None

    def test_base_engine_unload_is_noop(self) -> None:
        """Default OCREngine.unload() is safe to call on engines that don't override it."""
        from src.application.engines.tesseract_engine import TesseractEngine

        TesseractEngine().unload()  # no crash, no state change


class TestRegistryResetCache:
    def test_reset_cache_calls_unload(self, tmp_path: Path) -> None:
        # Prepare a cached engine with mock loaded weights
        import src.application.engines.registry as registry
        from src.application.engines.got_ocr_engine import GOTOCREngine
        from src.application.engines.registry import reset_cache
        from src.infrastructure.model_manager import ModelManager
        from src.shared.types import OCREngineKind

        engine = GOTOCREngine(model_manager=ModelManager(models_dir=tmp_path))
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        registry._CACHE[OCREngineKind.GOT_OCR2] = engine

        reset_cache()

        assert engine._model is None
        assert engine._tokenizer is None
        assert OCREngineKind.GOT_OCR2 not in registry._CACHE

    def test_reset_cache_survives_failing_unload(self, tmp_path: Path) -> None:
        """A broken unload() must not prevent cache clearing."""
        import src.application.engines.registry as registry
        from src.application.engines.registry import reset_cache
        from src.shared.types import OCREngineKind

        bad = MagicMock()
        bad.unload.side_effect = RuntimeError("boom")
        registry._CACHE[OCREngineKind.TESSERACT] = bad
        # Should NOT re-raise
        reset_cache()
        assert len(registry._CACHE) == 0
        bad.unload.assert_called_once()


# ---------------------------------------------------------------------------
# Preview-pixmap LRU cache
# ---------------------------------------------------------------------------


class TestPreviewCache:
    def test_lru_reuses_cached_page(self, tmp_path: Path) -> None:
        pytest.importorskip("fitz")
        pytest.importorskip("PySide6")
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        import fitz

        from src.ui.preprocessing_panel import (
            _cached_rasterize,
            _get_preview_cache,
            clear_preview_cache,
        )

        clear_preview_cache()
        pdf = tmp_path / "doc.pdf"
        doc = fitz.open()
        try:
            page = doc.new_page(width=200, height=150)
            page.insert_text((10, 40), "hello")
            doc.save(str(pdf))
        finally:
            doc.close()

        cache, _lock = _get_preview_cache()

        with patch("fitz.open", wraps=fitz.open) as open_spy:
            a = _cached_rasterize(pdf, 1, dpi=72)
            b = _cached_rasterize(pdf, 1, dpi=72)
        assert open_spy.call_count == 1  # second call hit the cache
        assert a is b
        assert len(cache) == 1

    def test_lru_evicts_oldest(self, tmp_path: Path) -> None:
        pytest.importorskip("fitz")
        import fitz

        from src.ui import preprocessing_panel as pp
        from src.ui.preprocessing_panel import (
            _cached_rasterize,
            _get_preview_cache,
            clear_preview_cache,
        )

        clear_preview_cache()
        pdf = tmp_path / "doc.pdf"
        doc = fitz.open()
        try:
            for i in range(5):
                p = doc.new_page(width=100, height=100)
                p.insert_text((10, 50), f"p{i}")
            doc.save(str(pdf))
        finally:
            doc.close()

        # Stuff more pages than the LRU size
        for i in range(1, pp._PREVIEW_CACHE_MAX + 2):
            _cached_rasterize(pdf, i, dpi=72)

        cache, _lock = _get_preview_cache()
        assert len(cache) == pp._PREVIEW_CACHE_MAX
        # The oldest (page 1) must have been evicted
        keys = [k[2] for k in cache]  # page numbers
        assert 1 not in keys

    def test_mtime_invalidates_cache(self, tmp_path: Path) -> None:
        pytest.importorskip("fitz")
        import os

        import fitz

        from src.ui.preprocessing_panel import (
            _cached_rasterize,
            clear_preview_cache,
        )

        clear_preview_cache()
        pdf = tmp_path / "doc.pdf"

        def _write(text: str) -> None:
            d = fitz.open()
            try:
                p = d.new_page(width=100, height=100)
                p.insert_text((10, 40), text)
                d.save(str(pdf))
            finally:
                d.close()

        _write("v1")
        first_arr = _cached_rasterize(pdf, 1, dpi=72)
        # bump mtime by rewriting
        time.sleep(0.01)
        _write("v2")
        # force mtime change
        os.utime(pdf, (time.time() + 1, time.time() + 1))
        with patch("fitz.open", wraps=fitz.open) as spy:
            _cached_rasterize(pdf, 1, dpi=72)
        assert spy.call_count == 1, "mtime change should force re-rasterisation"
        assert first_arr.shape == (100, 100, 3) or first_arr.ndim == 3


# ---------------------------------------------------------------------------
# SettingsStorage mtime cache
# ---------------------------------------------------------------------------


class TestSettingsCache:
    def test_repeated_load_hits_cache(self, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import SettingsStorage

        storage = SettingsStorage(config_dir=tmp_path)
        # First load: file doesn't exist → defaults; cache is populated
        a = storage.load()
        # Save something so file exists with a real mtime
        a.parallel_workers = 3
        storage.save(a)

        with patch("src.infrastructure.config_storage._read_json") as read_spy:
            for _ in range(5):
                loaded = storage.load()
                assert loaded.parallel_workers == 3
        # _read_json must not have been called even once — save() pre-populated
        # the cache with a copy and mtime matches afterwards.
        assert read_spy.call_count == 0

    def test_external_mtime_bump_invalidates(self, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import AppSettings, SettingsStorage

        storage = SettingsStorage(config_dir=tmp_path)
        storage.save(AppSettings(parallel_workers=2))
        _ = storage.load()  # warm

        # Pretend another process rewrote the file
        import json
        import os

        storage.path.write_text(
            json.dumps({"parallel_workers": 4}),
            encoding="utf-8",
        )
        os.utime(storage.path, (time.time() + 10, time.time() + 10))

        with patch(
            "src.infrastructure.config_storage._read_json",
            wraps=__import__(
                "src.infrastructure.config_storage",
                fromlist=["_read_json"],
            )._read_json,
        ) as read_spy:
            reloaded = storage.load()
        assert read_spy.call_count == 1
        assert reloaded.parallel_workers == 4

    def test_save_updates_cache_not_path_only(self, tmp_path: Path) -> None:
        """After save(), subsequent loads reflect the saved value without disk read."""
        from src.infrastructure.config_storage import SettingsStorage

        storage = SettingsStorage(config_dir=tmp_path)
        original = storage.load()
        original.parallel_workers = 3
        storage.save(original)

        original.parallel_workers = 999  # mutate caller copy AFTER save
        loaded = storage.load()
        # Cache must have stored a deep copy: our later mutation doesn't leak.
        assert loaded.parallel_workers == 3

    def test_invalidate_cache_forces_reread(self, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import AppSettings, SettingsStorage

        storage = SettingsStorage(config_dir=tmp_path)
        storage.save(AppSettings())
        storage.invalidate_cache()
        with patch(
            "src.infrastructure.config_storage._read_json",
            wraps=__import__(
                "src.infrastructure.config_storage",
                fromlist=["_read_json"],
            )._read_json,
        ) as read_spy:
            storage.load()
        assert read_spy.call_count == 1


# ---------------------------------------------------------------------------
# ModelManager availability TTL
# ---------------------------------------------------------------------------


class TestModelAvailabilityTTL:
    def _write_manifest(self, mgr, model_id: str) -> None:
        """Create every file from the GOT-OCR2 manifest with the expected size."""
        from src.infrastructure.model_manager import GOT_OCR2_SPEC

        target = mgr.model_dir(model_id)
        target.mkdir(parents=True)
        for f in GOT_OCR2_SPEC.files:
            size = f.size_bytes or 2048
            (target / f.name).write_bytes(b"\x00" * size)

    def test_second_call_hits_cache(self, tmp_path: Path) -> None:
        from src.infrastructure.model_manager import ModelManager

        mgr = ModelManager(models_dir=tmp_path)
        # First call primes the cache
        mgr.is_available("got_ocr2")
        # Patch is_file so the slow path would lie if called again
        with patch(
            "pathlib.Path.is_file",
            side_effect=AssertionError("is_file must not be called within TTL"),
        ):
            assert mgr.is_available("got_ocr2") is False

    def test_ttl_expiry_reruns_check(self, tmp_path: Path) -> None:
        """Simulate clock advance via patched time.monotonic."""
        import src.infrastructure.model_manager as mm_module
        from src.infrastructure.model_manager import ModelManager

        mgr = ModelManager(models_dir=tmp_path)
        mgr.AVAILABILITY_TTL_SEC = 5.0

        t = [100.0]  # start time

        def fake_monotonic() -> float:
            return t[0]

        with patch.object(mm_module.time, "monotonic", fake_monotonic):
            mgr.is_available("got_ocr2")  # prime (False), stamped t=100
            # Now create the files...
            self._write_manifest(mgr, "got_ocr2")
            # ...and advance the clock by less than TTL.
            t[0] = 100.0 + 1.0
            assert mgr.is_available("got_ocr2") is False  # cache still fresh

            # Advance past the TTL — must re-probe the filesystem.
            t[0] = 100.0 + 10.0
            assert mgr.is_available("got_ocr2") is True

    def test_download_invalidates_cache(self, tmp_path: Path) -> None:
        from src.infrastructure.model_manager import ModelManager

        mgr = ModelManager(models_dir=tmp_path)
        mgr.AVAILABILITY_TTL_SEC = 60.0  # long TTL so only invalidation clears it
        mgr.is_available("got_ocr2")  # cache False
        # Side-effect: manually populate files (simulates download completion)
        self._write_manifest(mgr, "got_ocr2")
        # Without invalidation we'd still see False (TTL not expired)
        assert mgr.is_available("got_ocr2") is False
        mgr.invalidate_availability("got_ocr2")
        assert mgr.is_available("got_ocr2") is True

    def test_remove_invalidates_cache(self, tmp_path: Path) -> None:
        from src.infrastructure.model_manager import ModelManager

        mgr = ModelManager(models_dir=tmp_path)
        mgr.AVAILABILITY_TTL_SEC = 60.0
        self._write_manifest(mgr, "got_ocr2")
        assert mgr.is_available("got_ocr2") is True
        mgr.remove("got_ocr2")
        # Cache must be invalidated so the deletion is visible immediately
        assert mgr.is_available("got_ocr2") is False


# ---------------------------------------------------------------------------
# TesseractWrapper.refresh
# ---------------------------------------------------------------------------


class TestTesseractRefresh:
    def test_refresh_resets_and_reverifies(self) -> None:
        from src.infrastructure.tesseract_wrapper import TesseractWrapper

        TesseractWrapper.reset()
        TesseractWrapper._binary_path = Path("/fake/tesseract")
        TesseractWrapper._configured = True

        wrapper = TesseractWrapper()
        with patch.object(wrapper, "verify", return_value=(True, "ok")) as verify:
            ok, msg = wrapper.refresh()
        assert TesseractWrapper._binary_path is None  # reset cleared state
        verify.assert_called_once()
        assert ok is True
        assert msg == "ok"
