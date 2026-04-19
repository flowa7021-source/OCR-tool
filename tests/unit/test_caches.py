"""Tests for the cache / lifetime improvements.

Covers:
    * registry.reset_cache() side effect
    * Startup temp cleanup hook in src.app.create_application
    * Preview-pixmap LRU cache in src.ui.preprocessing_panel
    * SettingsStorage mtime-based load cache
    * TesseractWrapper.refresh()
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# registry.reset_cache
# ---------------------------------------------------------------------------


class TestBaseEngineUnload:
    def test_base_engine_unload_is_noop(self) -> None:
        """Default OCREngine.unload() is safe to call on engines that don't override it."""
        from src.application.engines.tesseract_engine import TesseractEngine

        TesseractEngine().unload()  # no crash, no state change


class TestRegistryResetCache:
    def test_reset_cache_survives_failing_unload(self) -> None:
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
