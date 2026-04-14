"""Tests for the HTR model manager."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.infrastructure.model_manager import (
    GOT_OCR2_SPEC,
    ModelDownloadError,
    ModelFile,
    ModelManager,
    ModelSpec,
)


def _spec_for_test(tmp_url_base: str) -> ModelSpec:
    """Build a tiny throwaway spec pointing at file:// URLs."""
    return ModelSpec(
        model_id="test-model",
        label="Test",
        description="for tests",
        total_size_bytes=64,
        files=[
            ModelFile(name="weights.bin", url=f"{tmp_url_base}/weights.bin", size_bytes=32),
            ModelFile(name="config.json", url=f"{tmp_url_base}/config.json", size_bytes=8),
        ],
    )


def _serve_dir(directory: Path):  # type: ignore[no-untyped-def]
    """Start a quick HTTP server bound to a random port serving ``directory``."""
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    def _factory(*args, **kwargs):
        return SimpleHTTPRequestHandler(*args, directory=str(directory), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return f"http://127.0.0.1:{port}", server


# ---------------------------------------------------------------------------
# Spec / registry
# ---------------------------------------------------------------------------


class TestSpec:
    def test_got_ocr2_spec_present(self) -> None:
        assert GOT_OCR2_SPEC.model_id == "got_ocr2"
        assert GOT_OCR2_SPEC.files
        assert any(f.name == "model.safetensors" for f in GOT_OCR2_SPEC.files)

    def test_unknown_id_raises(self, tmp_path: Path) -> None:
        mgr = ModelManager(models_dir=tmp_path)
        with pytest.raises(KeyError):
            mgr.spec_for("does-not-exist")


# ---------------------------------------------------------------------------
# Presence
# ---------------------------------------------------------------------------


class TestPresence:
    def test_missing_dir_is_unavailable(self, tmp_path: Path) -> None:
        mgr = ModelManager(models_dir=tmp_path)
        assert mgr.is_available("got_ocr2") is False

    def test_partial_files_is_unavailable(self, tmp_path: Path) -> None:
        mgr = ModelManager(models_dir=tmp_path)
        target = mgr.model_dir("got_ocr2")
        target.mkdir(parents=True)
        # Drop one file from the manifest, leave the rest missing
        (target / GOT_OCR2_SPEC.files[0].name).write_bytes(b"x" * 2048)
        assert mgr.is_available("got_ocr2") is False

    def test_complete_files_are_available(self, tmp_path: Path, monkeypatch) -> None:
        # Use a tiny synthetic spec instead of the real GOT-OCR2 manifest
        from src.infrastructure import model_manager as mm

        spec = _spec_for_test("file:///stub")
        monkeypatch.setitem(mm._REGISTRY, spec.model_id, spec)
        mgr = ModelManager(models_dir=tmp_path)
        target = mgr.model_dir(spec.model_id)
        target.mkdir(parents=True)
        for f in spec.files:
            (target / f.name).write_bytes(b"x" * f.size_bytes)
        assert mgr.is_available(spec.model_id) is True
        assert spec.model_id in mgr.list_local_models()


# ---------------------------------------------------------------------------
# Download (over a local HTTP server, no real network)
# ---------------------------------------------------------------------------


class TestDownload:
    def test_download_streams_files_with_progress(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Prepare a server directory with our two fake "model" files
        srv_dir = tmp_path / "server"
        srv_dir.mkdir()
        (srv_dir / "weights.bin").write_bytes(b"\x42" * 32)
        (srv_dir / "config.json").write_bytes(b"{}\n" + b" " * 5)  # 8 bytes

        url_base, server = _serve_dir(srv_dir)
        try:
            from src.infrastructure import model_manager as mm

            spec = _spec_for_test(url_base)
            monkeypatch.setitem(mm._REGISTRY, spec.model_id, spec)

            events: list[tuple[int, int, str]] = []
            mgr = ModelManager(models_dir=tmp_path / "user")
            target = mgr.download(
                spec.model_id,
                progress_callback=lambda d, t, n: events.append((d, t, n)),
            )

            # Files exist and match
            assert (target / "weights.bin").stat().st_size == 32
            assert (target / "config.json").stat().st_size == 8
            assert mgr.is_available(spec.model_id)

            # We got at least one progress callback per file
            files_seen = {ev[2] for ev in events}
            assert files_seen == {"weights.bin", "config.json"}
            # Final cumulative progress should reach the grand total
            assert events[-1][0] >= 40
        finally:
            server.shutdown()

    def test_size_mismatch_raises(self, tmp_path: Path, monkeypatch) -> None:
        srv_dir = tmp_path / "server"
        srv_dir.mkdir()
        # Spec expects 32 bytes but server returns 4
        (srv_dir / "weights.bin").write_bytes(b"abcd")
        (srv_dir / "config.json").write_bytes(b"x" * 8)

        url_base, server = _serve_dir(srv_dir)
        try:
            from src.infrastructure import model_manager as mm

            spec = _spec_for_test(url_base)
            monkeypatch.setitem(mm._REGISTRY, spec.model_id, spec)

            mgr = ModelManager(models_dir=tmp_path / "user")
            with pytest.raises(ModelDownloadError) as exc_info:
                mgr.download(spec.model_id)
            assert "weights.bin" in str(exc_info.value) or "проверку" in str(exc_info.value)
            # The .part file must be cleaned up
            assert not list((tmp_path / "user" / spec.model_id).glob("*.part"))
        finally:
            server.shutdown()

    def test_cancel_aborts(self, tmp_path: Path, monkeypatch) -> None:
        srv_dir = tmp_path / "server"
        srv_dir.mkdir()
        (srv_dir / "weights.bin").write_bytes(b"\x00" * 32)
        (srv_dir / "config.json").write_bytes(b"x" * 8)
        url_base, server = _serve_dir(srv_dir)
        try:
            from src.infrastructure import model_manager as mm

            spec = _spec_for_test(url_base)
            monkeypatch.setitem(mm._REGISTRY, spec.model_id, spec)
            mgr = ModelManager(models_dir=tmp_path / "user")
            with pytest.raises(ModelDownloadError):
                mgr.download(spec.model_id, cancel=lambda: True)
        finally:
            server.shutdown()

    def test_existing_valid_files_skipped(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        srv_dir = tmp_path / "server"
        srv_dir.mkdir()
        (srv_dir / "weights.bin").write_bytes(b"\x42" * 32)
        (srv_dir / "config.json").write_bytes(b"x" * 8)
        url_base, server = _serve_dir(srv_dir)
        try:
            from src.infrastructure import model_manager as mm

            spec = _spec_for_test(url_base)
            monkeypatch.setitem(mm._REGISTRY, spec.model_id, spec)

            mgr = ModelManager(models_dir=tmp_path / "user")
            mgr.download(spec.model_id)
            target = mgr.model_dir(spec.model_id)
            mtime_first = (target / "weights.bin").stat().st_mtime_ns

            # Second call must be a no-op for already-valid files
            mgr.download(spec.model_id)
            mtime_second = (target / "weights.bin").stat().st_mtime_ns
            assert mtime_first == mtime_second
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_returns_false_when_absent(self, tmp_path: Path) -> None:
        mgr = ModelManager(models_dir=tmp_path)
        assert mgr.remove("got_ocr2") is False

    def test_remove_deletes_directory(self, tmp_path: Path) -> None:
        mgr = ModelManager(models_dir=tmp_path)
        target = mgr.model_dir("got_ocr2")
        target.mkdir(parents=True)
        (target / "x").write_text("x")
        assert mgr.remove("got_ocr2") is True
        assert not target.exists()
