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

    def test_got_ocr2_spec_includes_trust_remote_code_modules(self) -> None:
        """GOT-OCR 2.0 needs 4 Python modules to load via trust_remote_code.

        Regression: a production build shipped a manifest that was missing
        ``tokenization_qwen.py`` + siblings. AutoTokenizer then failed with
        ``OSError: ... does not appear to have a file named
        tokenization_qwen.py`` and the job crashed with a message pointing
        at the temp path under ``huggingface.co/C:\\Users\\...``. Each of
        these four modules lives at the HF repo root and is executed by
        ``trust_remote_code=True`` at load time, so every one MUST be in
        the manifest — otherwise the installer / runtime downloader skips
        it and the model directory is "complete" per is_available() but
        broken at inference time.
        """
        required_py_modules = {
            "tokenization_qwen.py",  # custom Qwen tokenizer class
            "modeling_GOT.py",       # main GOT model architecture
            "got_vision_b.py",       # vision encoder backbone
            "render_tools.py",       # helpers used by ocr_type='format'
        }
        manifest_names = {f.name for f in GOT_OCR2_SPEC.files}
        missing = required_py_modules - manifest_names
        assert not missing, (
            f"GOT-OCR 2.0 manifest is missing trust_remote_code modules: "
            f"{sorted(missing)}. Every .py file at the HF repo root must "
            "be listed or AutoTokenizer/AutoModel.from_pretrained crashes."
        )

    def test_got_ocr2_spec_urls_point_at_huggingface(self) -> None:
        """Every manifest URL must resolve to the stepfun-ai HF repo.

        A regression where a copy-paste accident pointed one entry at a
        different repo would produce a subtle corrupt-download failure
        (size check passes, hash fails at load time). Easier to catch
        at manifest-definition time with a URL prefix assertion.
        """
        expected_prefix = "https://huggingface.co/stepfun-ai/GOT-OCR2_0/resolve/main/"
        for f in GOT_OCR2_SPEC.files:
            assert f.url.startswith(expected_prefix), (
                f"{f.name!r} points at {f.url!r} — expected prefix "
                f"{expected_prefix!r}"
            )

    def test_unknown_id_raises(self, tmp_path: Path) -> None:
        mgr = ModelManager(models_dir=tmp_path)
        with pytest.raises(KeyError):
            mgr.spec_for("does-not-exist")

    def test_py_modules_have_min_size_floor(self) -> None:
        """The trust_remote_code .py modules must carry a ``min_size_bytes``
        floor — otherwise a truncated 200-byte download (HTTP 206
        interrupted, disk full mid-write, or an HTML error page
        accidentally served by a proxy) passes the default ``1024``
        floor AND the ``size_bytes==0 → no check`` path, leaving a
        corrupt module on disk that crashes the engine at load.

        Regression for the Log 3 failure class: the user hit an
        uncaught OSError from ``AutoTokenizer.from_pretrained`` because
        ``tokenization_qwen.py`` was missing; the NEXT shape of that
        same failure class would be a present-but-truncated file. This
        test guards against that NEXT shape.
        """
        py_modules = {
            f.name: f
            for f in GOT_OCR2_SPEC.files
            if f.name.endswith(".py")
        }
        assert py_modules, "expected trust_remote_code .py files in manifest"
        for name, spec in py_modules.items():
            assert spec.min_size_bytes > 0, (
                f"{name} has no min_size_bytes — truncated downloads "
                "would silently pass validation"
            )
            # Sanity: floor must NOT exceed the actual file on disk
            # (known from the HF listing). Otherwise validation would
            # reject all legitimate downloads.
            known_max = {
                "tokenization_qwen.py": 9_700,
                "modeling_GOT.py": 34_600,
                "got_vision_b.py": 16_500,
                "render_tools.py": 2_040,
            }
            if name in known_max:
                assert spec.min_size_bytes <= known_max[name], (
                    f"{name} min_size_bytes {spec.min_size_bytes} > "
                    f"actual {known_max[name]} — would reject a real download"
                )


class TestValidateFile:
    """Regression tests for the three-layer validation in _validate_file."""

    def test_tiny_file_below_min_size_floor_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """A .py file truncated to 300 bytes fails the min_size_bytes floor
        even though ``size_bytes=0`` would skip the exact-size check."""
        from src.infrastructure.model_manager import ModelFile, ModelManager

        spec = ModelFile(
            name="tokenization_qwen.py",
            url="file:///ignored",
            size_bytes=0,  # no exact-size check
            min_size_bytes=5 * 1024,  # 5 KB floor
        )
        truncated = tmp_path / "tokenization_qwen.py"
        truncated.write_bytes(b"# truncated by proxy\n" * 10)  # ~210 B
        assert ModelManager._validate_file(truncated, spec) is False

    def test_file_at_or_above_min_size_floor_is_accepted(
        self, tmp_path: Path
    ) -> None:
        from src.infrastructure.model_manager import ModelFile, ModelManager

        spec = ModelFile(
            name="tokenization_qwen.py",
            url="file:///ignored",
            size_bytes=0,
            min_size_bytes=5 * 1024,
        )
        ok = tmp_path / "tokenization_qwen.py"
        ok.write_bytes(b"x" * (6 * 1024))
        assert ModelManager._validate_file(ok, spec) is True

    def test_legacy_no_min_size_bytes_uses_1kb_default(
        self, tmp_path: Path
    ) -> None:
        """``min_size_bytes=0`` keeps the original 1 KB default floor."""
        from src.infrastructure.model_manager import ModelFile, ModelManager

        spec = ModelFile(
            name="config.json",
            url="file:///ignored",
            size_bytes=0,
            min_size_bytes=0,  # use default
        )
        # 900 bytes is below the default 1024 floor.
        small = tmp_path / "small.json"
        small.write_bytes(b"x" * 900)
        assert ModelManager._validate_file(small, spec) is False
        # 1100 bytes passes.
        big = tmp_path / "big.json"
        big.write_bytes(b"x" * 1100)
        assert ModelManager._validate_file(big, spec) is True


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
