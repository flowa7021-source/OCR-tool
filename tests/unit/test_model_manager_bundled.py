"""Tests for bundled model-weight fallback in ModelManager.

After we started shipping GOT-OCR 2.0 weights inside the installer,
``ModelManager`` gained a second lookup root (``bundled_dir``). These
tests lock in the expected behaviour:

* If the user-writable copy is complete → use it (supports user-driven
  re-download / override).
* Else if the bundled copy is complete → use that (installer bundle
  makes HTR work out of the box).
* Else neither works and :meth:`is_available` is False.
"""

from __future__ import annotations

from pathlib import Path


def _write_manifest(root: Path) -> None:
    """Create every file listed in :data:`GOT_OCR2_SPEC.files` under ``root``."""
    from src.infrastructure.model_manager import GOT_OCR2_SPEC

    model_dir = root / GOT_OCR2_SPEC.model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    for fobj in GOT_OCR2_SPEC.files:
        (model_dir / fobj.name).write_bytes(b"x" * 2048)  # plausible size


class TestBundledFallback:
    def test_bundled_only_is_available(self, tmp_path: Path) -> None:
        from src.infrastructure.model_manager import GOT_OCR2_SPEC, ModelManager

        user = tmp_path / "user"
        bundled = tmp_path / "bundled"
        _write_manifest(bundled)

        mgr = ModelManager(models_dir=user, bundled_dir=bundled)
        assert mgr.is_available(GOT_OCR2_SPEC.model_id)
        assert mgr.model_dir(GOT_OCR2_SPEC.model_id) == bundled / GOT_OCR2_SPEC.model_id

    def test_user_copy_wins_over_bundled(self, tmp_path: Path) -> None:
        from src.infrastructure.model_manager import GOT_OCR2_SPEC, ModelManager

        user = tmp_path / "user"
        bundled = tmp_path / "bundled"
        _write_manifest(user)
        _write_manifest(bundled)

        mgr = ModelManager(models_dir=user, bundled_dir=bundled)
        assert mgr.model_dir(GOT_OCR2_SPEC.model_id) == user / GOT_OCR2_SPEC.model_id

    def test_neither_complete_is_not_available(self, tmp_path: Path) -> None:
        from src.infrastructure.model_manager import GOT_OCR2_SPEC, ModelManager

        mgr = ModelManager(
            models_dir=tmp_path / "user",
            bundled_dir=tmp_path / "bundled",
        )
        assert not mgr.is_available(GOT_OCR2_SPEC.model_id)

    def test_partial_user_falls_through_to_bundled(self, tmp_path: Path) -> None:
        """User dir with SOME files falls through to complete bundled dir."""
        from src.infrastructure.model_manager import GOT_OCR2_SPEC, ModelManager

        user = tmp_path / "user"
        bundled = tmp_path / "bundled"
        _write_manifest(bundled)

        # Plant just one file in the user dir.
        partial_dir = user / GOT_OCR2_SPEC.model_id
        partial_dir.mkdir(parents=True)
        (partial_dir / GOT_OCR2_SPEC.files[0].name).write_bytes(b"oops")

        mgr = ModelManager(models_dir=user, bundled_dir=bundled)
        assert mgr.is_available(GOT_OCR2_SPEC.model_id)
        # Should resolve to the bundled location since user dir is incomplete.
        assert mgr.model_dir(GOT_OCR2_SPEC.model_id) == bundled / GOT_OCR2_SPEC.model_id


class TestDownloadScriptStructure:
    def test_download_script_exists_and_references_manifest(self) -> None:
        script = (
            Path(__file__).parent.parent.parent
            / ".github" / "scripts" / "download_got_ocr2.py"
        )
        assert script.exists()
        src = script.read_text(encoding="utf-8")
        # Must pull the manifest from the same source as the runtime —
        # the whole point is to avoid drift.
        assert "GOT_OCR2_SPEC" in src
        assert "src.infrastructure.model_manager" in src


class TestSpecContentsMatchRepoReality:
    """Guard the GOT-OCR 2.0 file manifest against 404 regressions.

    The HuggingFace repo ``stepfun-ai/GOT-OCR2_0`` does NOT ship a
    ``tokenizer.json`` (it uses a tiktoken-format ``qwen.tiktoken``
    instead). Downloading ``tokenizer.json`` returns 404 and kills
    the CI "Pre-download GOT-OCR 2.0 weights" step mid-flight.
    """

    def test_no_tokenizer_json_in_manifest(self) -> None:
        from src.infrastructure.model_manager import GOT_OCR2_SPEC

        names = {f.name for f in GOT_OCR2_SPEC.files}
        assert "tokenizer.json" not in names, (
            "tokenizer.json is NOT present in the stepfun-ai/GOT-OCR2_0 "
            "repo — listing it here causes a 404 mid-download. The real "
            "tokenizer ships as qwen.tiktoken + tokenizer_config.json "
            "+ special_tokens_map.json."
        )

    def test_qwen_tiktoken_is_in_manifest(self) -> None:
        """And the thing that IS the tokenizer must stay listed."""
        from src.infrastructure.model_manager import GOT_OCR2_SPEC

        names = {f.name for f in GOT_OCR2_SPEC.files}
        assert "qwen.tiktoken" in names

    def test_model_weights_and_configs_are_listed(self) -> None:
        """Sanity baseline: the files GOT-OCR 2.0 actually needs to run."""
        from src.infrastructure.model_manager import GOT_OCR2_SPEC

        names = {f.name for f in GOT_OCR2_SPEC.files}
        required = {
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
            "qwen.tiktoken",
        }
        missing = required - names
        assert not missing, f"Missing required files: {missing}"
