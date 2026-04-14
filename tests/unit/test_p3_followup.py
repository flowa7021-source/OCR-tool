"""Regression tests for the P3 audit-follow-up round.

Covers:

* Config import now rejects settings.json payloads that parse as JSON
  but contain **no** recognised AppSettings fields (previously they'd
  be silently accepted via ``AppSettings.from_dict`` falling back to
  all defaults).
* Pipeline's ``max_pages`` truncation log is a WARNING (so CLI users
  see it at the default verbosity), not INFO.
* Dead ``_rm_tree_best_effort`` helper is gone.
* ``release.yml`` no longer contains the placeholder "build" job;
  it invokes the reusable build-installer workflow.
"""

from __future__ import annotations

import json
import logging
import os
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------
# Tighter schema check on settings.json import
# --------------------------------------------------------------------------


class TestImportSettingsSchemaCheck:
    def _storages(self, tmp_path: Path):
        from src.infrastructure.config_storage import ProfileStorage, SettingsStorage

        return (
            SettingsStorage(config_dir=tmp_path / "cfg"),
            ProfileStorage(profiles_dir=tmp_path / "profiles"),
        )

    def test_rejects_json_without_any_known_field(self, tmp_path: Path) -> None:
        from src.application.config_backup import BackupFormatError, import_config

        path = tmp_path / "weird.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
            zf.writestr("settings.json", json.dumps({"greeting": "hi", "x": 1}))
        s, p = self._storages(tmp_path)
        with pytest.raises(BackupFormatError, match="распознаваемого поля"):
            import_config(path, s, p)

    def test_rejects_non_dict_settings(self, tmp_path: Path) -> None:
        from src.application.config_backup import BackupFormatError, import_config

        path = tmp_path / "list.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
            zf.writestr("settings.json", json.dumps([1, 2, 3]))
        s, p = self._storages(tmp_path)
        with pytest.raises(BackupFormatError, match="JSON-объектом"):
            import_config(path, s, p)

    def test_does_not_clobber_existing_settings_on_reject(
        self, tmp_path: Path
    ) -> None:
        """Confirm the import fails BEFORE the on-disk settings are touched."""
        from src.application.config_backup import BackupFormatError, import_config

        s, p = self._storages(tmp_path)
        # Pre-populate disk with a known-good setting.
        original = s.load()
        original.parallel_workers = 3
        original.theme = "light"
        s.save(original)

        path = tmp_path / "garbage.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
            zf.writestr("settings.json", '{"no_known_fields": true}')

        with pytest.raises(BackupFormatError):
            import_config(path, s, p)

        # Reload a fresh storage to bypass the in-memory cache.
        from src.infrastructure.config_storage import SettingsStorage

        after = SettingsStorage(config_dir=tmp_path / "cfg").load()
        assert after.parallel_workers == 3
        assert after.theme == "light"

    def test_accepts_settings_with_even_one_known_field(
        self, tmp_path: Path
    ) -> None:
        """A payload with `theme` alone is a valid backup (defaults fill the rest)."""
        from src.application.config_backup import import_config

        path = tmp_path / "min.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("manifest.json", json.dumps({"schema_version": 1}))
            zf.writestr("settings.json", json.dumps({"theme": "light"}))
        s, p = self._storages(tmp_path)
        result = import_config(path, s, p)
        assert result.settings_restored is True
        from src.infrastructure.config_storage import SettingsStorage

        assert SettingsStorage(config_dir=tmp_path / "cfg").load().theme == "light"


# --------------------------------------------------------------------------
# max_pages truncation now logs at WARNING
# --------------------------------------------------------------------------


class TestMaxPagesLogLevel:
    def test_truncation_logs_warning(self, tmp_path: Path, caplog) -> None:
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

        pdf = tmp_path / "big.pdf"
        doc = fitz.open()
        try:
            for i in range(4):
                p = doc.new_page(width=200, height=100)
                p.insert_text((10, 50), f"p{i}")
            doc.save(str(pdf))
        finally:
            doc.close()

        class _Stub:
            kind = None
            name = "stub"
            description = ""

            def is_available(self):
                return True, ""

            def run(self, preprocessed_pdf, output_pdf, config, progress_callback=None):
                import shutil

                import fitz as _fitz

                shutil.copy2(preprocessed_pdf, output_pdf)
                with _fitz.open(str(output_pdf)) as d:
                    return [
                        PageOCRResult(page_number=i + 1, text=f"p{i + 1}")
                        for i in range(d.page_count)
                    ]

            def unload(self):
                pass

        pre = PreprocessConfig()
        pre.binarization.method = BinarizationMethod.NONE
        pre.deskew.enabled = False
        profile = ProfileData(
            name="limited",
            ocr=OCRConfig(max_pages=2, dpi=72),
            preprocess=pre,
        )
        out = tmp_path / "out.pdf"
        job = OCRJobConfig(
            input_path=str(pdf), output_path=str(out), profile=profile
        )
        pipeline = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=TextPostprocessor(),
            tesseract=MagicMock(),
        )
        with caplog.at_level(logging.WARNING, logger="src.application.pipeline"), patch(
            "src.application.engines.get_engine", return_value=_Stub()
        ):
            result = pipeline.run(job)
        assert result.status is JobStatus.COMPLETED
        # The WARNING-level message must be present — CLI default verbosity
        # is WARNING so this is the minimum visibility bar.
        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "preview mode" in r.getMessage()
        ]
        assert warnings, caplog.text


# --------------------------------------------------------------------------
# Dead code removal
# --------------------------------------------------------------------------


class TestDeadCodeRemoval:
    def test_rm_tree_best_effort_is_gone(self) -> None:
        import src.application.config_backup as cb

        assert not hasattr(cb, "_rm_tree_best_effort")

    def test_no_stray_shutil_import(self) -> None:
        import src.application.config_backup as cb

        # It's legal to re-add shutil later, but right now it's dead.
        assert not hasattr(cb, "shutil")


# --------------------------------------------------------------------------
# Pipeline type contract
# --------------------------------------------------------------------------


class TestPipelineTypeHints:
    def test_accepts_none_postprocessor(self, tmp_path: Path) -> None:
        """The type hint is now ``TextPostprocessor | None``; None must work."""
        from unittest.mock import MagicMock

        from src.application.pipeline import OCRPipeline
        from src.core.image_preprocessor import ImagePreprocessor

        # Should construct without TypeError.
        p = OCRPipeline(
            preprocessor=ImagePreprocessor(),
            postprocessor=None,
            tesseract=MagicMock(),
        )
        # And _postprocess_text returns the input unchanged.
        assert p._postprocess_text("hello", None) == "hello"


# --------------------------------------------------------------------------
# release.yml structure
# --------------------------------------------------------------------------


class TestReleaseWorkflowStructure:
    def _load_release(self):
        yaml = pytest.importorskip("yaml")
        path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "release.yml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_build_job_uses_reusable_workflow(self) -> None:
        data = self._load_release()
        build_job = data["jobs"]["build"]
        assert "uses" in build_job, (
            "release.yml must delegate to build-installer.yml via "
            "`uses: ./.github/workflows/build-installer.yml`"
        )
        assert build_job["uses"].endswith("build-installer.yml")

    def test_build_has_no_placeholder_run_step(self) -> None:
        data = self._load_release()
        build_job = data["jobs"]["build"]
        # A reusable-workflow call has no 'steps' field.
        assert "steps" not in build_job

    def test_sign_downloads_build_artifact_by_name(self) -> None:
        data = self._load_release()
        sign_steps = data["jobs"]["sign"]["steps"]
        dl_step = next(
            s for s in sign_steps if "download-artifact" in s.get("uses", "")
        )
        name_expr = dl_step["with"]["name"]
        assert "needs.build.outputs.installer_artifact" in name_expr

    def test_build_installer_exposes_workflow_call(self) -> None:
        yaml = pytest.importorskip("yaml")
        path = Path(__file__).parent.parent.parent / ".github" / "workflows" / "build-installer.yml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        on = data.get("on", data.get(True))
        assert "workflow_call" in on, (
            "build-installer.yml must declare `workflow_call` so release.yml "
            "can invoke it"
        )
        outputs = on["workflow_call"].get("outputs", {})
        assert "installer_artifact" in outputs
