"""Tests for the universal_accurate preset + convenience helpers.

Covers:

* ``preprocess_universal(image)`` runs the full pipeline end-to-end
  and returns a binary image + skew angle — zero configuration.
* ``postprocess_universal(text)`` applies every cleanup step in one call.
* ``UNIVERSAL_PREPROCESS_CONFIG`` / ``UNIVERSAL_POSTPROCESS_CONFIG``
  enable the expected set of options (guard against a future refactor
  that silently disables, say, deskew or autocorrect).
* The ``universal_accurate`` builtin profile is created by
  ``ProfileManager.initialize_builtins()`` with the correct settings.
* MainWindow selects ``universal_accurate`` on first launch.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# --------------------------------------------------------------------------
# preprocess_universal
# --------------------------------------------------------------------------


class TestPreprocessUniversal:
    def test_returns_binary_image_and_angle(self) -> None:
        import numpy as np

        from src.core.image_preprocessor import preprocess_universal

        # 400x600 greyscale synthetic page with a dark stripe.
        img = np.full((400, 600, 3), 240, dtype=np.uint8)
        img[190:210, 50:550] = 30  # horizontal text-ish stripe

        out, angle = preprocess_universal(img)
        assert isinstance(out, np.ndarray)
        # Binarised → only two values (0 and 255), possibly one near-absent.
        unique_vals = set(np.unique(out).tolist())
        assert unique_vals.issubset({0, 255}), (
            f"universal preset must binarise; got pixel values {unique_vals}"
        )
        # Non-trivial output: must contain both foreground and background.
        assert 0 in unique_vals and 255 in unique_vals
        assert isinstance(angle, float)

    def test_config_enables_expected_steps(self) -> None:
        from src.core.image_preprocessor import UNIVERSAL_PREPROCESS_CONFIG
        from src.shared.types import BinarizationMethod, DenoiseMethod

        cfg = UNIVERSAL_PREPROCESS_CONFIG
        assert cfg.deskew.enabled
        assert cfg.deskew.auto_detect
        assert cfg.contrast.clahe_enabled
        # background removal is intentionally OFF in the universal
        # preset: at 600 DPI the large-kernel blur is the single most
        # expensive preprocessing step and gives only marginal accuracy
        # gain. low_quality_scan is the right pick for heavy page tint.
        assert not cfg.background.enabled
        assert cfg.denoise.enabled
        # The denoise chain must preserve thin strokes — multi-step
        # median + morph close is the design intent.
        methods = [s.method for s in cfg.denoise.steps]
        assert DenoiseMethod.MEDIAN in methods
        assert DenoiseMethod.MORPH_CLOSE in methods
        # Adaptive, not OTSU.
        assert cfg.binarization.method == BinarizationMethod.ADAPTIVE_GAUSSIAN

    def test_preprocessor_is_cached_across_calls(self) -> None:
        """The module-level ImagePreprocessor singleton should be reused."""
        import numpy as np

        import src.core.image_preprocessor as ip

        ip._UNIVERSAL_PREPROCESSOR = None
        img = np.full((200, 200), 200, dtype=np.uint8)
        ip.preprocess_universal(img)
        first = ip._UNIVERSAL_PREPROCESSOR
        assert first is not None
        ip.preprocess_universal(img)
        assert ip._UNIVERSAL_PREPROCESSOR is first


# --------------------------------------------------------------------------
# postprocess_universal
# --------------------------------------------------------------------------


class TestPostprocessUniversal:
    def test_config_enables_all_cleanup_steps(self) -> None:
        from src.core.text_postprocessor import UNIVERSAL_POSTPROCESS_CONFIG

        cfg = UNIVERSAL_POSTPROCESS_CONFIG
        assert cfg.normalize_unicode
        assert cfg.merge_hyphenated
        assert cfg.normalize_whitespace
        assert cfg.remove_artifacts
        assert cfg.autocorrect_russian
        assert cfg.autocorrect_english
        # No user rules by default.
        assert cfg.custom_rules == []

    def test_merges_hyphenated_and_normalizes_whitespace(self) -> None:
        from src.core.text_postprocessor import postprocess_universal

        # Hyphenated linebreak + excessive whitespace.
        raw = "при-\nвет    мир\n\n\n\nещё строчка  "
        out = postprocess_universal(raw)
        assert "привет" in out  # hyphenation merged
        assert "    " not in out  # runs of spaces collapsed
        # Don't collapse all blank lines — 2 max is the design.
        assert "\n\n\n\n" not in out

    def test_empty_input_is_a_noop(self) -> None:
        from src.core.text_postprocessor import postprocess_universal

        assert postprocess_universal("") == ""


# --------------------------------------------------------------------------
# ProfileManager builtin wiring
# --------------------------------------------------------------------------


class TestUniversalAccurateProfile:
    def test_initialize_builtins_creates_universal_accurate(
        self, tmp_path: Path
    ) -> None:
        from src.application.profile_manager import BUILTIN_NAMES, ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        assert "universal_accurate" in BUILTIN_NAMES
        storage = ProfileStorage(profiles_dir=tmp_path)
        manager = ProfileManager(storage)
        manager.initialize_builtins()

        profile = storage.load("universal_accurate")
        assert profile.builtin is True
        # Key preprocessing toggles match the universal preset.
        assert profile.preprocess.deskew.enabled
        assert profile.preprocess.contrast.clahe_enabled
        # Background removal is ON by design — it flattens scanner-
        # lamp gradients before CLAHE + binarisation, which gives
        # measurable accuracy wins on real scans. The ~500 ms per
        # page cost is acceptable for the "max accuracy" preset.
        assert profile.preprocess.background.enabled
        assert len(profile.preprocess.denoise.steps) >= 2
        # Every postprocess step on — including the word-level
        # Cyrillic/Latin look-alike fixup, which is critical for
        # Russian documents Tesseract OCRs with rus+eng.
        for flag in (
            "autocorrect_russian", "autocorrect_english",
            "merge_hyphenated", "normalize_whitespace",
            "normalize_unicode", "remove_artifacts",
            "fix_cyrillic_latin_confusion",
        ):
            assert getattr(profile.postprocess, flag) is True, flag
        # The "maximum accuracy" preset requires ≥ 500 DPI as of
        # Initiative 4 (bumped from 400 → 500). Dense small-font
        # Russian body text measured 25 % CER at 400 DPI on the
        # nightly benchmark; 500 DPI gives the LSTM more pixel
        # density per character without hitting Tesseract's
        # layout-analysis timeout ceiling.
        assert profile.ocr.dpi >= 500, (
            f"universal_accurate must request ≥500 DPI; got {profile.ocr.dpi}"
        )

    def test_description_hints_at_purpose(self, tmp_path: Path) -> None:
        """Sanity: description mentions accuracy / universal so users find it."""
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path)
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = storage.load("universal_accurate")
        text = (profile.description or "").lower()
        assert "универсал" in text or "accurate" in text

    # ------------------------------------------------------------------
    # Step 1 tuning lock: word-confidence filter. Step 2 (Apr 2026)
    # also touched Sauvola window / border_removal / garbage_filter
    # but was reverted — those numbers were speculative and hurt real-
    # document OCR (51.5 % → 44 % mean_confidence on the user's
    # transport-invoice scan). Any future change to those knobs MUST
    # be justified by a before/after run of
    # ``scripts/benchmark_universal.py`` before a lock-in test is
    # added here.
    # ------------------------------------------------------------------

    def test_drops_low_conf_words(self, tmp_path: Path) -> None:
        """Word-level confidence filter ON. Without this, every
        10-40%-conf stamp / signature guess ends up in the user-
        visible text and mean_confidence reads catastrophically
        low even when body-text accuracy is fine. See
        :mod:`src.core.confidence_filter` for the mechanism."""
        from src.application.profile_manager import ProfileManager
        from src.infrastructure.config_storage import ProfileStorage

        storage = ProfileStorage(profiles_dir=tmp_path)
        manager = ProfileManager(storage)
        manager.initialize_builtins()
        profile = storage.load("universal_accurate")
        assert profile.ocr.drop_low_conf_words is True, (
            "Step 1 regression: universal_accurate lost its word-"
            "confidence filter, results panel will refill with "
            "stamp/signature noise."
        )


# --------------------------------------------------------------------------
# MainWindow first-run selection
# --------------------------------------------------------------------------


class TestMainWindowDefaultProfile:
    def _stub_heavy(self, monkeypatch, tmp_path: Path) -> None:
        import src.shared.constants as constants

        for name in ("USER_DATA_DIR", "CONFIG_DIR", "PROFILES_DIR", "TEMP_DIR",
                     "LOGS_DIR", "RECOVERY_DIR", "OCR_CACHE_DIR"):
            monkeypatch.setattr(constants, name, tmp_path / name.lower())
            (tmp_path / name.lower()).mkdir(exist_ok=True)

        from src.infrastructure.tesseract_wrapper import TesseractWrapper

        monkeypatch.setattr(TesseractWrapper, "verify", lambda self: (True, "ok"))

        from PySide6.QtWidgets import QMessageBox

        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: None)

    def test_universal_accurate_selected_by_default(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        from PySide6.QtWidgets import QApplication

        self._stub_heavy(monkeypatch, tmp_path)
        QApplication.instance() or QApplication([])
        from src.app import create_application

        _, window = create_application([])
        try:
            # The combobox stores profile name in userData.
            assert window.profile_combo.currentData() == "universal_accurate"
        finally:
            window.close()
            window.deleteLater()

    def test_falls_back_to_index_zero_when_preset_missing(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """If someone strips universal_accurate from BUILTIN_NAMES the UI
        must still pick something sane, not crash with IndexError."""
        from PySide6.QtWidgets import QApplication

        self._stub_heavy(monkeypatch, tmp_path)
        QApplication.instance() or QApplication([])

        # Remove the preset from the manager before create_application.
        import src.application.profile_manager as pm

        monkeypatch.setattr(pm, "BUILTIN_NAMES", tuple(
            n for n in pm.BUILTIN_NAMES if n != "universal_accurate"
        ))

        original_init = pm.ProfileManager.initialize_builtins

        def stripped_init(self) -> None:
            original_init(self)
            # Also remove the file if it was seeded before patch applied.
            path = self.storage.profiles_dir / "universal_accurate.json"
            if path.exists():
                path.unlink()

        monkeypatch.setattr(pm.ProfileManager, "initialize_builtins", stripped_init)

        from src.app import create_application

        _, window = create_application([])
        try:
            # Some profile is selected, not None.
            assert window.profile_combo.currentData() is not None
        finally:
            window.close()
            window.deleteLater()
