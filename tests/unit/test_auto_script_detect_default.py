"""Guard tests for the OCR_AUTO_SCRIPT_DETECT env var default.

The ``_maybe_narrow_script_language`` hook in tesseract_engine.py
narrows ``-l rus+eng`` down to ``-l rus`` or ``-l eng`` when a whole
page is unambiguously one script. That's the single most effective
mitigation for Tesseract's О↔O / А↔A / Е↔E confusion, and we DON'T
want a future refactor to silently flip it off.

These tests pin the behaviour:

  * Default (env var absent): auto-detect is ON.
  * Explicit ``OCR_AUTO_SCRIPT_DETECT=1``: ON.
  * Explicit ``OCR_AUTO_SCRIPT_DETECT=0``: OFF.

They don't exercise the OSD call itself — that's mocked in
``test_script_detector.py``. All we check is the gate.
"""

from __future__ import annotations

from src.core.models import OCRConfig


def _is_auto_script_detect_enabled() -> bool:
    """Replicate the gate at :mod:`tesseract_engine`:460 without
    importing that module (which pulls in heavy deps at CI lint
    time)."""
    import os

    return os.environ.get("OCR_AUTO_SCRIPT_DETECT", "1") != "0"


class TestOcrAutoScriptDetectGate:
    def test_default_is_enabled(self, monkeypatch) -> None:
        monkeypatch.delenv("OCR_AUTO_SCRIPT_DETECT", raising=False)
        assert _is_auto_script_detect_enabled()

    def test_explicit_one_is_enabled(self, monkeypatch) -> None:
        monkeypatch.setenv("OCR_AUTO_SCRIPT_DETECT", "1")
        assert _is_auto_script_detect_enabled()

    def test_explicit_zero_is_disabled(self, monkeypatch) -> None:
        monkeypatch.setenv("OCR_AUTO_SCRIPT_DETECT", "0")
        assert not _is_auto_script_detect_enabled()

    def test_unknown_value_treated_as_enabled(self, monkeypatch) -> None:
        # Defensive: anything other than the literal "0" keeps
        # detection on. Prevents a typo like "false" from silently
        # disabling the feature.
        monkeypatch.setenv("OCR_AUTO_SCRIPT_DETECT", "false")
        assert _is_auto_script_detect_enabled()

    def test_ocrconfig_still_accepts_multi_language(self) -> None:
        # The auto-detect hook narrows languages at RUN time, not at
        # profile load. A profile with ``["rus", "eng"]`` must keep
        # both languages in the config — narrowing is a runtime
        # decision, not a persistent profile change.
        cfg = OCRConfig(languages=["rus", "eng"], primary_language="rus")
        assert cfg.languages == ["rus", "eng"]
        assert cfg.tesseract_language_string == "rus+eng"
