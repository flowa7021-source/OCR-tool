"""Tests that the OCRmyPDF wrapper wires user-words / user-patterns.

Stage D of Initiative 1: when ``OCRConfig.use_user_dictionaries`` is
true (the default), the Russian user-words and user-patterns files
that live under ``resources/tessdata/`` must be passed to
``ocrmypdf.ocr`` as the ``user_words=`` and ``user_patterns=`` kwargs.

When disabled — or when the bundled files are missing — the call
must still succeed without those kwargs, so the pipeline degrades
gracefully on a misconfigured deployment instead of aborting with
``FileNotFoundError``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.application.ocrmypdf_integration import (
    OCRmyPDFOptions,
    map_ocr_config,
    run_ocrmypdf,
)
from src.core.models import OCRConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_input_pdf(tmp_path: Path) -> Path:
    p = tmp_path / "in.pdf"
    p.write_bytes(b"%PDF-1.7\n")
    return p


@pytest.fixture
def fake_output_pdf(tmp_path: Path) -> Path:
    return tmp_path / "out.pdf"


def _make_options(
    cfg: OCRConfig,
    in_pdf: Path,
    out_pdf: Path,
) -> OCRmyPDFOptions:
    return map_ocr_config(cfg, in_pdf, out_pdf)


def _fake_ocrmypdf_modules(tmp_path: Path):
    """Return (fake_ocrmypdf, fake_exceptions) with a success write.

    ``ocrmypdf.ocr`` is wired to write a minimal non-empty text layer
    into ``out_pdf`` so the no-text retry path in
    ``_invoke_ocrmypdf_with_timeout_retry`` does not kick in.
    """

    def _write_non_empty_output(input_f: str, output_f: str, **kw: object) -> None:
        import fitz

        doc = fitz.open()
        try:
            page = doc.new_page(width=200, height=200)
            page.insert_text((10, 50), "ok", fontsize=12)
            doc.save(str(output_f))
        finally:
            doc.close()

    fake_ocrmypdf = MagicMock()
    fake_ocrmypdf.ocr = MagicMock(side_effect=_write_non_empty_output)

    class _FakeExitCodeError(Exception):
        exit_code = 0

    fake_exceptions = MagicMock()
    fake_exceptions.ExitCodeException = _FakeExitCodeError
    return fake_ocrmypdf, fake_exceptions


# ---------------------------------------------------------------------------
# OCRConfig field
# ---------------------------------------------------------------------------


class TestOCRConfigUseUserDictionariesField:
    def test_default_is_true(self) -> None:
        """By default the Russian dictionary/patterns should be on.

        Russian is the primary audience; the files are small; the
        accuracy win on tax IDs, dates, and entity abbreviations is
        significant.
        """
        cfg = OCRConfig()
        assert cfg.use_user_dictionaries is True

    def test_can_be_disabled(self) -> None:
        cfg = OCRConfig(use_user_dictionaries=False)
        assert cfg.use_user_dictionaries is False


# ---------------------------------------------------------------------------
# Wiring into ocrmypdf.ocr
# ---------------------------------------------------------------------------


class TestUserWordsWiring:
    """When enabled, paths are forwarded as kwargs to ocrmypdf.ocr."""

    def test_enabled_adds_user_words_and_user_patterns_kwargs(
        self, fake_input_pdf: Path, fake_output_pdf: Path, tmp_path: Path
    ) -> None:
        cfg = OCRConfig(
            languages=["rus", "eng"],
            primary_language="rus",
            use_user_dictionaries=True,
        )
        options = _make_options(cfg, fake_input_pdf, fake_output_pdf)

        fake_ocrmypdf, fake_exceptions = _fake_ocrmypdf_modules(tmp_path)
        with patch.dict(
            sys.modules,
            {
                "ocrmypdf": fake_ocrmypdf,
                "ocrmypdf.exceptions": fake_exceptions,
            },
        ):
            run_ocrmypdf(options)

        assert fake_ocrmypdf.ocr.call_count == 1
        call = fake_ocrmypdf.ocr.call_args
        assert "user_words" in call.kwargs, (
            "ocrmypdf.ocr should have received user_words=<path> when "
            "OCRConfig.use_user_dictionaries=True and primary_language=rus"
        )
        assert "user_patterns" in call.kwargs, (
            "ocrmypdf.ocr should have received user_patterns=<path> when "
            "OCRConfig.use_user_dictionaries=True and primary_language=rus"
        )
        # Paths must resolve to real files (the bundled resources).
        user_words_path = Path(call.kwargs["user_words"])
        user_patterns_path = Path(call.kwargs["user_patterns"])
        assert user_words_path.is_file()
        assert user_patterns_path.is_file()
        assert user_words_path.name == "user-words.rus"
        assert user_patterns_path.name == "user-patterns.rus"

    def test_disabled_omits_user_words_and_user_patterns_kwargs(
        self, fake_input_pdf: Path, fake_output_pdf: Path, tmp_path: Path
    ) -> None:
        cfg = OCRConfig(
            languages=["rus", "eng"],
            primary_language="rus",
            use_user_dictionaries=False,
        )
        options = _make_options(cfg, fake_input_pdf, fake_output_pdf)

        fake_ocrmypdf, fake_exceptions = _fake_ocrmypdf_modules(tmp_path)
        with patch.dict(
            sys.modules,
            {
                "ocrmypdf": fake_ocrmypdf,
                "ocrmypdf.exceptions": fake_exceptions,
            },
        ):
            run_ocrmypdf(options)

        call = fake_ocrmypdf.ocr.call_args
        assert "user_words" not in call.kwargs, (
            "user_words must NOT be passed when the feature is disabled"
        )
        assert "user_patterns" not in call.kwargs, (
            "user_patterns must NOT be passed when the feature is disabled"
        )

    def test_graceful_degradation_when_files_missing(
        self,
        fake_input_pdf: Path,
        fake_output_pdf: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If the bundled files are missing, the call still runs + logs.

        Scenario: an end-user extracted only part of the installer, or
        a custom build dropped the ``resources/tessdata/user-*`` files.
        The pipeline must not crash — it should just log a warning and
        skip the user-dict kwargs.
        """
        # Redirect the user-words lookup at an empty temp dir so the
        # integration module can't find the real bundled files.
        empty_tessdata = tmp_path / "empty-tessdata"
        empty_tessdata.mkdir()
        import src.application.ocrmypdf_integration as mod

        monkeypatch.setattr(mod, "TESSDATA_DIR", empty_tessdata, raising=False)

        cfg = OCRConfig(
            languages=["rus", "eng"],
            primary_language="rus",
            use_user_dictionaries=True,
        )
        options = _make_options(cfg, fake_input_pdf, fake_output_pdf)

        fake_ocrmypdf, fake_exceptions = _fake_ocrmypdf_modules(tmp_path)
        with (
            patch.dict(
                sys.modules,
                {
                    "ocrmypdf": fake_ocrmypdf,
                    "ocrmypdf.exceptions": fake_exceptions,
                },
            ),
            caplog.at_level(logging.WARNING, logger=mod.__name__),
        ):
            run_ocrmypdf(options)  # must NOT raise

        call = fake_ocrmypdf.ocr.call_args
        assert "user_words" not in call.kwargs
        assert "user_patterns" not in call.kwargs
        # And we surfaced the degradation.
        assert any(
            "user-words" in rec.message or "user-patterns" in rec.message
            for rec in caplog.records
        ), (
            "Expected a WARNING-level log about the missing user-dict "
            "files when graceful degradation kicks in."
        )

    def test_english_primary_does_not_send_russian_files(
        self, fake_input_pdf: Path, fake_output_pdf: Path, tmp_path: Path
    ) -> None:
        """ocrmypdf.ocr only accepts ONE user_words path per run.

        When primary_language is 'eng' we must forward the English
        files (or none) — NOT the Russian ones — to avoid polluting
        the dictionary with Cyrillic tokens for an English document.
        """
        cfg = OCRConfig(
            languages=["eng", "rus"],
            primary_language="eng",
            use_user_dictionaries=True,
        )
        options = _make_options(cfg, fake_input_pdf, fake_output_pdf)

        fake_ocrmypdf, fake_exceptions = _fake_ocrmypdf_modules(tmp_path)
        with patch.dict(
            sys.modules,
            {
                "ocrmypdf": fake_ocrmypdf,
                "ocrmypdf.exceptions": fake_exceptions,
            },
        ):
            run_ocrmypdf(options)

        call = fake_ocrmypdf.ocr.call_args
        # If we wire English too, they must be the .eng variant; if we
        # only wire Russian, they must be absent. Either way, the
        # Russian files must NOT be sent on an English-primary run.
        uw = call.kwargs.get("user_words")
        up = call.kwargs.get("user_patterns")
        if uw is not None:
            assert Path(uw).name == "user-words.eng", (
                f"English-primary OCR should not use Russian user-words: {uw!r}"
            )
        if up is not None:
            assert Path(up).name == "user-patterns.eng", (
                f"English-primary OCR should not use Russian user-patterns: {up!r}"
            )


# ---------------------------------------------------------------------------
# ``tesseract_thresholding`` kwarg shape
# ---------------------------------------------------------------------------


class TestTesseractThresholdingKwargType:
    """Regression guard: ``tesseract_thresholding`` MUST be a string from
    ocrmypdf's allowed set, NOT a bare int.

    ocrmypdf 16.x declares the API typed-dict slot as ``int | None`` but
    internally calls ``str(value)`` and re-parses through argparse with
    ``choices=('auto', 'otsu', 'adaptive-otsu', 'sauvola')``. Passing
    ``0`` crashes every OCR call on 16.13+ with

        argparse.ArgumentTypeError: '0' must be one of:
        auto, otsu, adaptive-otsu, sauvola

    which is exactly the failure the install smoke-test surfaced. The
    ocrmypdf mock doesn't validate choices, so this test asserts the
    kwarg SHAPE rather than functional correctness: a string from the
    valid set round-trips through ocrmypdf argparse; an int never does.
    """

    _VALID = {"auto", "otsu", "adaptive-otsu", "sauvola"}

    def test_thresholding_is_string_from_valid_set(
        self, fake_input_pdf: Path, fake_output_pdf: Path, tmp_path: Path,
    ) -> None:
        cfg = OCRConfig(languages=["rus"], primary_language="rus")
        options = _make_options(cfg, fake_input_pdf, fake_output_pdf)

        fake_ocrmypdf, fake_exceptions = _fake_ocrmypdf_modules(tmp_path)
        with patch.dict(
            sys.modules,
            {
                "ocrmypdf": fake_ocrmypdf,
                "ocrmypdf.exceptions": fake_exceptions,
            },
        ):
            run_ocrmypdf(options)

        call = fake_ocrmypdf.ocr.call_args
        value = call.kwargs.get("tesseract_thresholding")
        assert value is None or value in self._VALID, (
            "tesseract_thresholding must be one of "
            f"{self._VALID} or absent; got {value!r} "
            f"(type={type(value).__name__}). "
            "ocrmypdf 16.x rejects bare ints — see the docstring above."
        )
        assert not isinstance(value, bool) and not isinstance(value, int), (
            f"tesseract_thresholding must NOT be an int; got {value!r}"
        )
