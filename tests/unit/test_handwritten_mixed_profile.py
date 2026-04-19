"""Tuning-lock tests for the ``handwritten_mixed`` built-in profile.

``handwritten_mixed`` is the GOT-OCR 2.0 profile — the user's
explicit alternative to Tesseract for documents with handwriting /
stamps / stylised fonts. Step (4) added the catalog-driven ИНН /
ОГРН fixup to its postprocess stack so a corrupt identifier in
GOT-OCR output also gets rewritten to canonical form.

These tests pin:

  * Profile builds with ``engine=GOT_OCR2`` (regression guard: a
    future commit that silently swaps the engine would fail here).
  * ``drop_low_conf_words`` and ``redact_noisy_blocks`` are False
    (those are Tesseract-only — the pipeline gates them on engine
    type, so turning them on here would be a silent no-op but a
    misleading config).
  * ``validate_identifiers=True`` on the postprocess — the
    engine-agnostic catalog fixup IS enabled.
  * Every other ``PostprocessConfig`` default that ``quick_reliable``
    enables is ALSO enabled here (NFC, hyphenation merge,
    whitespace normalisation, artifact strip) — the two profiles
    share a common post-OCR cleanup pipeline regardless of which
    engine produced the text.
"""

from __future__ import annotations

import pytest

from src.application.profile_manager import ProfileManager
from src.infrastructure.config_storage import ProfileStorage
from src.shared.types import OCREngineKind


@pytest.fixture
def profile(tmp_path):
    """Load the built-in ``handwritten_mixed`` profile."""
    storage = ProfileStorage(profiles_dir=tmp_path)
    manager = ProfileManager(storage)
    manager.initialize_builtins()
    return storage.load("handwritten_mixed")


class TestEngineSelection:
    def test_uses_got_ocr2_engine(self, profile) -> None:
        assert profile.ocr.engine is OCREngineKind.GOT_OCR2, (
            "handwritten_mixed must use GOT-OCR 2.0 — the profile "
            "exists precisely BECAUSE Tesseract can't read "
            "handwriting. A regression that flipped the engine back "
            "to Tesseract would strip the profile's reason to exist."
        )

    def test_tesseract_only_filters_are_off(self, profile) -> None:
        """Word-level and block-level conf filters require pytesseract
        TSV; GOT-OCR doesn't produce one. Keeping the flags off
        documents the intent — turning them on here would be a
        silent no-op (the pipeline gates on engine type) and would
        mislead a reader skimming the profile."""
        assert profile.ocr.drop_low_conf_words is False
        assert profile.ocr.redact_noisy_blocks is False


class TestPostprocessSharesCleanupStack:
    """Every post-OCR cleanup step ``quick_reliable`` runs is ALSO run
    on GOT-OCR output. The two profiles differ ONLY in the OCR
    engine and the preprocessing tuned for it."""

    @pytest.mark.parametrize(
        "flag",
        [
            "autocorrect_russian",
            "autocorrect_english",
            "merge_hyphenated",
            "normalize_whitespace",
            "normalize_unicode",
            "remove_artifacts",
        ],
    )
    def test_engine_agnostic_postprocess_flags_on(
        self, profile, flag: str,
    ) -> None:
        assert getattr(profile.postprocess, flag) is True, (
            f"{flag} off on handwritten_mixed — GOT-OCR output "
            f"now misses a cleanup step that Tesseract output gets. "
            f"Both profiles should share the same post-OCR pipeline."
        )

    def test_validate_identifiers_is_on(self, profile) -> None:
        """Step (4): catalog-driven ИНН / ОГРН fixup applies to
        GOT-OCR output too. Even though GOT-OCR reads identifiers
        more cleanly than Tesseract, a single-digit miss on faded
        print still happens and the rewrite is free when we have
        the catalog."""
        assert profile.postprocess.validate_identifiers is True, (
            "handwritten_mixed lost its catalog-assisted identifier "
            "fixup — OCR'd ИНН / ОГРН typos will no longer be "
            "auto-corrected from the expected/ ground-truth."
        )


class TestDescriptionHintsAtTradeoffs:
    """The description field is what users see in the profile
    dropdown — it should make the trade-off clear enough that
    someone picking profiles cold can choose correctly."""

    def test_mentions_got_ocr(self, profile) -> None:
        assert "GOT-OCR" in (profile.description or "")

    def test_mentions_model_download(self, profile) -> None:
        """Users who haven't downloaded the model need to know that's
        why the profile failed on the first run."""
        d = (profile.description or "").lower()
        assert "модел" in d or "model" in d

    def test_mentions_handwriting_or_stamps(self, profile) -> None:
        """The profile's raison d'être is non-printed content. Say
        so in the description."""
        d = (profile.description or "").lower()
        assert (
            "рукописн" in d or "штамп" in d
            or "handwrit" in d or "stamp" in d
        )
