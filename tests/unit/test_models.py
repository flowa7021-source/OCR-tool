"""Tests for :mod:`src.core.models`."""

from __future__ import annotations

from src.core.models import OCRConfig, ProfileData, QueueItem
from src.shared.types import (
    BinarizationMethod,
    DenoiseMethod,
    OEM,
    PSM,
)


def test_profile_roundtrip(sample_profile: ProfileData) -> None:
    """Profile → dict → Profile round-trip preserves structure."""
    data = sample_profile.to_dict()
    rebuilt = ProfileData.from_dict(data)
    assert rebuilt.to_dict() == data


def test_ocr_language_string_primary_first() -> None:
    """Primary language must appear first in the Tesseract language string."""
    cfg = OCRConfig(languages=["rus", "eng"], primary_language="rus")
    assert cfg.tesseract_language_string == "rus+eng"

    cfg = OCRConfig(languages=["rus", "eng"], primary_language="eng")
    assert cfg.tesseract_language_string == "eng+rus"


def test_queue_item_progress_pct() -> None:
    """Progress percentage: 0/0 → 0; 5/10 → 50; 20/10 → capped at 100."""
    item = QueueItem()
    assert item.progress_pct == 0.0

    item.progress_current = 5
    item.progress_total = 10
    assert item.progress_pct == 50.0

    item.progress_current = 20
    item.progress_total = 10
    assert item.progress_pct == 100.0


def test_enum_deserialization(sample_profile: ProfileData) -> None:
    """Enum fields are reconstructed from their serialized string/int values."""
    sample_profile.preprocess.binarization.method = BinarizationMethod.SAUVOLA
    if sample_profile.preprocess.denoise.steps:
        sample_profile.preprocess.denoise.steps[0].method = DenoiseMethod.NLM

    data = sample_profile.to_dict()
    rebuilt = ProfileData.from_dict(data)

    assert isinstance(rebuilt.ocr.psm, PSM)
    assert isinstance(rebuilt.ocr.oem, OEM)
    assert isinstance(rebuilt.preprocess.binarization.method, BinarizationMethod)
    if rebuilt.preprocess.denoise.steps:
        assert isinstance(
            rebuilt.preprocess.denoise.steps[0].method, DenoiseMethod
        )


def test_queue_item_file_name_no_config() -> None:
    """file_name should fall back to a placeholder when config is None."""
    assert QueueItem().file_name == "(unknown)"
