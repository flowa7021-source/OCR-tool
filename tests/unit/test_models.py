"""Tests for :mod:`src.core.models`."""

from __future__ import annotations

from src.core.models import OCRConfig, ProfileData, QueueItem
from src.shared.types import (
    OEM,
    PSM,
    BinarizationMethod,
    DenoiseMethod,
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


def test_unknown_enum_value_in_profile_falls_back_to_default() -> None:
    """A JSON profile with an unknown enum member must load, not crash.

    Regression: ``_convert_value`` used to catch only ``TypeError``
    when coercing enum members. ``target_type(value)`` actually raises
    ``ValueError`` when the value is hashable but not a member of the
    enum — the natural failure mode for a downgraded binary or a
    hand-edited JSON (``"engine": "got_ocr3"``). The exception
    propagated uncaught and the whole profile failed to load, which
    in turn broke ``ProfileManager.get_current()`` and left the UI's
    profile dropdown empty.

    The new behaviour: unknown enum values are logged and dropped
    from the kwargs, so the dataclass falls back to its own default
    for that single field. The rest of the profile loads cleanly.
    """
    profile_json = {
        "name": "tweaked",
        "schema_version": 1,
        "ocr": {
            # Valid fields around an invalid one — we want to prove
            # the rest survives.
            "languages": ["rus", "eng"],
            "primary_language": "rus",
            "dpi": 400,
            # The hostile value.
            "engine": "got_ocr_9999_not_a_real_engine",
        },
    }

    # Must NOT raise.
    rebuilt = ProfileData.from_dict(profile_json)

    # The bad enum reverted to the OCRConfig default (TESSERACT).
    from src.shared.types import OCREngineKind

    assert rebuilt.ocr.engine is OCREngineKind.TESSERACT
    # Fields that were valid on the same dataclass still landed.
    assert rebuilt.ocr.dpi == 400
    assert rebuilt.ocr.languages == ["rus", "eng"]
    assert rebuilt.name == "tweaked"


def test_unknown_int_enum_value_also_falls_back() -> None:
    """``PSM`` / ``OEM`` / ``OptimizeLevel`` are IntEnums — same rule."""
    profile_json = {
        "name": "int-enum-tweak",
        "schema_version": 1,
        "ocr": {
            "psm": 99,  # no such PSM
            "oem": 7,   # no such OEM
            "optimize_level": 42,  # no such optimize level
            "dpi": 300,
        },
    }

    rebuilt = ProfileData.from_dict(profile_json)
    # Each bad int reverted to the OCRConfig default.
    assert rebuilt.ocr.psm is PSM.AUTO
    assert rebuilt.ocr.oem is OEM.LSTM_ONLY
    # dpi was valid — so it still applied.
    assert rebuilt.ocr.dpi == 300


def test_valid_enum_value_still_loads_correctly() -> None:
    """Guard against the ValueError-catching fix accidentally eating
    legitimate enum members."""
    profile_json = {
        "name": "clean",
        "schema_version": 1,
        "ocr": {
            "engine": "tesseract",
            "psm": 6,
            "oem": 1,
            "optimize_level": 2,
        },
    }
    rebuilt = ProfileData.from_dict(profile_json)

    from src.shared.types import OCREngineKind, OptimizeLevel

    assert rebuilt.ocr.engine is OCREngineKind.TESSERACT
    assert rebuilt.ocr.psm is PSM.SINGLE_BLOCK
    assert rebuilt.ocr.oem is OEM.LSTM_ONLY
    assert rebuilt.ocr.optimize_level is OptimizeLevel.LOSSY
