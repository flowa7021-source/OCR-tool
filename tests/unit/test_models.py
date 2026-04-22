"""OCRConfig + ProfileData round-trip and v12→v13 schema migration."""

from __future__ import annotations

from src.core.models import PROFILE_SCHEMA_VERSION, OCRConfig, ProfileData
from src.shared.types import OCREngineKind


def test_default_ocr_config_uses_easyocr() -> None:
    cfg = OCRConfig()
    assert cfg.engine is OCREngineKind.EASYOCR
    assert cfg.languages == ["ru", "en"]
    assert cfg.primary_language == "ru"
    assert cfg.dpi == 300
    assert cfg.gpu is False
    assert 0.0 <= cfg.min_keep_confidence <= 1.0


def test_profile_round_trip(sample_profile: ProfileData) -> None:
    data = sample_profile.to_dict()
    rebuilt = ProfileData.from_dict(data)
    assert rebuilt.name == sample_profile.name
    assert rebuilt.ocr.languages == sample_profile.ocr.languages
    assert rebuilt.ocr.primary_language == sample_profile.ocr.primary_language
    assert rebuilt.ocr.dpi == sample_profile.ocr.dpi
    assert rebuilt.schema_version == PROFILE_SCHEMA_VERSION


def test_v12_profile_migrates_to_easyocr() -> None:
    """A v12 profile with Tesseract fields is rewritten to v13."""
    legacy = {
        "name": "old",
        "schema_version": 12,
        "ocr": {
            "engine": "tesseract",
            "languages": ["rus", "eng"],
            "primary_language": "rus",
            "psm": 3,
            "oem": 1,
            "tesseract_timeout": 360,
            "char_whitelist": "0123456789",
            "use_user_dictionaries": True,
            "extra_tesseract_params": {"preserve_interword_spaces": "1"},
            "per_word_clahe_rescue": True,
            "optimize_level": 1,
            "dpi": 400,
            "confidence_threshold": 65.0,
        },
    }
    profile = ProfileData.from_dict(legacy)
    assert profile.schema_version == PROFILE_SCHEMA_VERSION
    assert profile.ocr.engine is OCREngineKind.EASYOCR
    assert profile.ocr.languages == ["ru", "en"]
    assert profile.ocr.primary_language == "ru"
    assert profile.ocr.allowlist == "0123456789"
    assert profile.ocr.dpi == 400
    assert profile.ocr.confidence_threshold == 65.0
    # Removed fields must NOT be present on the rebuilt config.
    for dead in ("psm", "oem", "tesseract_timeout", "char_whitelist",
                 "use_user_dictionaries", "extra_tesseract_params",
                 "per_word_clahe_rescue", "optimize_level"):
        assert not hasattr(profile.ocr, dead), f"{dead} leaked through"


def test_unknown_engine_value_falls_back_to_default() -> None:
    """Old / unknown engine string must not break profile loading."""
    profile = ProfileData.from_dict({
        "name": "weird",
        "schema_version": PROFILE_SCHEMA_VERSION,
        "ocr": {"engine": "some_unknown", "languages": ["ru"]},
    })
    assert profile.ocr.engine is OCREngineKind.EASYOCR
