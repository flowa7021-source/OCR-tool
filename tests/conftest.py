"""Shared pytest fixtures for the OCR-tool test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.models import (
    BinarizationConfig,
    ContrastConfig,
    DenoiseConfig,
    DenoiseStep,
    DeskewConfig,
    OCRConfig,
    PostprocessConfig,
    PreprocessConfig,
    ProfileData,
    RegexRule,
)
from src.shared.types import (
    BinarizationMethod,
    DenoiseMethod,
)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used across the suite."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (real model load / OCR)"
    )


# A minimal, valid single-page PDF.
_MINIMAL_PDF: bytes = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
    b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
    b"4 0 obj\n<< /Length 44 >>\nstream\n"
    b"BT /F1 24 Tf 100 700 Td (Hello World) Tj ET\n"
    b"endstream\nendobj\n"
    b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    b"xref\n"
    b"0 6\n"
    b"0000000000 65535 f \n"
    b"0000000009 00000 n \n"
    b"0000000056 00000 n \n"
    b"0000000109 00000 n \n"
    b"0000000213 00000 n \n"
    b"0000000308 00000 n \n"
    b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
    b"startxref\n379\n%%EOF\n"
)


@pytest.fixture
def tmp_profiles_dir(tmp_path: Path) -> Path:
    d = tmp_path / "profiles"
    d.mkdir()
    return d


@pytest.fixture
def tmp_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "config"
    d.mkdir()
    import src.infrastructure.config_storage as cs
    import src.shared.constants as const
    monkeypatch.setattr(cs, "CONFIG_DIR", d, raising=False)
    monkeypatch.setattr(const, "CONFIG_DIR", d, raising=False)
    return d


@pytest.fixture
def sample_profile() -> ProfileData:
    preprocess = PreprocessConfig(
        deskew=DeskewConfig(enabled=False, auto_detect=False, manual_angle=3.5),
        binarization=BinarizationConfig(
            method=BinarizationMethod.SAUVOLA,
            adaptive_block_size=25,
            sauvola_k=0.3,
        ),
        denoise=DenoiseConfig(
            enabled=True,
            steps=[
                DenoiseStep(method=DenoiseMethod.NLM, h=11),
                DenoiseStep(method=DenoiseMethod.MEDIAN, ksize=5),
            ],
        ),
        contrast=ContrastConfig(clahe_enabled=True, clahe_clip=3.5),
    )
    ocr = OCRConfig(
        languages=["en", "ru"],
        primary_language="en",
        dpi=400,
        confidence_threshold=75.0,
        skip_text=False,
    )
    postprocess = PostprocessConfig(
        autocorrect_russian=False,
        autocorrect_english=True,
        custom_rules=[
            RegexRule(pattern="foo", replacement="bar", is_regex=False),
            RegexRule(pattern=r"\d+", replacement="#", is_regex=True),
        ],
    )
    return ProfileData(
        name="sample_test_profile",
        description="Non-default sample profile for roundtrip tests",
        preprocess=preprocess,
        ocr=ocr,
        postprocess=postprocess,
        builtin=False,
    )


@pytest.fixture
def minimal_pdf_bytes() -> bytes:
    return _MINIMAL_PDF


@pytest.fixture
def fake_pdf_path(tmp_path: Path, minimal_pdf_bytes: bytes) -> Path:
    p = tmp_path / "sample.pdf"
    p.write_bytes(minimal_pdf_bytes)
    return p
