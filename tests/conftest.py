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
    OEM,
    PSM,
    BinarizationMethod,
    DenoiseMethod,
    OptimizeLevel,
)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used across the suite.

    ``exercise_preflight`` opts tests out of the autouse bypass of
    ``verify_required_for_ocrmypdf`` — those tests want to exercise
    the pre-flight logic against real / mocked registries directly.
    """
    config.addinivalue_line(
        "markers",
        "exercise_preflight: test wants the real external-tools "
        "pre-flight check, not the auto-mocked one",
    )

# A minimal, valid single-page PDF (approximately 400 bytes). Rendered blank.
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
    """Provide an empty directory suitable for :class:`ProfileStorage`."""
    d = tmp_path / "profiles"
    d.mkdir()
    return d


@pytest.fixture
def tmp_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a fresh directory for :class:`SettingsStorage`.

    Also monkeypatches the module-level ``CONFIG_DIR`` to the temp location so
    tests that depend on the default path behave deterministically.
    """
    d = tmp_path / "config"
    d.mkdir()
    import src.infrastructure.config_storage as cs
    import src.shared.constants as const

    monkeypatch.setattr(cs, "CONFIG_DIR", d, raising=False)
    monkeypatch.setattr(const, "CONFIG_DIR", d, raising=False)
    return d


@pytest.fixture
def sample_profile() -> ProfileData:
    """Return a :class:`ProfileData` populated with non-default values."""
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
        languages=["eng", "rus"],
        primary_language="eng",
        psm=PSM.SINGLE_BLOCK,
        oem=OEM.LSTM_ONLY,
        dpi=400,
        confidence_threshold=75.0,
        optimize_level=OptimizeLevel.LOSSY,
        skip_text=False,
    )
    postprocess = PostprocessConfig(
        autocorrect_russian=False,
        autocorrect_english=True,
        merge_hyphenated=True,
        normalize_whitespace=True,
        normalize_unicode=True,
        remove_artifacts=False,
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
    """Bytes of a minimal, valid 1-page PDF."""
    return _MINIMAL_PDF


@pytest.fixture
def fake_pdf_path(tmp_path: Path, minimal_pdf_bytes: bytes) -> Path:
    """Write the minimal PDF to ``tmp_path`` and return the path."""
    p = tmp_path / "sample.pdf"
    p.write_bytes(minimal_pdf_bytes)
    return p


@pytest.fixture(autouse=True)
def _bypass_external_tools_preflight(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Treat OCRmyPDF external binaries as present in every test.

    Many tests stub the entire OCR pipeline (they patch
    ``run_ocrmypdf`` / ``ocrmypdf.ocr``) and care only about orchestration,
    not about whether ``tesseract``/``gs`` are actually installed on the
    test machine. Without this autouse bypass, the pre-flight check
    added to ``OCRPipeline.run`` short-circuits those tests with a
    FAILED job status because the CI / dev environment legitimately
    doesn't have Ghostscript on PATH.

    Tests that specifically want to exercise the pre-flight logic
    (see ``tests/unit/test_external_tools.py``) opt out by adding the
    ``exercise_preflight`` marker.
    """
    if "exercise_preflight" in request.keywords:
        return
    try:
        from src.infrastructure import external_tools as _ext
    except Exception:  # noqa: BLE001
        return
    monkeypatch.setattr(_ext, "verify_required_for_ocrmypdf", lambda: [])
