"""Unit tests for pipeline PDF-analysis edge cases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fitz")

from src.application.pipeline import (
    CorruptPdfError,
    EmptyPdfError,
    EncryptedPdfError,
    OCRPipeline,
)


def _make_pipeline() -> OCRPipeline:
    return OCRPipeline(
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        tesseract=MagicMock(),
    )


def test_corrupt_pdf_raises_typed_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a real pdf file at all")
    pipeline = _make_pipeline()
    with pytest.raises(CorruptPdfError):
        pipeline._analyze_pdf(bad)


def test_save_png_handles_unicode_path(tmp_path: Path) -> None:
    """Regression: ``cv2.imwrite`` fails on non-ASCII Windows paths.

    End-user logs showed ``RuntimeError: Не удалось сохранить PNG``
    for every page on a machine where the user profile was
    ``C:\\Users\\Т.Н. 020\\...``. Root cause: ``cv2.imwrite`` calls
    ``fopen`` internally, which on Windows uses the ANSI code page
    and cannot open files whose path contains characters outside it.
    Workaround: encode via ``cv2.imencode`` and write through
    :meth:`Path.write_bytes`, which is Unicode-aware.

    This test exercises the Unicode-path case on any platform — on
    Linux it simply verifies the new encode-then-write flow works;
    on Windows it's the actual regression coverage.
    """
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    pipeline = _make_pipeline()
    # Arbitrary small 3-channel image.
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    unicode_dir = tmp_path / "Т.Н. 020" / "subdir with space"
    target = unicode_dir / "page_00001.png"

    # Must not raise.
    pipeline._save_png(image, target)

    assert target.exists(), f"PNG not written to {target}"
    # File should be a valid PNG (magic bytes \x89PNG).
    header = target.read_bytes()[:4]
    assert header == b"\x89PNG", f"output is not a valid PNG, header={header!r}"
    # And round-trip: decode it back.
    raw = np.frombuffer(target.read_bytes(), dtype=np.uint8)
    decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    assert decoded is not None
    assert decoded.shape == image.shape


def test_empty_pdf_raises_empty_error(tmp_path: Path) -> None:
    """Simulate a PDF reporting zero pages by mocking fitz.open."""
    pipeline = _make_pipeline()
    fake_doc = MagicMock()
    fake_doc.page_count = 0
    fake_doc.needs_pass = False

    with patch("fitz.open", return_value=fake_doc), pytest.raises(EmptyPdfError):
        pipeline._analyze_pdf(tmp_path / "empty.pdf")
    fake_doc.close.assert_called_once()


def test_encrypted_pdf_without_password_raises(tmp_path: Path) -> None:
    pipeline = _make_pipeline()
    fake_doc = MagicMock()
    fake_doc.needs_pass = True
    fake_doc.authenticate.return_value = 0  # failure
    fake_doc.page_count = 5

    with patch("fitz.open", return_value=fake_doc), pytest.raises(EncryptedPdfError):
        pipeline._analyze_pdf(tmp_path / "secret.pdf")
    fake_doc.close.assert_called_once()


def test_encrypted_pdf_with_empty_password_succeeds(tmp_path: Path) -> None:
    pipeline = _make_pipeline()
    fake_page = MagicMock()
    fake_page.get_text.return_value = "hello"
    fake_doc = MagicMock()
    fake_doc.needs_pass = True
    fake_doc.authenticate.return_value = 1  # success
    fake_doc.page_count = 2
    fake_doc.load_page.return_value = fake_page

    with patch("fitz.open", return_value=fake_doc):
        infos = pipeline._analyze_pdf(tmp_path / "weak.pdf")

    assert len(infos) == 2
    assert infos[0]["page_number"] == 1
    assert infos[0]["has_text"] is True


def test_run_top_level_catches_typed_errors(tmp_path: Path) -> None:
    """A corrupt PDF goes through run() and produces JobStatus.FAILED."""
    from src.core.models import OCRJobConfig, ProfileData
    from src.shared.types import JobStatus

    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a real pdf file at all")
    out = tmp_path / "bad_ocr.pdf"

    tess = MagicMock()
    tess.configure_pytesseract.return_value = None
    pipeline = OCRPipeline(
        preprocessor=MagicMock(),
        postprocessor=MagicMock(),
        tesseract=tess,
    )
    job = OCRJobConfig(
        input_path=str(bad),
        output_path=str(out),
        profile=ProfileData(name="default"),
    )

    result = pipeline.run(job)
    assert result.status is JobStatus.FAILED
    assert "повреждён" in (result.error or "") or "повреж" in (result.error or "")
