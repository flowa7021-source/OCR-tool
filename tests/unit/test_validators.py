"""Tests for :mod:`src.shared.validators`."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.validators import (
    ValidationError,
    validate_confidence,
    validate_dpi,
    validate_languages,
    validate_odd_int,
    validate_pdf_path,
    validate_workers,
)


class TestValidatePdfPath:
    def test_happy_path(self, fake_pdf_path: Path) -> None:
        result = validate_pdf_path(fake_pdf_path)
        assert result == fake_pdf_path.resolve()

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            validate_pdf_path(tmp_path / "does_not_exist.pdf")

    def test_not_a_pdf(self, tmp_path: Path) -> None:
        p = tmp_path / "note.txt"
        p.write_text("hi")
        with pytest.raises(ValidationError):
            validate_pdf_path(p)

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.pdf"
        p.write_bytes(b"")
        with pytest.raises(ValidationError):
            validate_pdf_path(p)

    def test_directory_not_file(self, tmp_path: Path) -> None:
        d = tmp_path / "subdir.pdf"
        d.mkdir()
        with pytest.raises(ValidationError):
            validate_pdf_path(d)


class TestValidateDpi:
    @pytest.mark.parametrize("dpi", [150, 300, 600])
    def test_valid(self, dpi: int) -> None:
        assert validate_dpi(dpi) == dpi

    @pytest.mark.parametrize("dpi", [0, 149, 601, 1200])
    def test_out_of_range(self, dpi: int) -> None:
        with pytest.raises(ValidationError):
            validate_dpi(dpi)

    def test_wrong_type(self) -> None:
        with pytest.raises(ValidationError):
            validate_dpi("300")  # type: ignore[arg-type]


class TestValidateWorkers:
    @pytest.mark.parametrize("workers", [1, 2, 4])
    def test_valid(self, workers: int) -> None:
        assert validate_workers(workers) == workers

    @pytest.mark.parametrize("workers", [0, 5, -1])
    def test_invalid(self, workers: int) -> None:
        with pytest.raises(ValidationError):
            validate_workers(workers)


class TestValidateConfidence:
    @pytest.mark.parametrize("val", [0.0, 50.0, 100.0])
    def test_valid(self, val: float) -> None:
        assert validate_confidence(val) == val

    @pytest.mark.parametrize("val", [-0.1, 100.1, 200.0])
    def test_invalid(self, val: float) -> None:
        with pytest.raises(ValidationError):
            validate_confidence(val)


class TestValidateLanguages:
    def test_valid(self) -> None:
        assert validate_languages(["rus", "eng"]) == ["rus", "eng"]
        assert validate_languages(["rus"]) == ["rus"]

    def test_empty(self) -> None:
        with pytest.raises(ValidationError):
            validate_languages([])

    def test_unsupported_language(self) -> None:
        with pytest.raises(ValidationError):
            validate_languages(["deu"])

    def test_mixed_unsupported(self) -> None:
        with pytest.raises(ValidationError):
            validate_languages(["rus", "fra"])


class TestValidateOddInt:
    @pytest.mark.parametrize("val", [3, 5, 31, 99])
    def test_valid(self, val: int) -> None:
        assert validate_odd_int(val) == val

    def test_even_rejected(self) -> None:
        with pytest.raises(ValidationError):
            validate_odd_int(4)

    def test_below_range(self) -> None:
        with pytest.raises(ValidationError):
            validate_odd_int(1)

    def test_above_range(self) -> None:
        with pytest.raises(ValidationError):
            validate_odd_int(101)

    def test_custom_range(self) -> None:
        assert validate_odd_int(7, min_val=5, max_val=9) == 7
        with pytest.raises(ValidationError):
            validate_odd_int(11, min_val=5, max_val=9)
