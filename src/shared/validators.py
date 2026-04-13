"""Validation helpers for paths, configs, and user input."""

from __future__ import annotations

from pathlib import Path

from src.shared.constants import MAX_DPI, MAX_PARALLEL_WORKERS, MIN_DPI, MIN_PARALLEL_WORKERS


class ValidationError(ValueError):
    """Raised when validation fails."""


def validate_pdf_path(path: Path | str) -> Path:
    """Ensure the path points to an existing, readable PDF.

    Args:
        path: File path to validate.

    Returns:
        Resolved absolute Path.

    Raises:
        ValidationError: If the file does not exist or is not a PDF.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValidationError(f"Файл не существует: {p}")
    if not p.is_file():
        raise ValidationError(f"Не является файлом: {p}")
    if p.suffix.lower() != ".pdf":
        raise ValidationError(f"Не PDF-файл: {p}")
    if p.stat().st_size == 0:
        raise ValidationError(f"Пустой файл: {p}")
    return p


def validate_output_path(path: Path | str) -> Path:
    """Ensure the parent directory exists for an output file."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def validate_dpi(dpi: int) -> int:
    """Validate DPI is within accepted range."""
    if not isinstance(dpi, int):
        raise ValidationError(f"DPI должен быть целым числом, получено: {type(dpi).__name__}")
    if not (MIN_DPI <= dpi <= MAX_DPI):
        raise ValidationError(f"DPI должен быть в диапазоне [{MIN_DPI}, {MAX_DPI}], получено: {dpi}")
    return dpi


def validate_workers(workers: int) -> int:
    """Validate parallel worker count."""
    if not (MIN_PARALLEL_WORKERS <= workers <= MAX_PARALLEL_WORKERS):
        raise ValidationError(
            f"Количество воркеров должно быть в [{MIN_PARALLEL_WORKERS}, {MAX_PARALLEL_WORKERS}]"
        )
    return workers


def validate_confidence(value: float) -> float:
    """Validate confidence threshold in [0, 100]."""
    if not (0.0 <= value <= 100.0):
        raise ValidationError(f"Confidence должен быть в [0, 100], получено: {value}")
    return float(value)


def validate_languages(langs: list[str]) -> list[str]:
    """Validate non-empty list of language codes."""
    if not langs:
        raise ValidationError("Список языков не может быть пустым")
    valid = {"rus", "eng"}
    for lang in langs:
        if lang not in valid:
            raise ValidationError(f"Неподдерживаемый язык: {lang}")
    return langs


def validate_odd_int(value: int, name: str = "value", min_val: int = 3, max_val: int = 99) -> int:
    """Validate an odd positive integer in range (used for kernel sizes)."""
    if value < min_val or value > max_val:
        raise ValidationError(f"{name} должен быть в [{min_val}, {max_val}], получено: {value}")
    if value % 2 == 0:
        raise ValidationError(f"{name} должен быть нечётным, получено: {value}")
    return value
