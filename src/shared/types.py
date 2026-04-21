"""Enums and type aliases used throughout the application."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import TypeAlias

import numpy as np

ImageArray: TypeAlias = np.ndarray
AngleDegrees: TypeAlias = float
FilePath: TypeAlias = str


class Language(StrEnum):
    """Supported OCR languages (EasyOCR codes)."""

    RUSSIAN = "ru"
    ENGLISH = "en"

    @property
    def label(self) -> str:
        return {"ru": "Русский", "en": "English"}[self.value]


class BinarizationMethod(StrEnum):
    NONE = "none"
    OTSU = "otsu"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    ADAPTIVE_MEAN = "adaptive_mean"
    SAUVOLA = "sauvola"


class DenoiseMethod(StrEnum):
    MEDIAN = "median"
    GAUSSIAN = "gaussian"
    MORPH_OPEN = "morph_open"
    MORPH_CLOSE = "morph_close"
    NLM = "nlm"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def label(self) -> str:
        return _JOB_STATUS_LABELS[self]

    @property
    def icon(self) -> str:
        return _JOB_STATUS_ICONS[self]


_JOB_STATUS_LABELS: dict[JobStatus, str] = {
    JobStatus.PENDING: "Ожидание",
    JobStatus.RUNNING: "Обработка",
    JobStatus.PAUSED: "Пауза",
    JobStatus.COMPLETED: "Завершено",
    JobStatus.FAILED: "Ошибка",
    JobStatus.CANCELLED: "Отменено",
}

_JOB_STATUS_ICONS: dict[JobStatus, str] = {
    JobStatus.PENDING: "⏳",
    JobStatus.RUNNING: "⚙️",
    JobStatus.PAUSED: "⏸️",
    JobStatus.COMPLETED: "✅",
    JobStatus.FAILED: "❌",
    JobStatus.CANCELLED: "⛔",
}


class ExportFormat(StrEnum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    CLIPBOARD = "clipboard"
    EXCEL = "excel"


class OCREngineKind(StrEnum):
    """Selectable OCR back-end engines.

    EasyOCR is the current implementation (replaced Tesseract in
    the April 2026 migration — see CHANGELOG). The enum stays a
    value-based type so profile JSON migrations from older schemas
    can map ``"tesseract"`` onto ``EASYOCR`` at load time.
    """

    EASYOCR = "easyocr"

    @property
    def label(self) -> str:
        return _OCR_ENGINE_LABELS[self]

    @property
    def description(self) -> str:
        return _OCR_ENGINE_DESCRIPTIONS[self]


_OCR_ENGINE_LABELS: dict[OCREngineKind, str] = {
    OCREngineKind.EASYOCR: "EasyOCR (нейронная модель, ru + en)",
}

_OCR_ENGINE_DESCRIPTIONS: dict[OCREngineKind, str] = {
    OCREngineKind.EASYOCR: (
        "Нейросетевой движок EasyOCR (CRAFT-детектор + CRNN-распознаватель) "
        "с русской и английской моделями. Поддерживает CPU и GPU."
    ),
}
