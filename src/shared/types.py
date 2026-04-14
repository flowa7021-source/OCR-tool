"""Enums and type aliases used throughout the application."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import TypeAlias

import numpy as np

# Type aliases
ImageArray: TypeAlias = np.ndarray  # OpenCV/numpy image
AngleDegrees: TypeAlias = float
FilePath: TypeAlias = str  # stringified pathlib.Path for serialization


class PSM(IntEnum):
    """Tesseract Page Segmentation Mode.

    See https://tesseract-ocr.github.io/tessdoc/tess3/ControlParams.html
    """

    OSD_ONLY = 0
    AUTO_OSD = 1
    AUTO_ONLY = 2
    AUTO = 3
    SINGLE_COLUMN = 4
    SINGLE_BLOCK_VERT = 5
    SINGLE_BLOCK = 6
    SINGLE_LINE = 7
    SINGLE_WORD = 8
    CIRCLE_WORD = 9
    SINGLE_CHAR = 10
    SPARSE_TEXT = 11
    SPARSE_TEXT_OSD = 12
    RAW_LINE = 13

    @property
    def label(self) -> str:
        """Human-readable label."""
        return _PSM_LABELS[self]

    @property
    def description(self) -> str:
        """Detailed description for tooltips."""
        return _PSM_DESCRIPTIONS[self]


_PSM_LABELS: dict[PSM, str] = {
    PSM.OSD_ONLY: "0 — Только определение ориентации (OSD)",
    PSM.AUTO_OSD: "1 — Авто-сегментация с OSD",
    PSM.AUTO_ONLY: "2 — Авто-сегментация без OSD и OCR",
    PSM.AUTO: "3 — Полная авто-сегментация (по умолчанию)",
    PSM.SINGLE_COLUMN: "4 — Один столбец текста",
    PSM.SINGLE_BLOCK_VERT: "5 — Единый вертикальный блок",
    PSM.SINGLE_BLOCK: "6 — Единый однородный блок",
    PSM.SINGLE_LINE: "7 — Одна строка текста",
    PSM.SINGLE_WORD: "8 — Одно слово",
    PSM.CIRCLE_WORD: "9 — Слово в круге",
    PSM.SINGLE_CHAR: "10 — Один символ",
    PSM.SPARSE_TEXT: "11 — Разреженный текст",
    PSM.SPARSE_TEXT_OSD: "12 — Разреженный текст с OSD",
    PSM.RAW_LINE: "13 — Raw line (без обработки)",
}

_PSM_DESCRIPTIONS: dict[PSM, str] = {
    PSM.OSD_ONLY: "Только определение ориентации страницы и скрипта. Без OCR.",
    PSM.AUTO_OSD: "Автоматическая сегментация с определением ориентации.",
    PSM.AUTO_ONLY: "Автоматическая сегментация без OSD и OCR.",
    PSM.AUTO: "Полностью автоматическая сегментация. Используется по умолчанию для большинства документов.",
    PSM.SINGLE_COLUMN: "Предполагает один столбец текста переменного размера.",
    PSM.SINGLE_BLOCK_VERT: "Единый вертикально выровненный блок текста.",
    PSM.SINGLE_BLOCK: "Единый однородный блок текста. Хорошо подходит для простых документов, договоров.",
    PSM.SINGLE_LINE: "Обрабатывает изображение как одну строку текста.",
    PSM.SINGLE_WORD: "Обрабатывает изображение как одно слово.",
    PSM.CIRCLE_WORD: "Одно слово, расположенное по окружности.",
    PSM.SINGLE_CHAR: "Обрабатывает изображение как один символ.",
    PSM.SPARSE_TEXT: "Находит максимум текста в произвольном порядке. Для чеков, таблиц, плакатов.",
    PSM.SPARSE_TEXT_OSD: "Разреженный текст с автоопределением ориентации.",
    PSM.RAW_LINE: "Обрабатывает как одну строку без специальной обработки.",
}


class OEM(IntEnum):
    """Tesseract OCR Engine Mode."""

    LEGACY_ONLY = 0
    LSTM_ONLY = 1
    LEGACY_LSTM = 2
    DEFAULT = 3

    @property
    def label(self) -> str:
        return _OEM_LABELS[self]

    @property
    def description(self) -> str:
        return _OEM_DESCRIPTIONS[self]


_OEM_LABELS: dict[OEM, str] = {
    OEM.LEGACY_ONLY: "0 — Legacy engine",
    OEM.LSTM_ONLY: "1 — LSTM neural net (рекомендуется)",
    OEM.LEGACY_LSTM: "2 — Legacy + LSTM",
    OEM.DEFAULT: "3 — Автовыбор движка",
}

_OEM_DESCRIPTIONS: dict[OEM, str] = {
    OEM.LEGACY_ONLY: "Старый движок на основе правил. Быстрее, но менее точный.",
    OEM.LSTM_ONLY: "Нейросетевой LSTM-движок. Наиболее точный для современных документов.",
    OEM.LEGACY_LSTM: "Комбинация обоих движков. Самый медленный.",
    OEM.DEFAULT: "Автовыбор на основе доступных данных.",
}


class Language(StrEnum):
    """Supported OCR languages (maps to tessdata filenames)."""

    RUSSIAN = "rus"
    ENGLISH = "eng"

    @property
    def label(self) -> str:
        return {"rus": "Русский", "eng": "English"}[self.value]


class BinarizationMethod(StrEnum):
    """Binarization algorithms."""

    NONE = "none"
    OTSU = "otsu"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    ADAPTIVE_MEAN = "adaptive_mean"
    SAUVOLA = "sauvola"


class DenoiseMethod(StrEnum):
    """Noise reduction algorithms (can be combined)."""

    MEDIAN = "median"
    GAUSSIAN = "gaussian"
    MORPH_OPEN = "morph_open"
    MORPH_CLOSE = "morph_close"
    NLM = "nlm"


class JobStatus(StrEnum):
    """Status of an OCR job in the queue."""

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
    """Supported export formats."""

    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    CLIPBOARD = "clipboard"


class OCREngineKind(StrEnum):
    """Selectable OCR back-end engines.

    * ``TESSERACT`` — the default LSTM-based engine from Tesseract 5.x,
      great at printed text in Russian and English, poor at handwriting.
    * ``GOT_OCR2`` — Stepfun GOT-OCR2.0 (2024): Apache-2.0 transformer
      model that handles printed and handwritten text in 80+ languages
      including Russian. Requires downloading a ~580 MB model weight
      before first use.

    The application ships with Tesseract bundled and advertises GOT-OCR2
    only when the optional ``htr`` extras + model weights are installed.
    """

    TESSERACT = "tesseract"
    GOT_OCR2 = "got_ocr2"

    @property
    def label(self) -> str:
        return _OCR_ENGINE_LABELS[self]

    @property
    def description(self) -> str:
        return _OCR_ENGINE_DESCRIPTIONS[self]


_OCR_ENGINE_LABELS: dict[OCREngineKind, str] = {
    OCREngineKind.TESSERACT: "Tesseract 5 (LSTM, печатный текст)",
    OCREngineKind.GOT_OCR2: "GOT-OCR 2.0 (рукописный + печатный)",
}

_OCR_ENGINE_DESCRIPTIONS: dict[OCREngineKind, str] = {
    OCREngineKind.TESSERACT: (
        "Встроенный Tesseract 5 с LSTM-моделями rus + eng. Быстрый и "
        "точный на печатном тексте, но не справляется с рукописью."
    ),
    OCREngineKind.GOT_OCR2: (
        "Transformer-модель GOT-OCR 2.0 (Stepfun, 2024). Распознаёт "
        "рукописный и печатный текст на 80+ языках, включая русский. "
        "Требует скачивания ~580 МБ модели и расширения htr."
    ),
}


class OptimizeLevel(IntEnum):
    """OCRmyPDF optimize parameter (0 = no optimization, 3 = maximum)."""

    NONE = 0
    LOSSLESS = 1
    LOSSY = 2
    AGGRESSIVE = 3
