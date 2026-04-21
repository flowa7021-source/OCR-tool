"""Per-field OCR retry (idea #6 top-10).

Pipeline retry-tier работает на уровне СТРАНИЦЫ (3 escalation-
уровня в tesseract_engine). Но когда проблема локализована —
раздел «6. Водитель» пустой или дал мусор, остальные поля ОК —
нужен targeted rescue: re-OCR только на bbox'e поля, с
specialized настройками (PSM=7 single-line, whitelist=Cyrillic,
user-words=FIO-lexicon).

Не заменяет pipeline retry — дополняет. Работает на already
processed PNG (preprocessed для OCR) без дополнительной
deskew/binarize.

Entry point — :func:`rescue_field`. Экспорт стэба для мокинга
(``_pytesseract_image_to_data``) — ключевой для unit-test'ов.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _crop_bbox(
    raster: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Безопасно вырезать регион из raster'а. ``None`` если bbox
    полностью вне границ или отрицательный start."""
    x, y, w, h = bbox
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    height, width = raster.shape[:2]
    if x >= width or y >= height:
        return None
    # Clip bbox к границам raster'а:
    x_end = min(x + w, width)
    y_end = min(y + h, height)
    return raster[y:y_end, x:x_end]


def _pytesseract_image_to_data(img, **kwargs) -> dict[str, Any]:
    """Thin wrapper для pytesseract.image_to_data — точка для
    monkeypatch'а в unit-тестах. Локальный import'им чтобы не
    требовать pytesseract на pure-unit тестах."""
    import pytesseract
    return pytesseract.image_to_data(img, **kwargs, output_type=pytesseract.Output.DICT)


def rescue_field(
    raster: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    psm: int = 7,
    lang: str = "rus+eng",
    whitelist: str | None = None,
    user_words: str | None = None,
    min_token_conf: float = 30.0,
) -> tuple[str, float]:
    """Re-OCR region ``bbox`` с targeted tesseract-config'ом.

    Args:
        raster: 2D или 3D ndarray preprocessed страницы.
        bbox: (x, y, w, h) region в пикселях raster'а.
        psm: Page Segmentation Mode. 7=single-line (обычно для
            полей), 6=uniform-block, 11=sparse-text.
        lang: Tesseract languages.
        whitelist: ``tessedit_char_whitelist`` — например
            только Cyrillic + digits для ИНН / имени.
        user_words: Путь к user-words файлу (FIO-лексикон
            для driver'а, ГРЗ-лексикон для vehicle).
        min_token_conf: Per-token conf threshold. Ниже —
            skip'ается из final text. 30 — средняя tesseract
            conf для «уверенных» токенов.

    Returns:
        ``(text, conf)`` где conf 0..1. ``("", 0.0)`` при любой
        ошибке (bbox out-of-bounds, tesseract crash, etc).
    """
    cropped = _crop_bbox(raster, bbox)
    if cropped is None or cropped.size == 0:
        return "", 0.0

    # Build tesseract config. Pre-pend PSM + OEM default.
    config_parts = [f"--psm {psm}", "--oem 1"]
    if whitelist:
        # Escape single quotes — tesseract -c syntax.
        esc = whitelist.replace('"', '\\"')
        config_parts.append(f'-c tessedit_char_whitelist="{esc}"')
    if user_words:
        config_parts.append(f'--user-words "{user_words}"')
    config = " ".join(config_parts)

    try:
        data = _pytesseract_image_to_data(
            cropped, lang=lang, config=config,
        )
    except Exception as exc:  # noqa: BLE001 — rescue must not propagate
        logger.debug("rescue_field: tesseract call failed: %s", exc)
        return "", 0.0

    # Фильтруем токены по min_token_conf и пустому text.
    words: list[str] = []
    confs: list[float] = []
    texts = data.get("text", [])
    confs_raw = data.get("conf", [])
    for text, conf in zip(texts, confs_raw, strict=False):
        text_s = str(text or "").strip()
        if not text_s:
            continue
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_f < min_token_conf:
            continue
        words.append(text_s)
        confs.append(conf_f)

    if not words:
        return "", 0.0

    return " ".join(words), sum(confs) / len(confs) / 100.0


__all__ = ["rescue_field"]
