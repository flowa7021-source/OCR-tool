"""EasyOCR engine: CRAFT detector + CRNN recognizer (ru + en).

Rasterises each preprocessed PDF page, runs ``Reader.readtext`` per
page (no per-page retry tier — EasyOCR's failure-mode is "returns
fewer blocks", not "crashes"), and writes a searchable PDF via
:mod:`src.infrastructure.searchable_pdf_builder`.

Models are loaded lazily at first :meth:`run` and kept on the engine
instance until :meth:`unload` (called by the registry on cache reset).
"""

from __future__ import annotations

import contextlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
import numpy as np

from src.application.engines.base import (
    EngineNotAvailableError,
    OCREngine,
    PageOCRResult,
    ProgressCallback,
)
from src.core.models import OCRConfig
from src.infrastructure.searchable_pdf_builder import (
    PageWords,
    build_searchable_pdf,
)
from src.shared.types import OCREngineKind

logger = logging.getLogger(__name__)

_DEFAULT_MIN_CONF: float = 0.1


def _resolve_workers(page_count: int) -> int:
    """Cap parallel page workers: ``OCR_PER_PAGE_WORKERS`` env or cpu/4."""
    env = os.environ.get("OCR_PER_PAGE_WORKERS")
    if env and env.isdigit() and int(env) >= 1:
        return min(page_count, int(env))
    return max(1, min(page_count, (os.cpu_count() or 2) // 2 or 1, 4))


class EasyOCREngine(OCREngine):
    """EasyOCR back-end."""

    kind = OCREngineKind.EASYOCR

    def __init__(self) -> None:
        self._reader = None
        self._reader_key: tuple | None = None

    @property
    def name(self) -> str:
        return "EasyOCR"

    @property
    def description(self) -> str:
        return (
            "CRAFT-детектор + CRNN-распознаватель (ru + en). "
            "Работает на CPU, опционально ускоряется на GPU."
        )

    def is_available(self) -> tuple[bool, str]:
        try:
            import easyocr  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            return False, (
                f"EasyOCR не установлен ({exc}). Установите пакет: "
                "pip install easyocr."
            )
        return True, "EasyOCR доступен."

    def _get_reader(self, languages: list[str], gpu: bool):
        """Lazy-load ``easyocr.Reader`` once per (langs, gpu) combo."""
        import easyocr

        key = (tuple(sorted(languages)), gpu)
        if self._reader is not None and self._reader_key == key:
            return self._reader
        logger.info("Loading EasyOCR model: langs=%s gpu=%s", languages, gpu)
        self._reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
        self._reader_key = key
        return self._reader

    def unload(self) -> None:
        self._reader = None
        self._reader_key = None

    def run(
        self,
        preprocessed_pdf: Path,
        output_pdf: Path,
        config: OCRConfig,
        progress_callback: ProgressCallback | None = None,
        *,
        original_input_pdf: Path | None = None,
    ) -> list[PageOCRResult]:
        ok, msg = self.is_available()
        if not ok:
            raise EngineNotAvailableError(msg)

        reader = self._get_reader(list(config.languages), bool(config.gpu))
        dpi = int(config.dpi)
        allowlist = config.allowlist or None
        min_conf = max(0.0, min(1.0, config.min_keep_confidence))

        doc = fitz.open(str(preprocessed_pdf))
        try:
            page_count = doc.page_count
            if page_count == 0:
                raise RuntimeError("Preprocessed PDF has zero pages")

            if progress_callback is not None:
                with contextlib.suppress(Exception):
                    progress_callback(0, page_count, "ocr")

            page_inputs: list[tuple[int, bytes, int, int]] = []
            for i in range(page_count):
                pix = doc[i].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
                page_inputs.append((i, pix.tobytes("png"),
                                    pix.width, pix.height))
        finally:
            doc.close()

        pages: list[PageWords | None] = [None] * page_count
        results: list[PageOCRResult | None] = [None] * page_count
        workers = _resolve_workers(page_count)

        def _run_one(idx: int, png: bytes, w: int, h: int):
            arr = np.frombuffer(
                fitz.Pixmap(png).samples, dtype=np.uint8,
            ).reshape(h, w)
            raw = reader.readtext(
                arr, detail=1, paragraph=False,
                allowlist=allowlist,
            )
            words: list[tuple[float, float, float, float, float, str]] = []
            texts: list[str] = []
            confs: list[float] = []
            for bbox, text, conf in raw:
                if conf < min_conf or not text.strip():
                    continue
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x, y = min(xs), min(ys)
                bw, bh = max(xs) - x, max(ys) - y
                words.append((float(x), float(y), float(bw), float(bh),
                              float(conf) * 100.0, text))
                texts.append(text)
                confs.append(float(conf))
            return idx, PageWords(png, w, h, dpi, words), \
                PageOCRResult(
                    page_number=idx + 1,
                    text="\n".join(texts),
                    mean_confidence=(
                        sum(confs) / len(confs) * 100.0 if confs else 0.0
                    ),
                    word_boxes=words,
                )

        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = [exe.submit(_run_one, *pi) for pi in page_inputs]
            for fut in as_completed(futures):
                idx, pw, pr = fut.result()
                pages[idx] = pw
                results[idx] = pr
                completed += 1
                if progress_callback is not None:
                    with contextlib.suppress(Exception):
                        progress_callback(completed, page_count, "ocr")

        build_searchable_pdf([p for p in pages if p is not None], output_pdf)
        logger.info("EasyOCR complete: %d pages → %s", page_count, output_pdf)
        return [r for r in results if r is not None]
