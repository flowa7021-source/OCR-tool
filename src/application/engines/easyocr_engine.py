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
        """Lazy-load ``easyocr.Reader`` once per (langs, resolved_gpu) combo.

        ``gpu=True`` from the profile means "use accelerator if one is
        present"; we resolve it against ``torch.cuda.is_available()``
        here so that a profile carried between a GPU workstation and a
        CPU-only laptop does not explode with a CUDA error on the
        laptop. Also respects the ``OCR_STUDIO_FORCE_CPU=1`` env var for
        debugging / benchmarking parity.
        """
        import os

        import easyocr
        import torch

        forced_cpu = os.environ.get("OCR_STUDIO_FORCE_CPU") == "1"
        cuda_ok = bool(
            not forced_cpu
            and gpu
            and torch.cuda.is_available()
        )
        key = (tuple(sorted(languages)), cuda_ok)
        if self._reader is not None and self._reader_key == key:
            return self._reader
        if gpu and not cuda_ok:
            logger.info(
                "EasyOCR: CUDA unavailable (forced_cpu=%s) — falling back to CPU",
                forced_cpu,
            )
        logger.info(
            "Loading EasyOCR model: langs=%s gpu=%s (resolved=%s)",
            languages, gpu, cuda_ok,
        )
        self._reader = easyocr.Reader(languages, gpu=cuda_ok, verbose=False)
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
        craft_kwargs = {
            "text_threshold": float(getattr(config, "craft_text_threshold", 0.7)),
            "low_text": float(getattr(config, "craft_low_text", 0.4)),
            "link_threshold": float(getattr(config, "craft_link_threshold", 0.4)),
            "canvas_size": int(getattr(config, "craft_canvas_size", 2560)),
            "contrast_ths": float(getattr(config, "easyocr_contrast_ths", 0.1)),
            "adjust_contrast": float(getattr(config, "easyocr_adjust_contrast", 0.5)),
        }
        # Optional post-OCR quality boosters.
        retry_enabled = bool(getattr(config, "low_conf_retry_enabled", False))
        retry_threshold = float(
            getattr(config, "low_conf_retry_threshold", 0.5),
        )
        lm_enabled = bool(
            getattr(config, "domain_lm_correction_enabled", False),
        )
        lm_skip_conf = float(getattr(config, "domain_lm_skip_conf", 0.85))
        lm_instance = None
        if lm_enabled:
            try:
                from src.shared.domain_lm import DomainLM
                lm_instance = DomainLM.from_default_corpus()
                logger.info(
                    "EasyOCR: domain LM loaded, %d vocab tokens",
                    len(lm_instance),
                )
                if len(lm_instance) == 0:
                    lm_instance = None  # empty → no-op
            except (ImportError, OSError) as e:
                logger.warning("EasyOCR: domain LM unavailable (%s)", e)
                lm_instance = None

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
                **craft_kwargs,
            )
            # Post-OCR boosters (opt-in via OCRConfig).
            if retry_enabled:
                from src.application.low_conf_retry import retry_low_confidence

                def _recognise(crop_arr):
                    return reader.readtext(
                        crop_arr, detail=1, paragraph=False,
                        allowlist=allowlist,
                        **craft_kwargs,
                    )

                raw, retry_stats = retry_low_confidence(
                    arr, raw, _recognise, threshold=retry_threshold,
                )
                if retry_stats.improved:
                    logger.info(
                        "EasyOCR page %d: low-conf retry improved "
                        "%d/%d words (unchanged=%d, failed=%d)",
                        idx, retry_stats.improved, retry_stats.attempted,
                        retry_stats.unchanged, retry_stats.failed,
                    )
            if lm_instance is not None:
                corrected_raw = []
                lm_changed = 0
                for bbox, text, conf in raw:
                    if not text.strip():
                        corrected_raw.append((bbox, text, conf))
                        continue
                    result = lm_instance.correct(
                        text, ocr_conf=float(conf),
                        min_ocr_conf_to_skip=lm_skip_conf,
                    )
                    if result.was_corrected:
                        corrected_raw.append((bbox, result.word, conf))
                        lm_changed += 1
                    else:
                        corrected_raw.append((bbox, text, conf))
                raw = corrected_raw
                if lm_changed:
                    logger.info(
                        "EasyOCR page %d: domain-LM corrected %d words",
                        idx, lm_changed,
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
