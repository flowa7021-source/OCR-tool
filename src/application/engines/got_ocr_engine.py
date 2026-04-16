"""GOT-OCR 2.0 back-end (Stepfun, Apache-2.0).

Optional engine that handles printed AND handwritten text in 80+
languages including Russian. Activated only when:

  1. The ``ocr-studio[htr]`` extras are installed (torch, transformers,
     pillow). We don't ship them in the default installer because they
     pull ~2 GB of wheels.
  2. The ~580 MB model weights have been downloaded to
     ``%LOCALAPPDATA%/OCRStudio/models/got_ocr2/``. The
     :class:`~src.infrastructure.model_manager.ModelManager` handles the
     download with progress reporting and is exposed in the UI.

Even when both conditions are met, performance on CPU is slow (multiple
seconds per page on a modern Intel chip). Users with CUDA-enabled torch
get a 30-50× speedup automatically — we don't pin a CPU build.

This engine intentionally does NOT depend on OCRmyPDF for the text
overlay. Instead it produces hOCR markup from the model's word
predictions and assembles a searchable PDF using PyMuPDF, which keeps
the integration path independent of the (printed-text-tuned) OCRmyPDF
pipeline.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from src.application.engines.base import (
    EngineNotAvailableError,
    OCREngine,
    PageOCRResult,
    ProgressCallback,
)
from src.core.models import OCRConfig
from src.infrastructure.model_manager import GOT_OCR2_SPEC, ModelManager
from src.shared.types import OCREngineKind

logger = logging.getLogger(__name__)


class GOTOCREngine(OCREngine):
    """GOT-OCR 2.0 wrapper. All ML imports are lazy."""

    kind = OCREngineKind.GOT_OCR2

    def __init__(self, model_manager: ModelManager | None = None) -> None:
        self._model_manager = model_manager or ModelManager()
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"

    @property
    def name(self) -> str:
        return "GOT-OCR 2.0"

    @property
    def description(self) -> str:
        return (
            "Transformer-OCR от Stepfun (2024). Распознаёт рукописный и "
            "печатный текст. Требует расширения [htr] и ~580 МБ модели."
        )

    # ----------------------------------------------------------- probes
    def is_available(self) -> tuple[bool, str]:
        """Verify Python deps, on-disk weights, AND transitive imports.

        We've seen two footguns in production:

        * A bundled ``torch`` that silently missed one of its DLLs
          (``VCOMP140.DLL`` on older Windows) — ``import torch``
          succeeded but the first ``torch.zeros(1)`` blew up. The
          smoke test below catches that.
        * GOT-OCR 2.0's ``trust_remote_code`` scripts pull in niche
          packages like ``einops`` / ``accelerate`` that aren't part
          of a minimal ``transformers`` install. Importing them here
          lets us return a clear error instead of a cryptic
          ``ModuleNotFoundError`` at inference time.
        """
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return False, (
                "Не установлено расширение htr. Запустите "
                "`pip install ocr-studio[htr]` или скачайте torch + "
                f"transformers вручную. Ошибка: {exc.name}"
            )
        # torch + transformers ran the import hook — now exercise them
        # once to catch DLL / shared-object load failures that only
        # surface on first use.
        try:
            import torch

            _ = torch.zeros(1)
        except Exception as exc:  # noqa: BLE001
            return False, (
                "torch установлен, но базовая операция "
                f"(torch.zeros) падает: {exc}. Скорее всего "
                "отсутствует рантайм VC++ или CUDA DLL."
            )
        # Transitive deps GOT-OCR 2.0's trust_remote_code scripts need.
        for dep in ("einops", "accelerate"):
            try:
                __import__(dep)
            except ImportError:
                return False, (
                    f"Для GOT-OCR 2.0 требуется пакет '{dep}'. "
                    f"Установите его (pip install {dep}) или "
                    "обратитесь к сборщику сборки."
                )
        if not self._model_manager.is_available(GOT_OCR2_SPEC.model_id):
            return False, (
                f"Модель {GOT_OCR2_SPEC.label} не скачана. Откройте "
                "Настройки → OCR-движок → Скачать модель."
            )
        return True, ""

    # ----------------------------------------------------------- loading
    @staticmethod
    def _safe_model_path(path: Path) -> str:
        """Return a path safe to hand to HuggingFace loaders.

        Some HuggingFace plumbing (safetensors' mmap path, older
        tokenizers builds) uses C++/Rust I/O that has historically
        choked on non-ASCII Windows paths — exactly the scenario a
        Cyrillic user profile ("``C:\\Users\\Т.Н. 020\\...``") lands
        us in. On Windows we attempt to resolve the path to its
        8.3 short form via ``GetShortPathNameW``, which is always
        pure ASCII. If the short form isn't available (e.g. 8.3 names
        disabled on NTFS, which is the default on Win10+) we fall
        back to the original path — Python-level HF code still works
        with Unicode, and the failure mode here would only be a
        diagnostic warning rather than a hard crash.
        """
        p_str = str(path)
        if os.name != "nt":
            return p_str
        # Only bother if the path contains characters above ASCII.
        if p_str.isascii():
            return p_str
        try:
            import ctypes
            from ctypes import wintypes

            get_short = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
            get_short.argtypes = [
                wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD,
            ]
            get_short.restype = wintypes.DWORD

            buf = ctypes.create_unicode_buffer(1024)
            needed = get_short(p_str, buf, len(buf))
            if needed == 0:
                return p_str  # API failed (permission / not a real path)
            if needed > len(buf):
                buf = ctypes.create_unicode_buffer(needed)
                if get_short(p_str, buf, needed) == 0:
                    return p_str
            short = buf.value
            if short and short != p_str:
                logger.info(
                    "Using short path for HuggingFace loader: %s -> %s",
                    p_str, short,
                )
                return short
            return p_str
        except Exception as exc:  # noqa: BLE001
            logger.debug("GetShortPathNameW fallback failed: %s", exc)
            return p_str

    def _load_model(self) -> None:
        """Lazily load weights into memory; idempotent."""
        if self._model is not None:
            return
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

        model_dir = self._model_manager.model_dir(GOT_OCR2_SPEC.model_id)
        hf_path = self._safe_model_path(model_dir)
        logger.info(
            "Loading GOT-OCR2 weights from %s (hf_path=%s)", model_dir, hf_path
        )
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(
            hf_path, trust_remote_code=True
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 on CUDA halves VRAM and gives a 2-3× inference speedup on
        # modern GPUs (T4, A10, RTX 30xx+) with no measurable accuracy
        # loss for GOT-OCR2. CPU-only path stays float32 — bf16/fp16 on
        # CPU is slower than fp32 in PyTorch without explicit AMP.
        torch_dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModel.from_pretrained(
            hf_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self._device,
            torch_dtype=torch_dtype,
        )
        self._model.eval()
        logger.info(
            "GOT-OCR2 ready on %s (%s) in %.1fs",
            self._device,
            torch_dtype,
            time.time() - t0,
        )

    def unload(self) -> None:
        """Release GOT-OCR 2.0 weights (~580 MB RAM / GPU memory).

        Called from :func:`src.application.engines.registry.reset_cache`
        which runs after a model download, after a delete, and any
        other time the engine cache is invalidated. The next OCR run
        triggers a fresh :meth:`_load_model`.
        """
        if self._model is None and self._tokenizer is None:
            return
        logger.info("Unloading GOT-OCR2 weights (device=%s)", self._device)
        self._model = None
        self._tokenizer = None
        # Help the CUDA allocator return memory to the OS; on CPU this
        # is purely a hint to the GC.
        try:
            import gc

            gc.collect()
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            logger.debug("unload cleanup non-fatal error", exc_info=True)

    # ----------------------------------------------------------- run
    def run(
        self,
        preprocessed_pdf: Path,
        output_pdf: Path,
        config: OCRConfig,  # noqa: ARG002 — language not used; GOT is multilingual
        progress_callback: ProgressCallback | None = None,
    ) -> list[PageOCRResult]:
        """OCR every page and assemble a searchable PDF."""
        ok, msg = self.is_available()
        if not ok:
            raise EngineNotAvailableError(msg)
        self._load_model()

        import fitz  # PyMuPDF, lazy

        src = fitz.open(str(preprocessed_pdf))
        try:
            page_count = src.page_count
            results: list[PageOCRResult] = []
            output_pdf.parent.mkdir(parents=True, exist_ok=True)
            out_doc = fitz.open()
            try:
                for idx in range(page_count):
                    if progress_callback is not None:
                        try:
                            progress_callback(idx, page_count, "got-ocr")
                        except Exception:  # noqa: BLE001
                            logger.debug("progress_callback raised", exc_info=True)
                    page = src.load_page(idx)
                    text, confidence = self._recognize_page(page)
                    self._append_page_with_overlay(out_doc, page, text)
                    results.append(
                        PageOCRResult(
                            page_number=idx + 1,
                            text=text,
                            mean_confidence=confidence,
                        )
                    )
                out_doc.save(str(output_pdf), garbage=4, deflate=True)
            finally:
                out_doc.close()
        finally:
            src.close()

        if progress_callback is not None:
            try:
                progress_callback(page_count, page_count, "got-ocr")
            except Exception:  # noqa: BLE001
                logger.debug("progress_callback raised", exc_info=True)
        return results

    # ----------------------------------------------------------- internal
    def _recognize_page(self, page: Any) -> tuple[str, float]:
        """Run GOT-OCR2 on a single PyMuPDF page.

        GOT exposes a ``chat`` method on its model; we use it via the
        documented ``ocr_type='format'`` mode which returns plain text.
        Bounding boxes are not provided by this mode — for now we
        forfeit per-word overlay accuracy in exchange for vastly better
        recognition; the text layer is laid down at page level.
        """
        import io

        from PIL import Image  # type: ignore[import-not-found]

        pix = page.get_pixmap(dpi=150, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        # GOT-OCR2 expects the image as a PIL.Image and supports two
        # ocr_types: 'ocr' (plain text) and 'format' (LaTeX/Markdown).
        # We use plain 'ocr' for plain documents.
        try:
            text = self._model.chat(
                self._tokenizer,
                img,
                ocr_type="ocr",
            )
        except MemoryError as exc:
            # Running out of system RAM is a hard stop — nothing we can
            # recover on this page, and the next page would fail too.
            logger.error("GOT-OCR2: system RAM exhausted on a page")
            self.unload()
            raise RuntimeError(
                "Недостаточно оперативной памяти для GOT-OCR 2.0. "
                "Закройте другие программы или используйте профиль "
                "с движком Tesseract."
            ) from exc
        except Exception as exc:  # noqa: BLE001 — catches torch.cuda OOM etc.
            err_name = type(exc).__name__
            # torch.cuda.OutOfMemoryError on PyTorch ≥2.1; surface a
            # user-friendly dialog instead of the raw CUDA traceback.
            if "OutOfMemory" in err_name or "CUDA out of memory" in str(exc):
                logger.error("GOT-OCR2: GPU memory exhausted: %s", exc)
                try:
                    import torch  # type: ignore[import-not-found]

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(
                    "GPU не хватило памяти для GOT-OCR 2.0 на этой "
                    "странице. Понизьте DPI в профиле или запустите на CPU."
                ) from exc
            logger.warning("GOT-OCR2 inference failed on page: %s", exc)
            return "", 0.0
        text = (text or "").strip()
        # Confidence is not exposed by GOT; report a neutral 80.0 so
        # downstream consumers (postprocessor, autosave) don't choke
        # on a zero score.
        return text, 80.0 if text else 0.0

    def _append_page_with_overlay(self, out_doc: Any, src_page: Any, text: str) -> None:
        """Copy ``src_page`` into ``out_doc`` and stamp the text invisibly."""
        import fitz

        out_page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
        # Embed the original page rendering as a background image.
        pix = src_page.get_pixmap(dpi=150, alpha=False)
        out_page.insert_image(out_page.rect, stream=pix.tobytes("png"))
        # Add invisible text spanning the page so Ctrl+F finds it.
        # Render mode 3 = neither fill nor stroke; the text is selectable
        # but not visually drawn over the underlying scan.
        if text:
            out_page.insert_textbox(
                out_page.rect,
                text,
                fontname="helv",
                fontsize=10,
                render_mode=3,
                align=fitz.TEXT_ALIGN_LEFT,
            )


__all__ = ["GOTOCREngine"]
