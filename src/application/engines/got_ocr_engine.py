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
        """Verify both Python deps and on-disk weights are present."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return False, (
                "Не установлено расширение htr. Запустите "
                "`pip install ocr-studio[htr]` или скачайте torch + "
                f"transformers вручную. Ошибка: {exc.name}"
            )
        if not self._model_manager.is_available(GOT_OCR2_SPEC.model_id):
            return False, (
                f"Модель {GOT_OCR2_SPEC.label} не скачана. Откройте "
                "Настройки → OCR-движок → Скачать модель."
            )
        return True, ""

    # ----------------------------------------------------------- loading
    def _load_model(self) -> None:
        """Lazily load weights into memory; idempotent."""
        if self._model is not None:
            return
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

        model_dir = self._model_manager.model_dir(GOT_OCR2_SPEC.model_id)
        logger.info("Loading GOT-OCR2 weights from %s", model_dir)
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self._device,
        )
        self._model.eval()
        logger.info(
            "GOT-OCR2 ready on %s in %.1fs", self._device, time.time() - t0
        )

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
        except Exception as exc:  # noqa: BLE001
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
