"""Export OCR results to various formats.

Supported outputs:
    * ``txt`` — plain UTF-8 text file with page separator lines.
    * ``docx`` — Microsoft Word document via ``python-docx``.
    * ``clipboard`` — copy text to the system clipboard (requires Qt runtime).
    * ``pdf``   — no-op here; the searchable PDF is produced by OCRmyPDF.

All methods operate on an already-completed :class:`JobResult` and never
re-run OCR.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from src.core.models import JobResult
from src.shared.constants import UI_PAGE_NUM_FORMAT
from src.shared.types import ExportFormat

logger = logging.getLogger(__name__)


class ExportError(RuntimeError):
    """Raised for export failures that callers are expected to handle."""


class ExportManager:
    """Serialize :class:`JobResult` instances to disk/clipboard."""

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def export(
        self,
        job_result: JobResult,
        output_path: Path,
        format: ExportFormat,
        encoding: str = "utf-8",
    ) -> Path:
        """Dispatch to the correct exporter based on ``format``.

        Args:
            job_result: Finished job.
            output_path: Destination path (ignored for ``CLIPBOARD``).
            format: Target export format.
            encoding: TXT encoding (ignored for other formats). Accepts any
                Python codec name; common choices are ``"utf-8"``,
                ``"utf-8-sig"`` (BOM) and ``"cp1251"``.

        Returns:
            Path to the written file (or the input ``output_path`` for
            clipboard exports).

        Raises:
            ExportError: If the requested format is unsupported.
        """
        if format == ExportFormat.TXT:
            return self.export_txt(job_result, output_path, encoding=encoding)
        if format == ExportFormat.DOCX:
            return self.export_docx(job_result, output_path)
        if format == ExportFormat.CLIPBOARD:
            self.copy_to_clipboard(self._concatenate_text(job_result))
            return output_path
        if format == ExportFormat.PDF:
            return self.export_pdf(job_result, output_path)
        raise ExportError(f"Неподдерживаемый формат экспорта: {format}")

    # ------------------------------------------------------------------
    # Concrete exporters
    # ------------------------------------------------------------------

    def export_txt(
        self,
        job_result: JobResult,
        output_path: Path,
        encoding: str = "utf-8",
    ) -> Path:
        """Write the OCR text to a ``.txt`` file.

        Each page is preceded by a header line derived from
        :data:`UI_PAGE_NUM_FORMAT`.

        Args:
            job_result: Finished job with per-page text.
            output_path: Destination path.
            encoding: Text encoding; use ``"utf-8-sig"`` for a BOM.

        Returns:
            The path that was written.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for page in job_result.pages:
            header = UI_PAGE_NUM_FORMAT.format(num=page.page_number)
            lines.append(header)
            lines.append("")
            if page.error:
                lines.append(f"[ERROR: {page.error}]")
            else:
                lines.append(page.text or "")
            lines.append("")

        text = "\n".join(lines).rstrip() + "\n"
        try:
            with open(output_path, "w", encoding=encoding, newline="\n") as fh:
                fh.write(text)
        except PermissionError as exc:
            raise ExportError(
                f"Файл открыт в другой программе:\n{output_path}\n\n"
                "Закройте его и попробуйте снова."
            ) from exc
        except OSError as exc:
            # Windows error 112 = ERROR_DISK_FULL; POSIX ENOSPC == 28.
            if getattr(exc, "errno", None) == 28 or "full" in str(exc).lower():
                raise ExportError(
                    f"Недостаточно места на диске для записи {output_path}"
                ) from exc
            raise ExportError(f"Не удалось записать TXT: {exc}") from exc
        logger.info(
            "Exported TXT: %s (%d pages, encoding=%s)",
            output_path,
            len(job_result.pages),
            encoding,
        )
        return output_path

    def export_pdf(self, job_result: JobResult, output_path: Path) -> Path:
        """Save the searchable PDF produced by OCRmyPDF to ``output_path``.

        The pipeline already writes a searchable PDF to
        ``job_result.output_path``. This method simply copies that file to
        a user-chosen destination — which is all "Save PDF as..." actually
        needs to do. When the destination matches the source it is a no-op
        (but still returns the existing path for uniformity).

        Args:
            job_result: Finished job carrying ``output_path`` of the OCR'd PDF.
            output_path: Destination path for the saved copy.

        Returns:
            The path that was written (or confirmed to already exist).

        Raises:
            ExportError: If the source PDF is missing or the copy fails.
        """
        source = Path(job_result.output_path)
        target = Path(output_path)
        if not source.exists():
            raise ExportError(
                f"OCR'd PDF не найден: {source}. Выполните распознавание заново."
            )
        if source.resolve() == target.resolve():
            logger.debug("PDF export: source == target (%s), nothing to copy", target)
            return target
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except PermissionError as exc:
            raise ExportError(
                f"Файл открыт в другой программе:\n{target}\n\n"
                "Закройте его (например, в Adobe Reader) и попробуйте снова."
            ) from exc
        except OSError as exc:
            if getattr(exc, "errno", None) == 28 or "full" in str(exc).lower():
                raise ExportError(
                    f"Недостаточно места на диске для сохранения PDF в {target}"
                ) from exc
            raise ExportError(f"Не удалось сохранить PDF: {exc}") from exc
        logger.info("Exported PDF: %s -> %s (%d bytes)", source, target, target.stat().st_size)
        return target

    def export_docx(self, job_result: JobResult, output_path: Path) -> Path:
        """Write the OCR text to a ``.docx`` file.

        Each OCR page becomes a group of paragraphs followed by a page break.

        Args:
            job_result: Finished job.
            output_path: Destination ``.docx`` path.

        Returns:
            The path that was written.

        Raises:
            ExportError: If ``python-docx`` is not installed.
        """
        try:
            from docx import Document  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ExportError(
                "Экспорт в DOCX требует пакет python-docx"
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)

        document = Document()
        pages = job_result.pages
        for idx, page in enumerate(pages):
            heading = UI_PAGE_NUM_FORMAT.format(num=page.page_number)
            document.add_heading(heading, level=2)
            if page.error:
                document.add_paragraph(f"[ERROR: {page.error}]")
            else:
                for para in (page.text or "").split("\n\n"):
                    text = para.strip("\n")
                    if text:
                        document.add_paragraph(text)
            if idx < len(pages) - 1:
                document.add_page_break()

        document.save(str(output_path))
        logger.info("Exported DOCX: %s (%d pages)", output_path, len(pages))
        return output_path

    def copy_to_clipboard(self, text: str) -> None:
        """Copy ``text`` to the system clipboard via Qt.

        Qt is imported lazily so that non-UI callers (tests, headless runs)
        can import this module without PySide6 installed.

        Raises:
            RuntimeError: If no Qt application instance is available.
        """
        try:
            from PySide6.QtWidgets import QApplication  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Clipboard requires Qt runtime") from exc

        app = QApplication.instance()
        if app is None:
            raise RuntimeError("Clipboard requires a running QApplication")
        clipboard = app.clipboard()
        if clipboard is None:  # pragma: no cover - platform-specific
            raise RuntimeError("System clipboard is not available")
        clipboard.setText(text)
        logger.info("Copied %d chars to clipboard", len(text))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _concatenate_text(job_result: JobResult) -> str:
        """Join every page's text with header separators (for clipboard)."""
        parts: list[str] = []
        for page in job_result.pages:
            parts.append(UI_PAGE_NUM_FORMAT.format(num=page.page_number))
            if page.error:
                parts.append(f"[ERROR: {page.error}]")
            else:
                parts.append(page.text or "")
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"
