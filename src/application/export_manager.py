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
import re
import shutil
from pathlib import Path

from src.core.models import JobResult, PageResult
from src.shared.constants import UI_PAGE_NUM_FORMAT
from src.shared.types import ExportFormat

logger = logging.getLogger(__name__)


def _mark_low_conf_words_txt(
    page: PageResult, *, marker_open: str = "[?", marker_close: str = "?]",
) -> str:
    """Return ``page.text`` with every token from ``page.low_confidence_words``
    wrapped in ``marker_open`` / ``marker_close``.

    Uses word-boundary regex matching so only whole-word occurrences
    are marked — ``с``-in-``с`` won't turn ``высокий`` into
    ``вы[?с?]окий``. Falls back to ``page.text`` unchanged when the
    low-conf list is empty (no-op fast path).

    The default markers are plain ASCII so the annotated output is
    still greppable / copy-pastable; callers that need alternative
    delimiters can override both strings.
    """
    if not page.low_confidence_words or not page.text:
        return page.text or ""
    # Deduplicate + sort longest-first so a word that is a prefix of
    # another doesn't win the match race (re.finditer is greedy but
    # alternation order still matters for equal-length candidates).
    unique = sorted(set(page.low_confidence_words), key=len, reverse=True)
    alternation = "|".join(re.escape(w) for w in unique if w)
    if not alternation:
        return page.text
    pattern = re.compile(rf"\b(?:{alternation})\b", re.UNICODE)
    return pattern.sub(
        lambda m: f"{marker_open}{m.group(0)}{marker_close}", page.text,
    )


def _split_for_docx_annotation(
    page: PageResult,
) -> list[tuple[str, bool]]:
    """Return ``page.text`` as a list of ``(span, is_low_conf)`` tuples.

    Used by :meth:`ExportManager.export_docx` to attach red font-colour
    to the low-conf spans without reimplementing word-boundary
    matching. Non-low-conf runs come through with ``is_low_conf=False``
    and render in default document colour.
    """
    if not page.low_confidence_words or not page.text:
        return [(page.text or "", False)]
    unique = sorted(set(page.low_confidence_words), key=len, reverse=True)
    alternation = "|".join(re.escape(w) for w in unique if w)
    if not alternation:
        return [(page.text, False)]
    pattern = re.compile(rf"\b(?:{alternation})\b", re.UNICODE)
    spans: list[tuple[str, bool]] = []
    last = 0
    for m in pattern.finditer(page.text):
        if m.start() > last:
            spans.append((page.text[last:m.start()], False))
        spans.append((m.group(0), True))
        last = m.end()
    if last < len(page.text):
        spans.append((page.text[last:], False))
    return spans


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
        if format == ExportFormat.EXCEL:
            return self.export_excel(job_result, output_path)
        raise ExportError(f"Неподдерживаемый формат экспорта: {format}")

    # ------------------------------------------------------------------
    # Concrete exporters
    # ------------------------------------------------------------------

    def export_txt(
        self,
        job_result: JobResult,
        output_path: Path,
        encoding: str = "utf-8",
        *,
        annotate_low_conf: bool = False,
    ) -> Path:
        """Write the OCR text to a ``.txt`` file.

        Each page is preceded by a header line derived from
        :data:`UI_PAGE_NUM_FORMAT`.

        Args:
            job_result: Finished job with per-page text.
            output_path: Destination path.
            encoding: Text encoding; use ``"utf-8-sig"`` for a BOM.
            annotate_low_conf: When True, wrap every word in
                :attr:`PageResult.low_confidence_words` with ``[?...?]``
                markers so the reader can spot uncertain tokens.
                No-op on pages with no low-conf list (including the
                default case where ``drop_low_conf_words`` already
                filtered them out upstream).

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
                body = (
                    _mark_low_conf_words_txt(page)
                    if annotate_low_conf
                    else (page.text or "")
                )
                lines.append(body)
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

    def export_docx(
        self,
        job_result: JobResult,
        output_path: Path,
        *,
        annotate_low_conf: bool = False,
    ) -> Path:
        """Write the OCR text to a ``.docx`` file.

        Each OCR page becomes a group of paragraphs followed by a page break.

        Args:
            job_result: Finished job.
            output_path: Destination ``.docx`` path.
            annotate_low_conf: When True, words from
                :attr:`PageResult.low_confidence_words` are rendered in
                red so the reader can spot uncertain tokens without
                scanning every character. No-op on pages with no
                low-conf list.

        Returns:
            The path that was written.

        Raises:
            ExportError: If ``python-docx`` is not installed.
        """
        try:
            from docx import Document  # type: ignore[import-not-found]
            from docx.shared import RGBColor  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ExportError(
                "Экспорт в DOCX требует пакет python-docx"
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Pure red for low-conf runs. Chosen for maximum contrast
        # against the default black body text on every built-in Word
        # theme; users who prefer a subtler highlight can adjust the
        # paragraph style after export.
        low_conf_rgb = RGBColor(0xCC, 0x00, 0x00)

        document = Document()
        pages = job_result.pages
        for idx, page in enumerate(pages):
            heading = UI_PAGE_NUM_FORMAT.format(num=page.page_number)
            document.add_heading(heading, level=2)
            if page.error:
                document.add_paragraph(f"[ERROR: {page.error}]")
            elif annotate_low_conf and page.low_confidence_words:
                # Split every paragraph into (span, is_low_conf)
                # tuples and emit one run per tuple so only the
                # uncertain tokens get the red colour.
                for para in (page.text or "").split("\n\n"):
                    text = para.strip("\n")
                    if not text:
                        continue
                    paragraph = document.add_paragraph()
                    # Build a proxy PageResult whose text is JUST this
                    # paragraph so the helper only marks this scope.
                    slice_page = PageResult(
                        page_number=page.page_number,
                        text=text,
                        low_confidence_words=page.low_confidence_words,
                    )
                    for span, is_low in _split_for_docx_annotation(slice_page):
                        if not span:
                            continue
                        run = paragraph.add_run(span)
                        if is_low:
                            run.font.color.rgb = low_conf_rgb
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

    def export_excel(
        self,
        job_result: JobResult,
        output_path: Path,
    ) -> Path:
        """Write the parser's structured rows to an ``.xlsx`` workbook.

        Delegates to :func:`src.tn_parser.excel.write_excel_safe` so the
        on-disk shape (13 columns, confidence colouring, frozen header,
        auto-filter, snapshot sidecar) matches byte-for-byte what the
        standalone ``python -m src.tn_parser`` CLI produces. Also writes
        a human-readable log next to the xlsx via
        :func:`src.tn_parser.report.build_log_lines` +
        :func:`write_log_safe` — the same ``<stem>.log`` operators are
        used to seeing from the parser's batch run.

        Side effects:
          * Writes ``<stem>.xlsx`` (or a timestamped fallback name if
            the primary path is locked by an open Excel window).
          * Writes ``<stem>.log`` next to it.
          * Writes ``<stem>.xlsx.snapshot.json`` (used by
            ``scripts/collect_feedback.py`` to diff operator edits).
          * Populates ``job_result.parsed.snapshot_path`` so downstream
            callers (UI, feedback tooling) can locate the sidecar
            without re-deriving the filename.

        Args:
            job_result: Finished job. MUST have ``parsed`` populated
                (profile with ``extract.enabled=True`` + at least one
                parser row); otherwise :class:`ExportError` is raised
                — silently producing an empty workbook would hide a
                profile / extraction-config bug from the user.
            output_path: Destination ``.xlsx`` path. The parent
                directory is created if missing.

        Returns:
            The actual path written. Differs from ``output_path``
            when the primary file was locked and
            :func:`write_excel_safe` fell back to a timestamped name.

        Raises:
            ExportError: When there is no parser output to export,
                the parser dependencies (openpyxl) are missing, or
                the filesystem refuses the write.
        """
        if job_result.parsed is None or not job_result.parsed.rows:
            raise ExportError(
                "Нет распознанных полей для экспорта в Excel. "
                "Проверьте, что профиль включает `extract.enabled=True` "
                "и что парсер нашёл хотя бы одну строку."
            )

        try:
            from src.tn_parser.excel import write_excel_safe
            from src.tn_parser.models import ParsedRow
            from src.tn_parser.report import build_log_lines, write_log_safe
        except ImportError as exc:
            # rapidfuzz / openpyxl absent from the frozen build.
            raise ExportError(
                "Экспорт в Excel недоступен: парсер накладных не "
                f"импортируется ({exc}). Переустановите приложение."
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Rehydrate ParsedRow dataclasses from the JSON-dict form we
        # stored on result.parsed (core/ is intentionally agnostic of
        # the parser's dataclass shape — see :class:`ParsedDocument`).
        rows = [ParsedRow.from_json_dict(d) for d in job_result.parsed.rows]

        try:
            actual_xlsx = write_excel_safe(rows, str(output_path))
        except PermissionError as exc:
            raise ExportError(
                f"Файл открыт в другой программе:\n{output_path}\n\n"
                "Закройте его (например, в Excel) и попробуйте снова."
            ) from exc
        except OSError as exc:
            if getattr(exc, "errno", None) == 28 or "full" in str(exc).lower():
                raise ExportError(
                    f"Недостаточно места на диске для записи {output_path}"
                ) from exc
            raise ExportError(f"Не удалось записать Excel: {exc}") from exc

        actual_xlsx_path = Path(actual_xlsx)
        # write_excel_safe writes the snapshot sidecar using the actual
        # path (not the original request), so derive it the same way
        # to avoid a dangling snapshot_path attribute on fallback.
        snapshot_path = str(actual_xlsx_path) + ".snapshot.json"
        job_result.parsed.snapshot_path = snapshot_path

        # Log file — same stem as the xlsx so a .xlsx -> .log pairing
        # works regardless of whether the xlsx took the fallback name.
        log_path = actual_xlsx_path.with_suffix(".log")
        source_name = Path(job_result.input_path).name or "input.pdf"
        try:
            log_lines = build_log_lines(
                input_path=job_result.input_path,
                output_path=actual_xlsx,
                elapsed_s=job_result.total_time_sec,
                rows_by_file={source_name: rows},
            )
            write_log_safe(str(log_path), log_lines)
        except (OSError, PermissionError) as exc:
            # Log-file problems must NOT kill the export — the
            # primary artifact (.xlsx) is already on disk. Emit a
            # WARNING so the support channel still sees the issue.
            logger.warning(
                "Excel log sidecar failed (%s: %s); xlsx still written to %s",
                type(exc).__name__, exc, actual_xlsx,
            )

        logger.info(
            "Exported Excel: %s (%d rows, snapshot=%s)",
            actual_xlsx, len(rows), snapshot_path,
        )
        return actual_xlsx_path

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
