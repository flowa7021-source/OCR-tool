# -*- coding: utf-8 -*-
"""Парсер транспортных накладных из PDF в Excel.

Десктопное приложение для Windows 10/11 на Python + Tkinter. Извлекает данные
российских транспортных накладных (ТН) из машиночитаемых PDF и сохраняет
результат в форматированный .xlsx + лог-файл с уровнями доверия.

Логика парсинга, записи Excel и формирования лога вынесена в пакет
`tn_parser/`. Этот модуль — тонкий GUI-слой поверх него.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import List, Optional

from tn_parser import (
    ParsedRow,
    build_log_lines,
    extract_raw_text,
    process_batch,
    write_excel_safe,
    write_log_safe,
)


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

APP_TITLE = "Парсер транспортных накладных"
OUTPUT_FILENAME = "extraction.xlsx"
LOG_FILENAME = "extraction.log"


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class ParserApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("760x560")
        self.root.minsize(680, 500)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.worker_thread: Optional[threading.Thread] = None
        self.msg_queue: "queue.Queue[tuple]" = queue.Queue()

        self._build_ui()
        self._poll_queue()

    # ---- UI ----------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}

        frame_top = ttk.Frame(self.root)
        frame_top.pack(fill="x", **pad)

        ttk.Label(frame_top, text="Папка с PDF:").grid(row=0, column=0, sticky="w")
        in_entry = ttk.Entry(frame_top, textvariable=self.input_var)
        in_entry.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(frame_top, text="Обзор…", command=self._choose_input).grid(row=0, column=2)

        ttk.Label(frame_top, text="Сохранить в:").grid(row=1, column=0, sticky="w")
        out_entry = ttk.Entry(frame_top, textvariable=self.output_var)
        out_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(frame_top, text="Обзор…", command=self._choose_output).grid(row=1, column=2)

        frame_top.columnconfigure(1, weight=1)

        actions = ttk.Frame(self.root)
        actions.pack(fill="x", **pad)
        self.run_btn = ttk.Button(
            actions, text="▶  Извлечь данные", command=self._on_run, state="disabled"
        )
        self.run_btn.pack(side="left")

        self.raw_btn = ttk.Button(
            actions, text="🔍  Сырой текст PDF…", command=self._on_show_raw
        )
        self.raw_btn.pack(side="left", padx=8)

        self.input_var.trace_add("write", lambda *_: self._refresh_run_state())
        self.output_var.trace_add("write", lambda *_: self._refresh_run_state())

        log_frame = ttk.LabelFrame(self.root, text="Лог")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = ScrolledText(log_frame, height=14, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", **pad)
        ttk.Label(bottom, text="Прогресс:").pack(side="left")
        self.progress = ttk.Progressbar(bottom, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=8)
        self.progress_label = ttk.Label(bottom, text="0/0")
        self.progress_label.pack(side="left")

    def _choose_input(self) -> None:
        path = filedialog.askdirectory(title="Выберите папку с PDF")
        if path:
            self.input_var.set(path)
            if not self.output_var.get():
                self.output_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Папка для сохранения")
        if path:
            self.output_var.set(path)

    def _refresh_run_state(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        if self.input_var.get().strip():
            self.run_btn.configure(state="normal")
        else:
            self.run_btn.configure(state="disabled")

    # ---- «Сырой текст» -----------------------------------------------------

    def _on_show_raw(self) -> None:
        initial = self.input_var.get().strip() or os.getcwd()
        pdf_path = filedialog.askopenfilename(
            title="Выберите PDF для просмотра",
            initialdir=initial,
            filetypes=[("PDF", "*.pdf"), ("Все файлы", "*.*")],
        )
        if not pdf_path:
            return

        try:
            text = extract_raw_text(pdf_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Не удалось прочитать PDF: {exc}")
            return

        self._open_text_window(os.path.basename(pdf_path), text or "[пусто]")

    def _open_text_window(self, title: str, text: str) -> None:
        win = tk.Toplevel(self.root)
        win.title(f"Сырой текст: {title}")
        win.geometry("820x620")
        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        widget = ScrolledText(frame, wrap="word")
        widget.pack(fill="both", expand=True)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

        def copy_all() -> None:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

        bottom = ttk.Frame(win)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(bottom, text="Копировать всё", command=copy_all).pack(side="right")
        ttk.Button(bottom, text="Закрыть", command=win.destroy).pack(side="right", padx=6)

    # ---- Логирование в UI --------------------------------------------------

    def _log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log(msg[1])
                elif kind == "progress":
                    done, total = msg[1], msg[2]
                    self.progress["maximum"] = max(total, 1)
                    self.progress["value"] = done
                    self.progress_label.configure(text=f"{done}/{total}")
                elif kind == "done":
                    self.run_btn.configure(state="normal")
                    self._refresh_run_state()
                elif kind == "error":
                    messagebox.showerror(APP_TITLE, msg[1])
                elif kind == "info":
                    messagebox.showinfo(APP_TITLE, msg[1])
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    # ---- Запуск обработки --------------------------------------------------

    def _on_run(self) -> None:
        in_dir = self.input_var.get().strip()
        out_dir = self.output_var.get().strip() or in_dir

        if not in_dir or not os.path.isdir(in_dir):
            messagebox.showwarning(APP_TITLE, "Выберите существующую папку с PDF.")
            return
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError as exc:
                messagebox.showerror(APP_TITLE, f"Не удалось создать папку: {exc}")
                return

        pdfs = sorted(
            os.path.join(in_dir, f)
            for f in os.listdir(in_dir)
            if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(in_dir, f))
        )
        if not pdfs:
            messagebox.showwarning(APP_TITLE, "В выбранной папке нет PDF-файлов.")
            return

        self.run_btn.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress["value"] = 0
        self.progress_label.configure(text=f"0/{len(pdfs)}")

        out_path = os.path.join(out_dir, OUTPUT_FILENAME)
        log_path = os.path.join(out_dir, LOG_FILENAME)
        self.worker_thread = threading.Thread(
            target=self._worker,
            args=(pdfs, in_dir, out_path, log_path),
            daemon=True,
        )
        self.worker_thread.start()

    def _worker(
        self,
        pdfs: List[str],
        input_dir: str,
        out_path: str,
        log_path: str,
    ) -> None:
        t0 = time.time()
        total = len(pdfs)

        def _on_progress(done, total_, pdf_path, rows) -> None:
            fname = os.path.basename(pdf_path)
            if rows and rows[0].note.startswith("ERROR:"):
                self.msg_queue.put(("log", f"Ошибка: {fname} — {rows[0].note[7:]}"))
            else:
                avg_pct = round(
                    sum(r.confidence.overall() for r in rows) / max(len(rows), 1) * 100
                )
                extra = f" ({len(rows)} накладных)" if len(rows) > 1 else ""
                self.msg_queue.put(("log", f"Обработано: {fname} — {avg_pct}%{extra}"))
            self.msg_queue.put(("progress", done, total_))

        try:
            results = process_batch(pdfs, progress=_on_progress)

            ordered_rows: List[ParsedRow] = []
            for p in pdfs:
                ordered_rows.extend(results.get(p, []))

            try:
                actual_xlsx = write_excel_safe(ordered_rows, out_path)
            except PermissionError as exc:
                self.msg_queue.put((
                    "error",
                    f"Не удалось сохранить Excel:\n{exc}\n\n"
                    "Возможно, файл открыт в другой программе или у вас "
                    "нет прав на запись в эту папку.",
                ))
                return

            if actual_xlsx != out_path:
                self.msg_queue.put((
                    "log",
                    f"⚠ Файл {os.path.basename(out_path)} занят другой программой "
                    "(вероятно, открыт в Excel).",
                ))
                self.msg_queue.put((
                    "log",
                    f"  Результат сохранён как {os.path.basename(actual_xlsx)}.",
                ))

            elapsed = time.time() - t0
            rows_by_fname = {os.path.basename(p): results[p] for p in pdfs if p in results}
            log_lines = build_log_lines(
                input_path=input_dir,
                output_path=actual_xlsx,
                elapsed_s=elapsed,
                rows_by_file=rows_by_fname,
            )
            try:
                actual_log = write_log_safe(log_path, log_lines)
            except PermissionError:
                actual_log = None  # не критично — просто не запишем лог

            ok_count = sum(
                1 for rows in results.values()
                if not any(r.note.startswith("ERROR:") for r in rows)
            )
            err_count = len(results) - ok_count

            self.msg_queue.put(("log", "──────────────────────────"))
            self.msg_queue.put((
                "log",
                f"Итого: {total} файлов, {ok_count} OK, {err_count} ошибок, "
                f"{len(ordered_rows)} строк (за {elapsed:.1f} с)",
            ))
            self.msg_queue.put(("log", f"Excel: {actual_xlsx}"))
            if actual_log:
                self.msg_queue.put(("log", f"Лог:   {actual_log}"))
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=4)
            self.msg_queue.put(("log", f"Критическая ошибка: {exc}\n{tb}"))
            self.msg_queue.put(("error", f"Сбой обработки: {exc}"))
        finally:
            self.msg_queue.put(("done",))


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def main() -> int:
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        themes = style.theme_names()
        for preferred in ("vista", "winnative", "clam"):
            if preferred in themes:
                style.theme_use(preferred)
                break
    except tk.TclError:
        pass
    ParserApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
