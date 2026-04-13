"""Main window tying together viewers, panels, queue, and menus."""

from __future__ import annotations

import base64
import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from src.core.models import OCRJobConfig, ProfileData, QueueItem
from src.shared.constants import (
    APP_NAME,
    APP_VERSION,
    IMPORT_FILE_FILTERS,
    LOGS_DIR,
    LOG_FILE_NAME,
)
from src.shared.types import ExportFormat
from src.application.recovery_manager import RecoveryManager
from src.infrastructure.file_utils import safe_unique_path, suggest_output_path
from src.ui.icons import app_icon, load_icon
from src.ui.pdf_viewer import PDFViewer
from src.ui.postprocess_panel import PostprocessPanel
from src.ui.preferences_dialog import PreferencesDialog
from src.ui.preprocessing_panel import PreprocessingPanel
from src.ui.progress_widget import ProgressWidget
from src.ui.queue_panel import QueuePanel
from src.ui.results_panel import ResultsPanel
from src.ui.settings_panel import SettingsPanel

if TYPE_CHECKING:
    from src.application.export_manager import ExportManager
    from src.application.parallel_processor import ParallelProcessor
    from src.application.profile_manager import ProfileManager
    from src.application.queue_manager import QueueManager
    from src.infrastructure.config_storage import SettingsStorage

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(
        self,
        profile_manager: "ProfileManager",
        queue_manager: "QueueManager",
        parallel_processor: "ParallelProcessor",
        export_manager: "ExportManager",
        settings_storage: "SettingsStorage",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._profile_manager = profile_manager
        self._queue_manager = queue_manager
        self._parallel_processor = parallel_processor
        self._export_manager = export_manager
        self._settings_storage = settings_storage
        self._last_result = None
        self._current_profile: ProfileData | None = None
        self._recovery = RecoveryManager()

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setAcceptDrops(True)
        self.resize(1400, 900)
        self.setWindowIcon(app_icon())

        self._build_widgets()
        self._build_docks()
        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()
        self._wire_signals()

        self._load_profiles_to_combobox()
        self._restore_window_state()

    # ------------------------------------------------------------ build UI
    def _build_widgets(self) -> None:
        self.pdf_viewer = PDFViewer(self)
        self.settings_panel = SettingsPanel(self)
        self.preprocessing_panel = PreprocessingPanel(self)
        self.postprocess_panel = PostprocessPanel(self)
        self.queue_panel = QueuePanel(self)
        self.progress_widget = ProgressWidget(self)
        self.results_panel = ResultsPanel(self)

        # Right side: settings on top, then preprocessing / postprocessing tabs below
        right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        right_splitter.addWidget(self.settings_panel)

        self.right_tabs = QTabWidget(self)
        self.right_tabs.addTab(self.preprocessing_panel, "Предобработка")
        self.right_tabs.addTab(self.postprocess_panel, "Постобработка")
        right_splitter.addWidget(self.right_tabs)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 2)

        # Central horizontal splitter: viewer | right
        central_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        central_splitter.addWidget(self.pdf_viewer)
        central_splitter.addWidget(right_splitter)
        central_splitter.setStretchFactor(0, 3)
        central_splitter.setStretchFactor(1, 2)
        self.setCentralWidget(central_splitter)

    def _build_docks(self) -> None:
        # Queue dock (bottom)
        queue_wrapper = QWidget(self)
        q_layout = QVBoxLayout(queue_wrapper)
        q_layout.setContentsMargins(0, 0, 0, 0)
        q_layout.addWidget(self.progress_widget)
        q_layout.addWidget(self.queue_panel, 1)

        self.queue_dock = QDockWidget("Очередь обработки", self)
        self.queue_dock.setObjectName("queueDock")
        self.queue_dock.setWidget(queue_wrapper)
        self.queue_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.queue_dock)

        # Results dock (bottom, tabbed with queue)
        self.results_dock = QDockWidget("Результаты распознавания", self)
        self.results_dock.setObjectName("resultsDock")
        self.results_dock.setWidget(self.results_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.results_dock)
        self.tabifyDockWidget(self.queue_dock, self.results_dock)
        self.queue_dock.raise_()

    def _build_toolbar(self) -> None:
        tb = QToolBar("Главная", self)
        tb.setObjectName("mainToolBar")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.action_open = QAction(load_icon("open"), "Открыть PDF", self)
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self._on_open_file)
        tb.addAction(self.action_open)

        tb.addSeparator()
        tb.addWidget(QLabel("Профиль: "))
        self.profile_combo = QComboBox(self)
        self.profile_combo.setMinimumWidth(180)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        tb.addWidget(self.profile_combo)

        tb.addSeparator()

        self.action_start = QAction(load_icon("start"), "Старт", self)
        self.action_start.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self.action_start.triggered.connect(self._on_start)
        tb.addAction(self.action_start)

        self.action_save = QAction(load_icon("save"), "Сохранить результат", self)
        self.action_save.setShortcut(QKeySequence.StandardKey.Save)
        self.action_save.triggered.connect(self._on_save)
        tb.addAction(self.action_save)

    def _build_menus(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&Файл")
        file_menu.addAction(self.action_open)
        folder_act = QAction("Открыть папку…", self)
        folder_act.triggered.connect(self._on_open_folder)
        file_menu.addAction(folder_act)
        file_menu.addSeparator()
        self.recent_menu = QMenu("Недавние файлы", self)
        file_menu.addMenu(self.recent_menu)
        self._rebuild_recent_menu()
        file_menu.addSeparator()
        prefs_act = QAction("Настройки…", self)
        prefs_act.setShortcut(QKeySequence("Ctrl+,"))
        prefs_act.triggered.connect(self._on_preferences)
        file_menu.addAction(prefs_act)
        file_menu.addSeparator()
        exit_act = QAction("Выход", self)
        exit_act.setShortcut(QKeySequence.StandardKey.Quit)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        prof_menu = menubar.addMenu("&Профиль")
        dup_act = QAction("Дублировать…", self)
        dup_act.triggered.connect(self._on_duplicate_profile)
        prof_menu.addAction(dup_act)
        save_prof_act = QAction("Сохранить изменения в профиле", self)
        save_prof_act.triggered.connect(self._on_save_profile)
        prof_menu.addAction(save_prof_act)
        del_act = QAction("Удалить…", self)
        del_act.triggered.connect(self._on_delete_profile)
        prof_menu.addAction(del_act)
        prof_menu.addSeparator()
        import_act = QAction("Импорт…", self)
        import_act.triggered.connect(self._on_import_profile)
        prof_menu.addAction(import_act)
        export_act = QAction("Экспорт…", self)
        export_act.triggered.connect(self._on_export_profile)
        prof_menu.addAction(export_act)

        view_menu = menubar.addMenu("&Вид")
        view_menu.addAction(self.queue_dock.toggleViewAction())
        view_menu.addAction(self.results_dock.toggleViewAction())

        help_menu = menubar.addMenu("&Справка")
        about_act = QAction("О программе", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)
        log_act = QAction("Открыть лог-файл", self)
        log_act.triggered.connect(self._on_open_log)
        help_menu.addAction(log_act)

        # Profile quick-switch shortcuts
        for i in range(4):
            act = QAction(f"Профиль {i + 1}", self)
            act.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            act.triggered.connect(lambda _checked=False, idx=i: self._switch_profile_by_index(idx))
            self.addAction(act)

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.status_profile_label = QLabel("Профиль: —")
        self.status_resource_label = QLabel("")
        sb.addPermanentWidget(self.status_profile_label)
        sb.addPermanentWidget(self.status_resource_label)
        self._resource_timer = QTimer(self)
        self._resource_timer.setInterval(2000)
        self._resource_timer.timeout.connect(self._update_resource_usage)
        self._resource_timer.start()
        self._update_resource_usage()

    def _wire_signals(self) -> None:
        self.pdf_viewer.document_opened.connect(self._on_document_opened)
        self.pdf_viewer.page_changed.connect(self._on_page_changed)
        self.queue_panel.attach_queue(self._queue_manager)
        self.queue_panel.files_dropped.connect(self._enqueue_files)
        self.queue_panel.remove_requested.connect(self._queue_manager.remove)
        self.queue_panel.move_up_requested.connect(self._queue_manager.move_up)
        self.queue_panel.move_down_requested.connect(self._queue_manager.move_down)
        self.queue_panel.pause_requested.connect(self._queue_manager.pause)
        self.queue_panel.resume_requested.connect(self._queue_manager.resume)
        self.queue_panel.cancel_requested.connect(self._queue_manager.cancel)
        self.results_panel.export_requested.connect(self._on_export_requested)

    # ------------------------------------------------------------ profiles
    def _load_profiles_to_combobox(self) -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        profiles = self._profile_manager.list_profiles()
        for p in profiles:
            prefix = "⭐ " if p.builtin else ""
            self.profile_combo.addItem(f"{prefix}{p.name}", userData=p.name)
        self.profile_combo.blockSignals(False)
        if profiles:
            self.profile_combo.setCurrentIndex(0)
            self._apply_profile(profiles[0])

    def _on_profile_selected(self, idx: int) -> None:
        name = self.profile_combo.itemData(idx)
        if not name:
            return
        try:
            profile = self._profile_manager.load(name)
        except Exception as exc:
            logger.exception("Failed to load profile %s: %s", name, exc)
            return
        self._apply_profile(profile)

    def _apply_profile(self, profile: ProfileData) -> None:
        self._current_profile = copy.deepcopy(profile)
        self.settings_panel.set_config(profile.ocr)
        self.preprocessing_panel.set_config(profile.preprocess)
        self.postprocess_panel.set_config(profile.postprocess)
        self.status_profile_label.setText(f"Профиль: {profile.name}")
        try:
            self._profile_manager.set_current(profile.name)
        except Exception:  # noqa: BLE001
            pass

    def _switch_profile_by_index(self, idx: int) -> None:
        if 0 <= idx < self.profile_combo.count():
            self.profile_combo.setCurrentIndex(idx)

    def _current_profile_with_overrides(self) -> ProfileData:
        base = self._current_profile or ProfileData(name="ad-hoc")
        clone = copy.deepcopy(base)
        clone.ocr = self.settings_panel.get_config()
        clone.preprocess = self.preprocessing_panel.get_config()
        clone.postprocess = self.postprocess_panel.get_config()
        return clone

    # ------------------------------------------------------------ actions
    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть PDF", "", IMPORT_FILE_FILTERS
        )
        if not path:
            return
        self._open_pdf(Path(path))

    def _open_pdf(self, path: Path) -> None:
        """Open a PDF file in the viewer and push it onto the recent list."""
        self.pdf_viewer.open(path)
        self._add_to_recent(path)

    def _on_open_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Выбрать папку с PDF")
        if not directory:
            return
        pdfs = sorted(Path(directory).rglob("*.pdf"))
        if not pdfs:
            QMessageBox.information(self, APP_NAME, "В выбранной папке нет PDF-файлов.")
            return
        self._enqueue_files(pdfs)

    def _on_document_opened(self, path_str: str) -> None:
        self.preprocessing_panel.set_preview_source(Path(path_str), 1)

    def _on_page_changed(self, page_num: int) -> None:
        if self.pdf_viewer.document_path is not None:
            self.preprocessing_panel.set_preview_source(
                self.pdf_viewer.document_path, page_num
            )

    def _on_start(self) -> None:
        if self.pdf_viewer.document_path is None:
            QMessageBox.information(self, APP_NAME, "Сначала откройте PDF.")
            return
        self._enqueue_files([self.pdf_viewer.document_path])

    def _enqueue_files(self, paths: list[Path]) -> None:
        profile = self._current_profile_with_overrides()
        for input_path in paths:
            try:
                output_path = safe_unique_path(suggest_output_path(input_path))
                job_cfg = OCRJobConfig(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    profile=profile,
                )
                item = QueueItem(config=job_cfg, progress_total=0)
                self._queue_manager.add(item)
                try:
                    self._recovery.snapshot(item)
                except Exception:  # noqa: BLE001
                    logger.debug("recovery snapshot failed", exc_info=True)
                self._submit_job(item)
                self._add_to_recent(input_path)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to enqueue %s: %s", input_path, exc)
                QMessageBox.critical(self, APP_NAME, f"Не удалось добавить {input_path.name}:\n{exc}")

    def _submit_job(self, item: QueueItem) -> None:
        if item.config is None:
            return
        try:
            future = self._parallel_processor.submit(
                item.config,
                on_progress=lambda jid, cur, tot, stage: self._on_job_progress(jid, cur, tot, stage),
                on_complete=lambda result, jid=item.job_id: self._on_job_complete(jid, result),
                on_error=lambda exc, jid=item.job_id: self._on_job_failed(jid, exc),
                job_id=item.job_id,
            )
            from src.shared.types import JobStatus as _JS

            self._queue_manager.update_status(item.job_id, _JS.RUNNING)
            logger.info("Submitted job %s: %s", item.job_id, future)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Submission failed: %s", exc)
            QMessageBox.critical(self, APP_NAME, f"Ошибка запуска обработки:\n{exc}")

    def _on_job_progress(self, job_id: str, current: int, total: int, stage: str) -> None:
        """Progress callback — runs on a non-UI thread. Marshal into GUI thread."""
        QTimer.singleShot(
            0, lambda: self._apply_job_progress(job_id, current, total, stage)
        )

    def _apply_job_progress(self, job_id: str, current: int, total: int, stage: str) -> None:
        try:
            self._queue_manager.update_progress(job_id, current, total)
        except Exception as exc:  # noqa: BLE001
            logger.debug("update_progress raised: %s", exc)
        # Update the ProgressWidget too
        item = self._queue_manager.get(job_id) if hasattr(self._queue_manager, "get") else None
        name = item.file_name if item is not None else ""
        self.progress_widget.set_current_file(f"{name} — {stage}", current, total)
        # Periodically refresh the recovery snapshot so a crash resumes from ~ here
        if item is not None and current % 5 == 0:
            try:
                self._recovery.snapshot(item)
            except Exception:  # noqa: BLE001
                pass

    def _on_job_complete(self, job_id: str, result: object) -> None:
        # Called from worker thread — marshal into GUI thread
        QTimer.singleShot(0, lambda: self._apply_job_result(job_id, result))

    def _on_job_failed(self, job_id: str, exc: BaseException) -> None:
        QTimer.singleShot(
            0, lambda: self._queue_manager.update_status(job_id, __import__("src.shared.types", fromlist=["JobStatus"]).JobStatus.FAILED, str(exc))
        )

    def _apply_job_result(self, job_id: str, result: object) -> None:
        self._last_result = result
        self.results_panel.set_result(result)  # type: ignore[arg-type]
        # Propagate status into the queue
        try:
            from src.core.models import JobResult as _JR

            if isinstance(result, _JR):
                self._queue_manager.update_status(
                    job_id, result.status, result.error or ""
                )
        except Exception:  # noqa: BLE001
            logger.debug("Could not update queue status", exc_info=True)
        # Remove from recovery snapshot
        try:
            self._recovery.remove(job_id)
        except Exception:  # noqa: BLE001
            pass

    def _on_save(self) -> None:
        if self._last_result is None:
            QMessageBox.information(self, APP_NAME, "Нет готовых результатов.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить как", "", "PDF (*.pdf);;Text (*.txt);;Word (*.docx)"
        )
        if not path:
            return
        fmt = ExportFormat.PDF
        low = path.lower()
        if low.endswith(".txt"):
            fmt = ExportFormat.TXT
        elif low.endswith(".docx"):
            fmt = ExportFormat.DOCX
        try:
            self._export_manager.export(self._last_result, Path(path), fmt)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Ошибка экспорта: {exc}")

    def _on_export_requested(self, fmt: ExportFormat, path: Path | None) -> None:
        if self._last_result is None:
            QMessageBox.information(self, APP_NAME, "Нет готовых результатов.")
            return
        if path is None:
            target, _ = QFileDialog.getSaveFileName(self, "Сохранить", "", "Все файлы (*.*)")
            if not target:
                return
            path = Path(target)
        try:
            self._export_manager.export(self._last_result, path, fmt)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Ошибка экспорта: {exc}")

    # ------------------------------------------------------------ profile ops
    def _on_save_profile(self) -> None:
        if self._current_profile is None:
            return
        profile = self._current_profile_with_overrides()
        if profile.builtin:
            QMessageBox.information(
                self, APP_NAME,
                "Встроенный профиль нельзя изменить. Дублируйте его и сохраните под новым именем.",
            )
            return
        try:
            self._profile_manager.save(profile)
            QMessageBox.information(self, APP_NAME, f"Профиль '{profile.name}' сохранён.")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Ошибка сохранения: {exc}")

    def _on_duplicate_profile(self) -> None:
        if self._current_profile is None:
            return
        from PySide6.QtWidgets import QInputDialog

        new_name, ok = QInputDialog.getText(
            self, "Дублировать профиль", "Имя нового профиля:"
        )
        if not ok or not new_name.strip():
            return
        try:
            new_prof = self._profile_manager.duplicate(self._current_profile.name, new_name.strip())
            self._load_profiles_to_combobox()
            idx = self.profile_combo.findData(new_prof.name)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Ошибка дублирования: {exc}")

    def _on_delete_profile(self) -> None:
        if self._current_profile is None:
            return
        if self._current_profile.builtin:
            QMessageBox.information(self, APP_NAME, "Встроенный профиль нельзя удалить.")
            return
        resp = QMessageBox.question(
            self, APP_NAME, f"Удалить профиль '{self._current_profile.name}'?"
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        try:
            self._profile_manager.delete(self._current_profile.name)
            self._load_profiles_to_combobox()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Ошибка удаления: {exc}")

    def _on_import_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Импорт профиля", "", "JSON (*.json)")
        if not path:
            return
        try:
            profile = self._profile_manager.import_profile(Path(path))
            self._load_profiles_to_combobox()
            idx = self.profile_combo.findData(profile.name)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Ошибка импорта: {exc}")

    def _on_export_profile(self) -> None:
        if self._current_profile is None:
            return
        default_name = f"{self._current_profile.name}.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт профиля", default_name, "JSON (*.json)"
        )
        if not path:
            return
        try:
            self._profile_manager.export_profile(self._current_profile.name, Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Ошибка экспорта: {exc}")

    # ------------------------------------------------------------ recovery
    def prompt_recovery(self) -> None:
        """Check for leftover recovery snapshots and offer to resume them."""
        try:
            pending = self._recovery.list_pending()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not inspect recovery: %s", exc)
            return
        if not pending:
            return
        names = "\n".join(f" • {Path(p.config.input_path).name}" for p in pending if p.config)
        resp = QMessageBox.question(
            self,
            APP_NAME,
            (
                f"Обнаружены незавершённые задания ({len(pending)}):\n\n{names}\n\n"
                "Возобновить их обработку сейчас?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            try:
                self._recovery.clear_all()
            except Exception:  # noqa: BLE001
                logger.debug("clear_all recovery failed", exc_info=True)
            return
        for item in pending:
            try:
                self._queue_manager.add(item)
                self._submit_job(item)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Could not resume job %s: %s", item.job_id, exc)

    # ------------------------------------------------------------ recent
    def _add_to_recent(self, path: Path) -> None:
        try:
            settings = self._settings_storage.load()
            p = str(path.resolve())
            recent = [r for r in settings.recent_files if r != p]
            recent.insert(0, p)
            settings.recent_files = recent[:10]
            self._settings_storage.save(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update recent files: %s", exc)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.clear()
        try:
            settings = self._settings_storage.load()
            recent = list(settings.recent_files)
        except Exception:  # noqa: BLE001
            recent = []
        if not recent:
            act = QAction("(пусто)", self)
            act.setEnabled(False)
            self.recent_menu.addAction(act)
            return
        for idx, item in enumerate(recent):
            label = f"{idx + 1}. {Path(item).name}"
            act = QAction(label, self)
            act.setToolTip(item)
            act.triggered.connect(lambda _checked=False, p=item: self._open_pdf(Path(p)))
            self.recent_menu.addAction(act)
        self.recent_menu.addSeparator()
        clear_act = QAction("Очистить список", self)
        clear_act.triggered.connect(self._clear_recent)
        self.recent_menu.addAction(clear_act)

    def _clear_recent(self) -> None:
        try:
            settings = self._settings_storage.load()
            settings.recent_files = []
            self._settings_storage.save(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not clear recent files: %s", exc)
        self._rebuild_recent_menu()

    # ------------------------------------------------------------ preferences
    def _on_preferences(self) -> None:
        dlg = PreferencesDialog(self._settings_storage, self)
        if dlg.exec() == dlg.DialogCode.Accepted:
            QMessageBox.information(
                self,
                APP_NAME,
                "Настройки сохранены. Некоторые изменения (например, число воркеров) "
                "вступят в силу после перезапуска.",
            )

    # ------------------------------------------------------------ help
    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            APP_NAME,
            f"<h3>{APP_NAME} {APP_VERSION}</h3>"
            "<p>Профессиональное OCR-приложение на базе Tesseract 5.5 и OCRmyPDF.</p>"
            "<p>Лицензия: MIT.</p>",
        )

    def _on_open_log(self) -> None:
        from src.ui.log_viewer import LogViewer

        log_path = LOGS_DIR / LOG_FILE_NAME
        try:
            dlg = LogViewer(log_path, self)
            dlg.exec()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Log viewer failed: %s", exc)
            # Fall back to system default
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_path)))

    # ------------------------------------------------------------ misc
    def _update_resource_usage(self) -> None:
        try:
            import psutil  # lazy
            mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu = psutil.cpu_percent(interval=None)
            self.status_resource_label.setText(f"RAM: {mem_mb:.0f} МБ | CPU: {cpu:.0f}%")
        except Exception:
            self.status_resource_label.setText("Ready")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths: list[Path] = []
        for url in event.mimeData().urls():
            local = Path(url.toLocalFile())
            if local.is_dir():
                paths.extend(sorted(local.rglob("*.pdf")))
            elif local.suffix.lower() == ".pdf":
                paths.append(local)
        if paths:
            self._enqueue_files(paths)
            event.acceptProposedAction()

    # ------------------------------------------------------------ persist
    def _restore_window_state(self) -> None:
        try:
            settings = self._settings_storage.load()
            if settings.window_geometry:
                self.restoreGeometry(QByteArray.fromBase64(settings.window_geometry.encode()))
            if settings.window_state:
                self.restoreState(QByteArray.fromBase64(settings.window_state.encode()))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not restore window state: %s", exc)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        try:
            settings = self._settings_storage.load()
            settings.window_geometry = bytes(self.saveGeometry().toBase64()).decode()
            settings.window_state = bytes(self.saveState().toBase64()).decode()
            self._settings_storage.save(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save window state: %s", exc)
        try:
            self._parallel_processor.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)
