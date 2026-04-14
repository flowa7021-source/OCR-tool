"""Main window tying together viewers, panels, queue, and menus."""

from __future__ import annotations

import contextlib
import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray, Qt, QTimer
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

from src.application.recovery_manager import RecoveryManager
from src.core.models import JobResult, OCRJobConfig, ProfileData, QueueItem
from src.infrastructure.file_utils import safe_unique_path, suggest_output_path
from src.shared.constants import (
    APP_NAME,
    APP_VERSION,
    IMPORT_FILE_FILTERS,
    LOG_FILE_NAME,
    LOGS_DIR,
)
from src.shared.types import ExportFormat, JobStatus
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
        profile_manager: ProfileManager,
        queue_manager: QueueManager,
        parallel_processor: ParallelProcessor,
        export_manager: ExportManager,
        settings_storage: SettingsStorage,
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
        # One-shot per-session flag: don't nag the user again after they
        # acknowledged the active max_pages preview limit.
        self._max_pages_warned: bool = False
        # Tray notifier is built lazily in setupUi once an icon is available.
        self._tray: object | None = None

        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setAcceptDrops(True)
        # Start comfortably large on modern monitors but remain usable
        # on 1366x768 laptops via the minimum size below.
        self.resize(1680, 1000)
        self.setMinimumSize(1200, 780)
        self.setWindowIcon(app_icon())
        # Tray notifier uses the same icon so completion toasts match.
        try:
            from src.ui.tray_notifier import TrayNotifier

            self._tray = TrayNotifier(app_icon(), self)
        except Exception:  # noqa: BLE001
            logger.debug("Tray notifier unavailable", exc_info=True)
            self._tray = None

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

        # Minimum widths so nothing collapses into an unusable sliver when
        # the user drags the splitter handles.
        self.pdf_viewer.setMinimumWidth(520)
        self.settings_panel.setMinimumWidth(420)
        self.preprocessing_panel.setMinimumWidth(420)
        self.postprocess_panel.setMinimumWidth(420)

        # Right side: settings on top, then preprocessing / postprocessing tabs below
        right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(self.settings_panel)

        self.right_tabs = QTabWidget(self)
        self.right_tabs.setMinimumWidth(440)
        self.right_tabs.setDocumentMode(True)
        self.right_tabs.addTab(self.preprocessing_panel, "Предобработка")
        self.right_tabs.addTab(self.postprocess_panel, "Постобработка")
        right_splitter.addWidget(self.right_tabs)
        # Explicit initial pixel sizes — settings on top is short, tabs below tall.
        right_splitter.setSizes([320, 620])
        right_splitter.setStretchFactor(0, 0)
        right_splitter.setStretchFactor(1, 1)

        # Central horizontal splitter: viewer | right
        central_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        central_splitter.setChildrenCollapsible(False)
        central_splitter.addWidget(self.pdf_viewer)
        central_splitter.addWidget(right_splitter)
        # Give the viewer ~60% of the initial window width and reserve
        # ~40% (min 440) for the config panels. Explicit sizes beat
        # pure stretch factors when the content's sizeHint is small.
        central_splitter.setSizes([960, 640])
        central_splitter.setStretchFactor(0, 3)
        central_splitter.setStretchFactor(1, 2)
        self.setCentralWidget(central_splitter)

    def _build_docks(self) -> None:
        # Queue dock (bottom)
        queue_wrapper = QWidget(self)
        queue_wrapper.setMinimumHeight(220)
        q_layout = QVBoxLayout(queue_wrapper)
        q_layout.setContentsMargins(4, 4, 4, 4)
        q_layout.setSpacing(6)
        q_layout.addWidget(self.progress_widget)
        q_layout.addWidget(self.queue_panel, 1)

        self.queue_dock = QDockWidget("Очередь обработки", self)
        self.queue_dock.setObjectName("queueDock")
        self.queue_dock.setWidget(queue_wrapper)
        self.queue_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.queue_dock.setMinimumHeight(240)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.queue_dock)

        # Results dock (bottom, tabbed with queue)
        self.results_panel.setMinimumHeight(220)
        self.results_dock = QDockWidget("Результаты распознавания", self)
        self.results_dock.setObjectName("resultsDock")
        self.results_dock.setWidget(self.results_panel)
        self.results_dock.setMinimumHeight(240)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.results_dock)
        self.tabifyDockWidget(self.queue_dock, self.results_dock)
        self.queue_dock.raise_()

        # Reserve enough vertical real estate for the bottom dock by default
        # (~260 px), leaving plenty for the viewer/panels above.
        self.resizeDocks([self.queue_dock], [260], Qt.Orientation.Vertical)

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

        self.action_pause = QAction(load_icon("pause"), "Пауза / Возобновить", self)
        self.action_pause.setToolTip(
            "Поставить на паузу следующие задания в очереди (запущенные дойдут до конца)"
        )
        self.action_pause.setShortcut(QKeySequence("Ctrl+P"))
        self.action_pause.triggered.connect(self._on_toggle_pause)
        tb.addAction(self.action_pause)

        self.action_stop = QAction(load_icon("stop"), "Стоп", self)
        self.action_stop.setToolTip("Отменить все ожидающие задания очереди")
        self.action_stop.setShortcut(QKeySequence("Ctrl+."))
        self.action_stop.triggered.connect(self._on_stop)
        tb.addAction(self.action_stop)

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
        backup_export_act = QAction("Экспорт конфигурации…", self)
        backup_export_act.setToolTip(
            "Сохранить настройки и пользовательские профили в ZIP "
            "(для переноса на другую машину)"
        )
        backup_export_act.triggered.connect(self._on_export_config_backup)
        file_menu.addAction(backup_export_act)
        backup_import_act = QAction("Импорт конфигурации…", self)
        backup_import_act.setToolTip(
            "Восстановить настройки и профили из ранее сохранённого ZIP"
        )
        backup_import_act.triggered.connect(self._on_import_config_backup)
        file_menu.addAction(backup_import_act)
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

        # OCR-engine submenu (download + manage HTR weights)
        engine_menu = menubar.addMenu("&Движок OCR")
        download_got_act = QAction("Скачать GOT-OCR 2.0 (рукописный)", self)
        download_got_act.triggered.connect(self._on_download_got_model)
        engine_menu.addAction(download_got_act)
        remove_got_act = QAction("Удалить модель GOT-OCR 2.0", self)
        remove_got_act.triggered.connect(self._on_remove_got_model)
        engine_menu.addAction(remove_got_act)

        view_menu = menubar.addMenu("&Вид")
        view_menu.addAction(self.queue_dock.toggleViewAction())
        view_menu.addAction(self.results_dock.toggleViewAction())
        view_menu.addSeparator()
        self.action_overlay = QAction("Подсветить распознанный текст", self)
        self.action_overlay.setCheckable(True)
        self.action_overlay.setShortcut(QKeySequence("Ctrl+H"))
        self.action_overlay.setToolTip(
            "Показать/скрыть bounding box распознанных слов поверх скана "
            "(цвет по уровню confidence)"
        )
        self.action_overlay.toggled.connect(self._on_toggle_overlay)
        view_menu.addAction(self.action_overlay)

        view_menu.addSeparator()
        self.action_light_theme = QAction("Светлая тема", self)
        self.action_light_theme.setCheckable(True)
        self.action_light_theme.setToolTip(
            "Переключить между тёмной и светлой темой (без перезапуска)"
        )
        try:
            current_theme = (self._settings_storage.load().theme or "dark").lower()
        except Exception:  # noqa: BLE001
            current_theme = "dark"
        self.action_light_theme.setChecked(current_theme == "light")
        self.action_light_theme.toggled.connect(self._on_toggle_theme)
        view_menu.addAction(self.action_light_theme)

        help_menu = menubar.addMenu("&Справка")
        about_act = QAction("О программе", self)
        about_act.setShortcut(QKeySequence(Qt.Key.Key_F1))
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)
        log_act = QAction("Открыть лог-файл", self)
        log_act.triggered.connect(self._on_open_log)
        help_menu.addAction(log_act)
        diag_act = QAction("Экспорт диагностики…", self)
        diag_act.triggered.connect(self._on_export_diagnostics)
        help_menu.addAction(diag_act)
        update_act = QAction("Проверить обновления", self)
        update_act.triggered.connect(self._on_check_updates)
        help_menu.addAction(update_act)

        # Common keyboard shortcuts: Ctrl+Q quits, F1 opens About,
        # Escape closes any modeless dialog that we spawn. Escape in
        # modal QDialogs is handled by Qt's default reject()-to-Esc
        # wiring already.
        quit_act = QAction("Выход", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        self.addAction(quit_act)

        # Profile quick-switch shortcuts
        for i in range(4):
            act = QAction(f"Профиль {i + 1}", self)
            act.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            act.triggered.connect(lambda _checked=False, idx=i: self._switch_profile_by_index(idx))
            self.addAction(act)

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        self.status_queue_label = QLabel("Очередь: 0")
        self.status_profile_label = QLabel("Профиль: —")
        self.status_resource_label = QLabel("")
        sb.addPermanentWidget(self.status_queue_label)
        sb.addPermanentWidget(self.status_profile_label)
        sb.addPermanentWidget(self.status_resource_label)
        self._resource_timer = QTimer(self)
        self._resource_timer.setInterval(2000)
        self._resource_timer.timeout.connect(self._update_resource_usage)
        self._resource_timer.timeout.connect(self._update_queue_stats)
        self._resource_timer.start()
        self._update_resource_usage()
        self._update_queue_stats()

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
        self.queue_panel.retry_requested.connect(self._on_retry_job)
        self.queue_panel.open_output_requested.connect(self._on_open_job_output)
        self.results_panel.export_requested.connect(self._on_export_requested)
        self.results_panel.open_pdf_requested.connect(self._on_open_pdf_result)

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
        with contextlib.suppress(Exception):
            self._profile_manager.set_current(profile.name)

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

    def _warn_if_max_pages_active(self, profile: ProfileData, file_count: int) -> None:
        """Show a one-shot info dialog if the profile truncates long PDFs.

        The pipeline already logs the truncation (see pipeline.py around
        line 152), but unless the user checks the log they won't realise
        that only the first N pages of a 500-page scan got OCR'd. This
        preflight banner makes it explicit — once per session per setting.
        """
        try:
            max_pages = int(getattr(profile.ocr, "max_pages", 0) or 0)
        except (TypeError, ValueError):
            return
        if max_pages <= 0 or self._max_pages_warned:
            return
        self._max_pages_warned = True
        plural = "файл" if file_count == 1 else "файлов"
        QMessageBox.information(
            self,
            APP_NAME,
            (
                f"В текущем профиле активен лимит «предпросмотра»: "
                f"будут обработаны только первые {max_pages} страниц каждого "
                f"из {file_count} {plural}. Снять ограничение: "
                f"«Настройки вывода → Макс. страниц = 0»."
            ),
        )

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
        from src.shared.validators import ValidationError, validate_pdf_path

        try:
            path = validate_pdf_path(path)
        except ValidationError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
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

    def _on_toggle_pause(self) -> None:
        """Toggle paused state for all PENDING jobs in the queue.

        Already-running jobs are not affected — the process pool cannot
        interrupt a worker mid-page safely. New submissions won't happen
        while paused because QueueManager returns nothing from next_pending().
        """
        items = self._queue_manager.list_items()
        pending = [i for i in items if i.status is JobStatus.PENDING]
        paused = [i for i in items if i.status is JobStatus.PAUSED]
        if pending:
            for item in pending:
                self._queue_manager.pause(item.job_id)
            logger.info("Paused %d queued job(s)", len(pending))
        elif paused:
            for item in paused:
                self._queue_manager.resume(item.job_id)
                self._submit_job(item)
            logger.info("Resumed %d paused job(s)", len(paused))

    def _on_stop(self) -> None:
        """Cancel every pending/running job we can reach."""
        resp = QMessageBox.question(
            self,
            APP_NAME,
            "Отменить все ожидающие задания очереди? "
            "Уже запущенные дойдут до конца текущей страницы.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        cancelled = 0
        try:
            cancelled = self._parallel_processor.cancel_all()
        except Exception as exc:  # noqa: BLE001
            logger.warning("cancel_all failed: %s", exc)

        for item in self._queue_manager.list_items():
            if item.status in (JobStatus.PENDING, JobStatus.PAUSED):
                self._queue_manager.cancel(item.job_id)

        self.statusBar().showMessage(f"Отменено заданий: {cancelled}", 3000)

    def _enqueue_files(self, paths: list[Path]) -> None:
        profile = self._current_profile_with_overrides()
        self._warn_if_max_pages_active(profile, len(paths))
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
            self._queue_manager.update_status(item.job_id, JobStatus.RUNNING)
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
            with contextlib.suppress(Exception):
                self._recovery.snapshot(item)

    def _on_job_complete(self, job_id: str, result: object) -> None:
        # Called from worker thread — marshal into GUI thread
        QTimer.singleShot(0, lambda: self._apply_job_result(job_id, result))

    def _on_job_failed(self, job_id: str, exc: BaseException) -> None:
        QTimer.singleShot(
            0,
            lambda: self._queue_manager.update_status(
                job_id, JobStatus.FAILED, str(exc)
            ),
        )

    # ---- Queue-panel context-menu actions -----------------------------

    def _on_retry_job(self, job_id: str) -> None:
        """Re-submit a previously-failed / cancelled job to the pool.

        Resets its progress and status to PENDING, then submits anew.
        For successfully-completed jobs this still works and produces
        a fresh OCR pass (useful if the profile changed in between).
        """
        item = self._queue_manager.get(job_id)
        if item is None or item.config is None:
            logger.warning("Retry requested for unknown job %s", job_id)
            return
        self._queue_manager.update_progress(job_id, 0, item.progress_total or 0)
        self._queue_manager.update_status(job_id, JobStatus.PENDING, "")
        logger.info("Retrying job %s", job_id)
        self._submit_job(item)

    def _on_open_job_output(self, job_id: str) -> None:
        """Open the searchable PDF for ``job_id`` with the system handler."""
        item = self._queue_manager.get(job_id)
        if item is None or item.config is None:
            return
        path = Path(item.config.output_path)
        if not path.exists():
            QMessageBox.information(
                self, APP_NAME,
                f"Результирующий файл ещё не готов:\n{path}",
            )
            return
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to open job output: %s", exc)
            QMessageBox.critical(self, APP_NAME, f"Не удалось открыть файл: {exc}")

    def _apply_job_result(self, job_id: str, result: object) -> None:
        self._last_result = result
        self.results_panel.set_result(result)  # type: ignore[arg-type]
        # Feed word boxes to the viewer so the overlay works on the result PDF.
        try:
            self._populate_overlay_from_result(result)
        except Exception:  # noqa: BLE001
            logger.debug("Overlay population failed", exc_info=True)
        # Propagate status into the queue
        try:
            if isinstance(result, JobResult):
                self._queue_manager.update_status(
                    job_id, result.status, result.error or ""
                )
        except Exception:  # noqa: BLE001
            logger.debug("Could not update queue status", exc_info=True)
        # Remove from recovery snapshot
        with contextlib.suppress(Exception):
            self._recovery.remove(job_id)
        # Fire a system-tray toast so the user knows a long job finished
        # even if the window isn't focused. Skip toasts when the user has
        # turned them off, when the window is already active (redundant),
        # or when the queue is still grinding through more items (we only
        # want a single summary ping per batch).
        try:
            self._maybe_emit_completion_toast(result)
        except Exception:  # noqa: BLE001
            logger.debug("Completion toast failed", exc_info=True)

    def _maybe_emit_completion_toast(self, result: object) -> None:
        """Fire a tray toast on batch completion or per-job failure.

        We deliberately stay quiet when the main window is already active —
        the user is watching the progress widget and a toast would be noise.
        Success toasts are suppressed while more jobs are still pending to
        avoid spamming one toast per file in a large batch; one final
        summary toast is emitted when the queue drains. Failures always
        surface immediately so the user doesn't miss an error.
        """
        if self._tray is None:
            return
        if not isinstance(result, JobResult):
            return
        try:
            notify = self._settings_storage.load().notify_on_complete
        except Exception:  # noqa: BLE001
            notify = True
        if not notify:
            return
        if self.isActiveWindow():
            return

        name = Path(result.output_path).name if result.output_path else "задание"
        if result.status is JobStatus.FAILED:
            self._tray.notify_failure(  # type: ignore[attr-defined]
                APP_NAME,
                f"Ошибка при обработке: {name}",
            )
            return
        if result.status is not JobStatus.COMPLETED:
            return
        # Anything still waiting? Hold the success toast until the batch drains.
        remaining = sum(
            1
            for it in self._queue_manager.list_items()
            if it.status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.PAUSED)
        )
        if remaining > 0:
            return
        done_count = sum(
            1
            for it in self._queue_manager.list_items()
            if it.status is JobStatus.COMPLETED
        )
        msg = (
            f"Файл готов: {name}"
            if done_count <= 1
            else f"Обработано файлов: {done_count}"
        )
        self._tray.notify_complete(APP_NAME, msg)  # type: ignore[attr-defined]

    def _on_save(self) -> None:
        if self._last_result is None:
            QMessageBox.information(self, APP_NAME, "Нет готовых результатов.")
            return
        default_name = Path(self._last_result.output_path).name
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат как",
            default_name,
            "PDF (*.pdf);;Text (*.txt);;Word (*.docx)",
        )
        if not path:
            return
        fmt = ExportFormat.PDF
        low = path.lower()
        if low.endswith(".txt"):
            fmt = ExportFormat.TXT
        elif low.endswith(".docx"):
            fmt = ExportFormat.DOCX
        encoding = self.results_panel.txt_encoding()
        try:
            self._export_manager.export(
                self._last_result, Path(path), fmt, encoding=encoding
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Ошибка экспорта: {exc}")
            return
        self.statusBar().showMessage(f"Сохранено: {path}", 5000)

    def _on_export_requested(self, fmt: ExportFormat, path: Path | None) -> None:
        if self._last_result is None:
            QMessageBox.information(self, APP_NAME, "Нет готовых результатов.")
            return
        if path is None and fmt is not ExportFormat.CLIPBOARD:
            default_ext = {
                ExportFormat.TXT: "txt",
                ExportFormat.DOCX: "docx",
                ExportFormat.PDF: "pdf",
            }.get(fmt, "")
            filter_map = {
                ExportFormat.TXT: "Text (*.txt)",
                ExportFormat.DOCX: "Word (*.docx)",
                ExportFormat.PDF: "PDF (*.pdf)",
            }
            # For PDF default to the existing job output filename (so users
            # see e.g. "document_ocr.pdf", same as when they let the pipeline
            # pick the path). For TXT/DOCX reuse the input stem + extension.
            if fmt is ExportFormat.PDF:
                default_name = Path(self._last_result.output_path).name
            elif default_ext:
                default_name = Path(self._last_result.input_path).with_suffix(f".{default_ext}").name
            else:
                default_name = ""
            dialog_title = (
                "Сохранить PDF как" if fmt is ExportFormat.PDF else "Сохранить"
            )
            target, _ = QFileDialog.getSaveFileName(
                self, dialog_title, default_name, filter_map.get(fmt, "Все файлы (*.*)")
            )
            if not target:
                return
            path = Path(target)
        encoding = self.results_panel.txt_encoding()
        try:
            self._export_manager.export(
                self._last_result,
                path if path is not None else Path(""),
                fmt,
                encoding=encoding,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Ошибка экспорта: {exc}")

    # ------------------------------------------------------------ HTR model
    def _on_download_got_model(self) -> None:
        """Open a download dialog for the GOT-OCR 2.0 weights."""
        try:
            from src.application.engines.registry import reset_cache
            from src.infrastructure.model_manager import GOT_OCR2_SPEC, ModelManager
            from src.ui.model_download_dialog import ModelDownloadDialog
        except ImportError as exc:
            QMessageBox.critical(self, APP_NAME, f"Не удалось загрузить компоненты HTR: {exc}")
            return
        manager = ModelManager()
        dlg = ModelDownloadDialog(manager, GOT_OCR2_SPEC, self)
        dlg.exec()
        # Refresh the engine combo so the just-downloaded engine becomes
        # selectable without restart.
        reset_cache()
        if self._current_profile is not None:
            self.settings_panel.set_config(self._current_profile.ocr)

    def _on_remove_got_model(self) -> None:
        """Delete the GOT-OCR 2.0 weights from disk."""
        from src.infrastructure.model_manager import GOT_OCR2_SPEC, ModelManager

        manager = ModelManager()
        if not manager.is_available(GOT_OCR2_SPEC.model_id):
            QMessageBox.information(
                self, APP_NAME, "Модель GOT-OCR 2.0 не установлена."
            )
            return
        resp = QMessageBox.question(
            self, APP_NAME,
            f"Удалить модель {GOT_OCR2_SPEC.label} (~580 МБ)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        try:
            manager.remove(GOT_OCR2_SPEC.model_id)
            from src.application.engines.registry import reset_cache

            reset_cache()
            QMessageBox.information(self, APP_NAME, "Модель удалена.")
        except OSError as exc:
            QMessageBox.critical(self, APP_NAME, f"Не удалось удалить: {exc}")

    def _open_paths_from_secondary(self, paths: list[Path]) -> None:
        """Called by SingleInstanceGuard when a second instance forwards argv.

        Raises the existing window to the front and opens each PDF.
        """
        self.raise_()
        self.activateWindow()
        for p in paths:
            if p.suffix.lower() == ".pdf" and p.exists():
                self._open_pdf(p)

    def _on_open_pdf_result(self) -> None:
        """Open the produced searchable PDF with the system default handler."""
        if self._last_result is None:
            QMessageBox.information(self, APP_NAME, "Нет готовых результатов.")
            return
        pdf_path = Path(self._last_result.output_path)
        if not pdf_path.exists():
            QMessageBox.critical(
                self, APP_NAME,
                f"PDF не найден:\n{pdf_path}\n\nВозможно, файл был удалён или перемещён.",
            )
            return
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf_path)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Open PDF failed: %s", exc)
            QMessageBox.critical(self, APP_NAME, f"Не удалось открыть PDF: {exc}")

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

    # ------------------------------------------------------------ overlay
    def _on_toggle_overlay(self, checked: bool) -> None:
        """Menu callback: switch OCR overlay on the PDF viewer."""
        self.pdf_viewer.set_overlay_visible(bool(checked))

    # ------------------------------------------------------------ theme
    def _on_toggle_theme(self, checked: bool) -> None:
        """Menu callback: re-apply the light or dark stylesheet at runtime.

        The new choice is persisted to ``SettingsStorage`` so the next
        launch honours it (see ``src/app.py``). Unchecking means dark.
        """
        from PySide6.QtWidgets import QApplication

        from src.ui.theme import apply_theme

        kind = "light" if checked else "dark"
        app = QApplication.instance()
        if app is not None:
            try:
                apply_theme(app, kind)
            except Exception as exc:  # noqa: BLE001
                logger.exception("apply_theme(%s) failed: %s", kind, exc)
        try:
            settings = self._settings_storage.load()
            settings.theme = kind
            self._settings_storage.save(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist theme preference: %s", exc)

    def _populate_overlay_from_result(self, result: object) -> None:
        """Extract word boxes from the produced searchable PDF and feed the viewer.

        The OCRmyPDF output already contains an invisible text layer with the
        word positions; PyMuPDF ``page.get_text("words")`` gives us
        ``(x0, y0, x1, y1, text, ...)`` tuples in PDF user-space points.
        We have no direct confidence from that layer, so we fall back to the
        per-page mean confidence stored on the JobResult.
        """
        from src.core.models import JobResult

        if not isinstance(result, JobResult):
            return
        pdf_path = Path(result.output_path)
        if not pdf_path.exists():
            return
        try:
            import fitz
        except ImportError:
            return
        try:
            doc = fitz.open(str(pdf_path))
        except Exception:  # noqa: BLE001
            return
        try:
            self.pdf_viewer.clear_word_boxes()
            for page_index in range(doc.page_count):
                pg = result.pages[page_index] if page_index < len(result.pages) else None
                default_conf = pg.mean_confidence if pg is not None else 80.0
                boxes: list[tuple[float, float, float, float, float]] = []
                try:
                    words = doc[page_index].get_text("words")
                except Exception:  # noqa: BLE001
                    words = []
                for word in words:
                    try:
                        x0, y0, x1, y1 = float(word[0]), float(word[1]), float(word[2]), float(word[3])
                    except (TypeError, ValueError, IndexError):
                        continue
                    boxes.append((x0, y0, x1 - x0, y1 - y0, default_conf))
                self.pdf_viewer.set_word_boxes(page_index + 1, boxes)
        finally:
            doc.close()

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
            settings.add_recent_file(str(path.resolve()))
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
                "Настройки сохранены. Тема применяется сразу; число воркеров "
                "вступит в силу после перезапуска.",
            )

    # --------------------------------------------------- config backup/restore
    def _on_export_config_backup(self) -> None:
        """Save settings + user profiles into a portable ZIP bundle."""
        from src.application.config_backup import default_backup_name, export_config
        from src.infrastructure.config_storage import ProfileStorage

        target, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт конфигурации",
            default_backup_name(),
            "Zip (*.zip)",
        )
        if not target:
            return
        try:
            path = export_config(
                Path(target),
                settings_storage=self._settings_storage,
                profile_storage=ProfileStorage(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Config export failed: %s", exc)
            QMessageBox.critical(
                self, APP_NAME, f"Не удалось выполнить экспорт:\n{exc}"
            )
            return
        QMessageBox.information(
            self, APP_NAME, f"Конфигурация сохранена:\n{path}"
        )

    def _on_import_config_backup(self) -> None:
        """Restore settings + user profiles from a previously exported ZIP."""
        from src.application.config_backup import BackupFormatError, import_config
        from src.infrastructure.config_storage import ProfileStorage

        source, _ = QFileDialog.getOpenFileName(
            self,
            "Импорт конфигурации",
            "",
            "Zip (*.zip);;Все файлы (*)",
        )
        if not source:
            return

        confirm = QMessageBox.question(
            self,
            APP_NAME,
            (
                "Импорт конфигурации перезапишет текущие настройки и "
                "одноимённые пользовательские профили. Продолжить?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            result = import_config(
                Path(source),
                settings_storage=self._settings_storage,
                profile_storage=ProfileStorage(),
            )
        except BackupFormatError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Config import failed: %s", exc)
            QMessageBox.critical(
                self, APP_NAME, f"Не удалось выполнить импорт:\n{exc}"
            )
            return

        # Refresh the profile combo with restored entries.
        try:
            self._load_profiles_to_combobox()
        except Exception:  # noqa: BLE001
            logger.debug("profile combo reload after import failed", exc_info=True)

        # Re-apply the (possibly new) theme immediately.
        if result.settings_restored:
            try:
                from PySide6.QtWidgets import QApplication

                from src.ui.theme import apply_theme

                app = QApplication.instance()
                if app is not None:
                    theme = (self._settings_storage.load().theme or "dark").lower()
                    apply_theme(app, theme)
                    if hasattr(self, "action_light_theme"):
                        self.action_light_theme.blockSignals(True)
                        self.action_light_theme.setChecked(theme == "light")
                        self.action_light_theme.blockSignals(False)
            except Exception:  # noqa: BLE001
                logger.debug("Post-import theme re-apply failed", exc_info=True)

        summary = [
            f"Схема бэкапа: v{result.schema_version}",
            f"Настройки: {'восстановлены' if result.settings_restored else 'не найдены в архиве'}",
            f"Новых профилей: {len(result.profiles_added)}",
            f"Перезаписано профилей: {len(result.profiles_overwritten)}",
        ]
        QMessageBox.information(self, APP_NAME, "\n".join(summary))

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
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_path)))

    def _on_export_diagnostics(self) -> None:
        """Build a privacy-safe diagnostics zip for bug reports."""
        from datetime import datetime

        default_name = (
            f"ocr-studio-diagnostics-{datetime.now():%Y%m%d-%H%M%S}.zip"
        )
        target, _ = QFileDialog.getSaveFileName(
            self, "Экспорт диагностики", default_name, "Zip (*.zip)"
        )
        if not target:
            return
        try:
            from src.application.diagnostics import build_diagnostics_zip

            path = build_diagnostics_zip(Path(target))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Diagnostics export failed: %s", exc)
            QMessageBox.critical(
                self, APP_NAME, f"Не удалось собрать диагностику: {exc}"
            )
            return
        QMessageBox.information(
            self, APP_NAME,
            f"Диагностика сохранена:\n{path}\n\n"
            "Файл можно приложить к issue на GitHub.",
        )

    def _on_check_updates(self, *, quiet: bool = False) -> None:
        """Manual update check. ``quiet=True`` suppresses the
        'you are up-to-date' dialog (used for the automatic startup
        check, where we only want to bug the user about real upgrades).
        """
        from src.application.update_checker import check_async

        def _report(info) -> None:  # noqa: ANN001
            QTimer.singleShot(0, lambda: self._show_update_info(info, quiet=quiet))

        check_async(_report)

    def _show_update_info(self, info, *, quiet: bool) -> None:  # noqa: ANN001
        if info is None:
            if not quiet:
                QMessageBox.information(
                    self, APP_NAME, "Не удалось проверить обновления."
                )
            return
        if info.is_newer:
            # Persistent status-bar hint + offer to open the release page.
            self.statusBar().showMessage(
                f"Доступна новая версия: {info.latest_version}", 10_000
            )
            resp = QMessageBox.information(
                self, APP_NAME,
                (
                    f"Доступна новая версия {info.latest_version} "
                    f"(текущая: {info.current_version}).\n\n"
                    f"{(info.body or '').strip()[:500]}\n\n"
                    "Открыть страницу релиза?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if resp == QMessageBox.StandardButton.Yes:
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices

                QDesktopServices.openUrl(QUrl(info.release_url))
        elif not quiet:
            QMessageBox.information(
                self, APP_NAME,
                f"У вас актуальная версия ({info.current_version}).",
            )

    # ------------------------------------------------------------ misc
    def _update_queue_stats(self) -> None:
        """Refresh the queue-summary label in the status bar."""
        try:
            from collections import Counter

            counts = Counter(i.status for i in self._queue_manager.list_items())
            total = sum(counts.values())
            done = counts.get(JobStatus.COMPLETED, 0)
            running = counts.get(JobStatus.RUNNING, 0)
            pending = counts.get(JobStatus.PENDING, 0)
            failed = counts.get(JobStatus.FAILED, 0)
            parts = [f"Очередь: {done}/{total}"]
            if running:
                parts.append(f"▶ {running}")
            if pending:
                parts.append(f"⏳ {pending}")
            if failed:
                parts.append(f"❌ {failed}")
            self.status_queue_label.setText(" | ".join(parts))
            # Also refresh ProgressWidget's overall bar
            self.progress_widget.set_overall(done, total)
        except Exception as exc:  # noqa: BLE001
            logger.debug("queue stats update failed: %s", exc)

    def _update_resource_usage(self) -> None:
        from src.infrastructure.file_utils import bytes_human

        suffix = ""
        try:
            doc_path = self.pdf_viewer.document_path
            if doc_path is not None and doc_path.exists():
                suffix = f" | PDF: {bytes_human(doc_path.stat().st_size)}"
        except Exception:  # noqa: BLE001
            suffix = ""
        try:
            import psutil  # lazy
            mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu = psutil.cpu_percent(interval=None)
            self.status_resource_label.setText(
                f"RAM: {mem_mb:.0f} МБ | CPU: {cpu:.0f}%{suffix}"
            )
        except Exception:  # noqa: BLE001
            self.status_resource_label.setText(f"Ready{suffix}")

    # Drag-and-drop visual feedback: while a PDF is dragged over the
    # window, wrap the central splitter in a dashed accent-coloured
    # border so the user sees a clear drop target.
    _DROP_STYLE = (
        "QMainWindow::separator { }"
        " QSplitter#dropHint { border: 3px dashed palette(highlight); }"
    )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and any(
            Path(u.toLocalFile()).suffix.lower() == ".pdf"
            or Path(u.toLocalFile()).is_dir()
            for u in event.mimeData().urls()
        ):
            event.acceptProposedAction()
            cw = self.centralWidget()
            if cw is not None:
                cw.setObjectName("dropHint")
                cw.setStyleSheet(self._DROP_STYLE)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802, ANN001
        """Clear the drop-target highlight when the drag exits."""
        cw = self.centralWidget()
        if cw is not None:
            cw.setStyleSheet("")
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        cw = self.centralWidget()
        if cw is not None:
            cw.setStyleSheet("")
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
        with contextlib.suppress(Exception):
            self._parallel_processor.shutdown(wait=False)
        if self._tray is not None:
            with contextlib.suppress(Exception):
                self._tray.shutdown()  # type: ignore[attr-defined]
        # Garbage-collect old temporary working directories from previous
        # runs (anything older than 24 h). Best-effort, never blocks shutdown.
        with contextlib.suppress(Exception):
            from src.infrastructure.file_utils import cleanup_temp_dir
            from src.shared.constants import TEMP_DIR

            removed = cleanup_temp_dir(TEMP_DIR, older_than_hours=24)
            if removed:
                logger.info("Cleaned %d stale temp file(s) from %s", removed, TEMP_DIR)
        super().closeEvent(event)
