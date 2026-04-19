"""Main window tying together viewers, panels, queue, and menus."""

from __future__ import annotations

import contextlib
import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QByteArray,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
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


class _OverlaySignals(QObject):
    """Qt bridge for :class:`_OverlayExtractorRunnable`.

    ``ready`` uses ``object`` rather than ``dict`` because PySide6's
    meta-type copy-conversion rejects dicts whose values are nested
    Python tuples of floats with ``Cannot copy-convert (dict) to C++``.
    ``object`` is the standard pass-through for arbitrary Python data.
    """

    ready = Signal(object)  # {page_number: [(x, y, w, h, conf), ...]}
    failed = Signal(str)


class _PoolPrewarmRunnable(QRunnable):
    """Start the ProcessPoolExecutor off the GUI thread.

    ``multiprocessing.Manager()`` spawns a subprocess which on Windows
    takes ~1 s; first ``ProcessPoolExecutor.submit()`` spawn-starts a
    worker and imports the whole bundle, another 2-6 s. If both happen
    synchronously when the user clicks "Start OCR" the GUI freezes for
    3-8 seconds and looks dead. This runnable pays that cost right
    after window-show so the later click returns to the event loop
    instantly.
    """

    def __init__(self, parallel_processor) -> None:  # noqa: ANN001
        super().__init__()
        self._pp = parallel_processor

    @Slot()
    def run(self) -> None:  # noqa: D401
        try:
            self._pp.prewarm()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pool prewarm raised: %s", exc)


class _TesseractVerifySignals(QObject):
    result = Signal(bool, str)  # (ok, message)


class _JobBridge(QObject):
    """Marshal ParallelProcessor callbacks onto the GUI thread.

    ``ParallelProcessor`` fires ``on_progress`` from its internal drain
    thread (``ocr-progress-drain``) and ``on_complete`` / ``on_error``
    from the ``concurrent.futures`` result-dispatcher thread. Neither
    owns a Qt event loop, so ``QTimer.singleShot(0, callable)`` invoked
    from them silently drops the callable — the UI then never sees any
    progress or completion, which surfaces to the user as "I pressed
    Start OCR and nothing happens, the file just hangs".

    Emitting a ``Signal`` on a QObject living in the GUI thread and
    connecting it with ``Qt.QueuedConnection`` is the supported Qt
    cross-thread handoff: the signal's payload is queued into the
    main-thread event loop regardless of which thread called ``emit``.

    ``update_info`` rides the same bridge so the background
    update-checker thread can deliver its payload safely. Earlier
    revisions used ``QMetaObject.invokeMethod`` with ``Q_ARG(object,
    ...)`` for that, which raised ``RuntimeError: qArgDataFromPyType:
    Unable to find a QMetaType for "object"`` on PySide6 — Q_ARG only
    accepts Qt-registered types, not arbitrary Python objects.
    Signals, by contrast, carry Python objects natively.
    """

    progress = Signal(str, int, int, str)  # job_id, current, total, stage
    completed = Signal(str, object)  # job_id, JobResult
    failed = Signal(str, object)  # job_id, Exception
    update_info = Signal(object, bool)  # UpdateInfo | None, quiet


class _TempCleanupRunnable(QRunnable):
    """Walk ``TEMP_DIR`` in a worker thread and delete stale workdirs.

    Fired once shortly after window-show. Walking a deep ``temp/``
    tree on an HDD with 100+ leftover job dirs can easily take
    half a second — enough to be noticeable as a startup "sticky".
    """

    @Slot()
    def run(self) -> None:  # noqa: D401
        try:
            from src.infrastructure.file_utils import cleanup_temp_dir
            from src.shared.constants import TEMP_DIR

            removed = cleanup_temp_dir(TEMP_DIR, older_than_hours=24)
            if removed:
                logger.info(
                    "Background temp cleanup: removed %d stale file(s) from %s",
                    removed, TEMP_DIR,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Background temp cleanup failed: %s", exc)


class _RecoverySnapshotRunnable(QRunnable):
    """Persist a :class:`QueueItem` recovery snapshot off the GUI thread.

    Fire-and-forget from ``_apply_job_progress`` every 5 pages. A
    failed write is logged at DEBUG and otherwise ignored — the
    worst-case is a slightly stale snapshot on a hard crash, which is
    strictly better than a janked progress bar.
    """

    def __init__(self, recovery, item) -> None:  # noqa: ANN001
        super().__init__()
        self._recovery = recovery
        self._item = item

    @Slot()
    def run(self) -> None:  # noqa: D401
        try:
            self._recovery.snapshot(self._item)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Background recovery snapshot failed: %s", exc)


class _TesseractVerifyRunnable(QRunnable):
    """Run TesseractWrapper.verify() in a thread pool.

    Verification spawns ``tesseract --version`` as a subprocess. On a
    cold Windows boot that's a 200-800 ms disk read, and if the binary
    is missing the subprocess may take several seconds to time out.
    Running it on the GUI thread made the window unresponsive at
    startup for no good reason — push it into QThreadPool.

    ``parent`` parents the :class:`QObject` signals bridge to the
    caller so the queued connection is automatically severed when the
    parent is destroyed (e.g. under pytest teardown). Otherwise
    Qt dispatches the queued slot to a freed C++ object and aborts
    the interpreter.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__()
        self.signals = _TesseractVerifySignals(parent)

    @Slot()
    def run(self) -> None:  # noqa: D401
        try:
            from src.infrastructure.tesseract_wrapper import TesseractWrapper

            ok, msg = TesseractWrapper().verify()
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, str(exc)
        self.signals.result.emit(ok, msg)


class _OverlayExtractorRunnable(QRunnable):
    """Read word boxes from a searchable PDF in a worker thread.

    PyMuPDF's ``get_text("words")`` on a freshly-opened document is
    fully thread-safe because we open + close the doc inside this one
    runnable. The GUI thread is freed for input during the extraction,
    which used to freeze for several seconds on 500-page outputs.
    """

    def __init__(
        self,
        pdf_path: Path,
        per_page_confidence: list[float],
        parent: QObject | None = None,
    ) -> None:
        super().__init__()
        # See the analogous note on _TesseractVerifyRunnable — parent
        # the signal bridge so queued slots can't target a deleted
        # MainWindow.
        self.signals = _OverlaySignals(parent)
        self._pdf_path = pdf_path
        self._confidences = per_page_confidence

    @Slot()
    def run(self) -> None:  # noqa: D401
        try:
            import fitz
        except ImportError as exc:
            self.signals.failed.emit(f"PyMuPDF unavailable: {exc}")
            return
        mapping: dict[int, list[tuple[float, float, float, float, float]]] = {}
        try:
            doc = fitz.open(str(self._pdf_path))
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(f"open failed: {exc}")
            return
        try:
            for page_index in range(doc.page_count):
                conf = (
                    self._confidences[page_index]
                    if page_index < len(self._confidences)
                    else 80.0
                )
                try:
                    words = doc[page_index].get_text("words")
                except Exception:  # noqa: BLE001
                    continue
                boxes: list[tuple[float, float, float, float, float]] = []
                for word in words:
                    try:
                        x0, y0, x1, y1 = (
                            float(word[0]), float(word[1]),
                            float(word[2]), float(word[3]),
                        )
                    except (TypeError, ValueError, IndexError):
                        continue
                    boxes.append((x0, y0, x1 - x0, y1 - y0, conf))
                mapping[page_index + 1] = boxes
        finally:
            doc.close()
        self.signals.ready.emit(mapping)


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
        # Default size targets a common 1080p laptop (1366×768). Minimum
        # fits a 1024×768 panel after accounting for title-bar + taskbar
        # chrome; scrollbars take over inside each config panel below
        # that, so the app stays functional on low-end laptops / thin
        # clients without cropping any controls.
        self.resize(1440, 860)
        self.setMinimumSize(1000, 620)
        self.setWindowIcon(app_icon())
        # Tray notifier uses the same icon so completion toasts match.
        try:
            from src.ui.tray_notifier import TrayNotifier

            self._tray = TrayNotifier(app_icon(), self)
        except Exception:  # noqa: BLE001
            logger.debug("Tray notifier unavailable", exc_info=True)
            self._tray = None

        # Cross-thread bridge for ParallelProcessor callbacks. Must be
        # created before ``_build_*`` / ``_wire_signals`` so any early
        # job submission (e.g. crash-recovery prompt) marshals safely.
        self._job_bridge = _JobBridge(self)
        self._job_bridge.progress.connect(
            self._apply_job_progress, Qt.ConnectionType.QueuedConnection
        )
        self._job_bridge.completed.connect(
            self._apply_job_result, Qt.ConnectionType.QueuedConnection
        )
        self._job_bridge.failed.connect(
            self._apply_job_failure, Qt.ConnectionType.QueuedConnection
        )
        self._job_bridge.update_info.connect(
            self._show_update_info_slot, Qt.ConnectionType.QueuedConnection
        )

        self._build_widgets()
        self._build_docks()
        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()
        self._wire_signals()

        self._load_profiles_to_combobox()
        self._restore_window_state()

        # Deferred: verify Tesseract in a background thread AFTER the
        # window is visible so the user never stares at a blank screen
        # while `tesseract --version` runs a subprocess.
        # QTimer with ``self`` as parent is automatically deleted when
        # the window dies, so the scheduled callback can't fire on a
        # torn-down widget.
        #
        # Suppressed under pytest: test fixtures routinely create and
        # tear down MainWindows inside a single event-loop tick, and a
        # background-thread result landing after ``deleteLater()`` can
        # segfault via "double free or corruption". Production runs
        # never set PYTEST_CURRENT_TEST.
        import os as _os

        if not _os.environ.get("PYTEST_CURRENT_TEST"):
            self._verify_timer = QTimer(self)
            self._verify_timer.setSingleShot(True)
            self._verify_timer.timeout.connect(self._start_tesseract_verify_async)
            self._verify_timer.start(100)

            # Pre-warm the ProcessPoolExecutor + multiprocessing.Manager
            # in a background thread so the user's FIRST click of
            # "Start OCR" doesn't pay the 3-8 second Windows
            # spawn-start cost on the GUI thread. Fires 500 ms after
            # window-show so the visible UI stabilises first.
            self._prewarm_timer = QTimer(self)
            self._prewarm_timer.setSingleShot(True)
            self._prewarm_timer.timeout.connect(self._prewarm_parallel_pool)
            self._prewarm_timer.start(500)

            # Temp-dir GC also moved off the GUI thread. Previously ran
            # synchronously in ``src/app.py`` before QApplication was
            # instantiated — which still delayed the visible window by
            # 100-500 ms on an HDD with many leftover job dirs.
            self._temp_cleanup_timer = QTimer(self)
            self._temp_cleanup_timer.setSingleShot(True)
            self._temp_cleanup_timer.timeout.connect(self._schedule_temp_cleanup)
            self._temp_cleanup_timer.start(1500)

    # ------------------------------------------------------------ build UI
    def _build_widgets(self) -> None:
        self.pdf_viewer = PDFViewer(self)
        self.settings_panel = SettingsPanel(self)
        self.preprocessing_panel = PreprocessingPanel(self)
        self.postprocess_panel = PostprocessPanel(self)
        self.queue_panel = QueuePanel(self)
        self.progress_widget = ProgressWidget(self)
        self.results_panel = ResultsPanel(self)

        # Minimum widths reduced so the full UI fits on a 1280×720 screen.
        # Each config panel now lives inside a QScrollArea so it keeps
        # working even if the user squeezes the splitter well below its
        # content's sizeHint — before this change, resize could freeze
        # while Qt tried to reflow a mandatory-600-px-wide form.
        self.pdf_viewer.setMinimumWidth(360)

        def _scroll(widget: QWidget, min_width: int = 300) -> QWidget:
            from PySide6.QtWidgets import QScrollArea

            sa = QScrollArea(self)
            sa.setWidgetResizable(True)
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            sa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            sa.setWidget(widget)
            sa.setMinimumWidth(min_width)
            # Lose the inner frame — nesting two frames looks noisy.
            sa.setFrameShape(QScrollArea.Shape.NoFrame)
            return sa

        settings_scroll = _scroll(self.settings_panel, min_width=280)
        preprocess_scroll = _scroll(self.preprocessing_panel, min_width=280)
        postprocess_scroll = _scroll(self.postprocess_panel, min_width=280)

        # Right side: settings on top, then preprocessing / postprocessing tabs below
        right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(settings_scroll)

        self.right_tabs = QTabWidget(self)
        self.right_tabs.setMinimumWidth(320)
        self.right_tabs.setDocumentMode(True)
        self.right_tabs.addTab(preprocess_scroll, "Предобработка")
        self.right_tabs.addTab(postprocess_scroll, "Постобработка")
        right_splitter.addWidget(self.right_tabs)
        # Explicit initial pixel sizes — settings on top is short, tabs below tall.
        right_splitter.setSizes([260, 500])
        right_splitter.setStretchFactor(0, 0)
        right_splitter.setStretchFactor(1, 1)

        # Central horizontal splitter: viewer | right
        central_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        central_splitter.setChildrenCollapsible(False)
        central_splitter.addWidget(self.pdf_viewer)
        central_splitter.addWidget(right_splitter)
        # Viewer ~60 %, panels ~40 %. Initial pixel sizes fit 1280×720.
        central_splitter.setSizes([760, 500])
        central_splitter.setStretchFactor(0, 3)
        central_splitter.setStretchFactor(1, 2)
        self.setCentralWidget(central_splitter)

    def _build_docks(self) -> None:
        # Queue dock (bottom). Tighter min-heights so the whole app
        # fits comfortably on 1280×720 screens.
        queue_wrapper = QWidget(self)
        queue_wrapper.setMinimumHeight(140)
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
        self.queue_dock.setMinimumHeight(160)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.queue_dock)

        # Results dock (bottom, tabbed with queue)
        self.results_panel.setMinimumHeight(140)
        self.results_dock = QDockWidget("Результаты распознавания", self)
        self.results_dock.setObjectName("resultsDock")
        self.results_dock.setWidget(self.results_panel)
        self.results_dock.setMinimumHeight(160)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.results_dock)
        self.tabifyDockWidget(self.queue_dock, self.results_dock)
        self.queue_dock.raise_()

        # Reserve ~180 px for the dock by default so the viewer + panels
        # still dominate the visible area.
        self.resizeDocks([self.queue_dock], [180], Qt.Orientation.Vertical)

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
        sysinfo_act = QAction("Системная информация", self)
        sysinfo_act.triggered.connect(self._on_show_system_info)
        help_menu.addAction(sysinfo_act)
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
            # Prefer the "universal_accurate" preset as the first-run
            # pick — it's the opinionated max-accuracy bundle users get
            # "out of the box" without hand-tuning every knob. If it's
            # missing (tests with a stripped-down ProfileManager) we
            # fall back to whatever index 0 happens to be.
            preferred_idx = next(
                (i for i, p in enumerate(profiles) if p.name == "universal_accurate"),
                0,
            )
            self.profile_combo.setCurrentIndex(preferred_idx)
            self._apply_profile(profiles[preferred_idx])

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
                # Recovery snapshot off the GUI thread — batch drag-drop
                # of 50 files would otherwise do 50 synchronous disk
                # writes while the user is still moving the mouse.
                self._schedule_recovery_snapshot(item)
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
        """Progress callback — runs on a non-UI thread. Marshal into GUI thread.

        NOTE: ``QTimer.singleShot(0, lambda)`` was used here historically
        and silently failed because the caller (the ParallelProcessor
        drain thread) has no Qt event loop, so the timer never fired
        and the user saw no progress at all. The QueuedConnection-based
        signal bridge is the only safe cross-thread dispatch.
        """
        self._job_bridge.progress.emit(job_id, int(current), int(total), str(stage))

    @Slot(str, int, int, str)
    def _apply_job_progress(self, job_id: str, current: int, total: int, stage: str) -> None:
        try:
            self._queue_manager.update_progress(job_id, current, total)
        except Exception as exc:  # noqa: BLE001
            logger.debug("update_progress raised: %s", exc)
        # Update the ProgressWidget too
        item = self._queue_manager.get(job_id) if hasattr(self._queue_manager, "get") else None
        name = item.file_name if item is not None else ""
        self.progress_widget.set_current_file(f"{name} — {stage}", current, total)
        # Periodically refresh the recovery snapshot so a crash resumes
        # from ~ here. Moved off the GUI thread via QThreadPool:
        # ``snapshot`` is a disk write (JSON + replace) which on HDDs
        # can easily hit 50-200 ms and was visibly janking the progress
        # bar.
        if item is not None and current % 5 == 0:
            self._schedule_recovery_snapshot(item)

    def _on_job_complete(self, job_id: str, result: object) -> None:
        # Called from the concurrent.futures result-dispatcher thread,
        # which has no Qt event loop. Use the signal bridge — see
        # ``_on_job_progress`` for the full explanation.
        self._job_bridge.completed.emit(job_id, result)

    def _on_job_failed(self, job_id: str, exc: BaseException) -> None:
        # Same cross-thread situation as _on_job_complete.
        self._job_bridge.failed.emit(job_id, exc)

    @Slot(str, object)
    def _apply_job_failure(self, job_id: str, exc: object) -> None:
        """GUI-thread handler for worker failures."""
        try:
            self._queue_manager.update_status(
                job_id, JobStatus.FAILED, str(exc)
            )
        except Exception:  # noqa: BLE001
            logger.debug("Failed to mark job %s FAILED", job_id, exc_info=True)

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

    @Slot(str, object)
    def _apply_job_result(self, job_id: str, result: object) -> None:
        self._last_result = result
        self.results_panel.set_result(result)  # type: ignore[arg-type]
        # Feed word boxes to the viewer so the overlay works on the result
        # PDF. Crucially this must NOT block the main thread: on a
        # 500-page output the old synchronous loop iterated ~150 k word
        # boxes and froze the GUI for several seconds. Offload to
        # QThreadPool; the result is marshalled back via a signal.
        try:
            self._schedule_overlay_population(result)
        except Exception:  # noqa: BLE001
            logger.debug("Overlay scheduling failed", exc_info=True)
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

    def _schedule_recovery_snapshot(self, item) -> None:  # noqa: ANN001
        """Submit a background recovery snapshot. Fire-and-forget."""
        try:
            runnable = _RecoverySnapshotRunnable(self._recovery, item)
            QThreadPool.globalInstance().start(runnable)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not schedule recovery snapshot: %s", exc)

    def _schedule_temp_cleanup(self) -> None:
        """Submit the startup temp-dir GC to the thread pool."""
        try:
            QThreadPool.globalInstance().start(_TempCleanupRunnable())
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not schedule temp cleanup: %s", exc)

    def _prewarm_parallel_pool(self) -> None:
        """Submit the ProcessPoolExecutor prewarm to the thread pool.

        Cheap wrapper — the real work lives in :class:`_PoolPrewarmRunnable`
        which runs on QThreadPool so the GUI thread is never blocked by
        the ~1 s manager spawn and ~2-6 s first-worker import.
        """
        try:
            runnable = _PoolPrewarmRunnable(self._parallel_processor)
            QThreadPool.globalInstance().start(runnable)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not schedule pool prewarm: %s", exc)

    def _start_tesseract_verify_async(self) -> None:
        """Kick Tesseract verification off the GUI thread.

        Called shortly after window-show so a slow or missing Tesseract
        subprocess doesn't freeze startup. The status-bar message tells
        the user what we're doing; the result handler (either success
        or a non-blocking warning) runs on the main thread.
        """
        with contextlib.suppress(Exception):
            self.statusBar().showMessage("Проверка Tesseract…", 0)
        runnable = _TesseractVerifyRunnable(parent=self)
        runnable.signals.result.connect(
            self._on_tesseract_verify_done,
            Qt.ConnectionType.QueuedConnection,
        )
        QThreadPool.globalInstance().start(runnable)

    @Slot(bool, str)
    def _on_tesseract_verify_done(self, ok: bool, message: str) -> None:
        """Handle Tesseract verify result back on the GUI thread."""
        if ok:
            logger.info("Tesseract verified: %s", message)
            self.statusBar().showMessage(
                f"Tesseract готов: {message}", 5000
            )
            return
        logger.warning("Tesseract verification reported: %s", message)
        self.statusBar().showMessage(
            "Tesseract не прошёл проверку — OCR недоступен", 0
        )
        # Non-blocking info; the user can click past without affecting
        # the app's responsiveness.
        QMessageBox.warning(
            self,
            APP_NAME,
            (
                f"Tesseract не прошёл проверку:\n\n{message}\n\n"
                "OCR будет недоступен до установки Tesseract 5.x."
            ),
        )

    def _schedule_overlay_population(self, result: object) -> None:
        """Submit overlay extraction to the thread pool.

        Wraps :class:`_OverlayExtractorRunnable` so the GUI thread stays
        responsive while PyMuPDF walks every page of the output PDF.
        """
        if not isinstance(result, JobResult):
            return
        pdf_path = Path(result.output_path)
        if not pdf_path.exists():
            return
        confs = [float(p.mean_confidence or 80.0) for p in result.pages]
        runnable = _OverlayExtractorRunnable(pdf_path, confs, parent=self)
        runnable.signals.ready.connect(
            self._apply_overlay_bulk, Qt.ConnectionType.QueuedConnection
        )
        runnable.signals.failed.connect(
            lambda msg: logger.debug("Overlay extraction: %s", msg),
            Qt.ConnectionType.QueuedConnection,
        )
        QThreadPool.globalInstance().start(runnable)

    @Slot(object)
    def _apply_overlay_bulk(
        self,
        mapping: dict[int, list[tuple[float, float, float, float, float]]],
    ) -> None:
        """Feed a page→boxes mapping into the viewer in a single call."""
        try:
            self.pdf_viewer.clear_word_boxes()
            self.pdf_viewer.set_word_boxes_bulk(mapping)
        except Exception:  # noqa: BLE001
            logger.debug("Bulk overlay apply failed", exc_info=True)

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

        # Refresh the profile combo with restored entries. This is the
        # user-visible confirmation that the new profiles actually show
        # up in the dropdown, so a failure here is worth surfacing rather
        # than hiding in the debug log.
        try:
            self._load_profiles_to_combobox()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Profile combo reload after import failed: %s", exc)
            QMessageBox.warning(
                self,
                APP_NAME,
                (
                    "Импорт прошёл, но обновить список профилей не удалось. "
                    "Перезапустите приложение, чтобы увидеть новые профили.\n\n"
                    f"{exc}"
                ),
            )

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

    def _on_show_system_info(self) -> None:
        """Show a summary of detected hardware + current app settings.

        Lets a user (or a support engineer) see what the app thinks
        of the host — useful when diagnosing "is it using all my
        cores?" / "why only 1 worker?" type questions. Nothing here
        leaves the machine; this is a local-only inspection dialog.
        """
        import platform

        from src.infrastructure.host_resources import detect

        host = detect()
        try:
            settings = self._settings_storage.load()
        except Exception:  # noqa: BLE001
            settings = None
        lines: list[str] = []
        lines.append(f"<b>OS:</b> {platform.platform()}")
        lines.append(f"<b>Python:</b> {platform.python_version()}")
        if host.detected:
            lines.append(
                f"<b>CPU:</b> {host.cpu_count} logical threads"
            )
            lines.append(
                f"<b>RAM:</b> {host.total_ram_gb:.1f} GB "
                f"(доступно: {host.available_ram_gb:.1f} GB)"
            )
            lines.append(
                f"<b>Свободно на диске:</b> {host.free_disk_gb:.1f} GB"
            )
        else:
            lines.append("<i>Детектор psutil недоступен — данные о железе не собраны.</i>")
        try:
            import torch  # type: ignore[import-not-found]

            if torch.cuda.is_available():
                lines.append(
                    f"<b>GPU:</b> CUDA {torch.version.cuda} — "
                    f"{torch.cuda.get_device_name(0)} "
                    f"({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB VRAM)"
                )
            else:
                lines.append("<b>GPU:</b> CUDA недоступна — GOT-OCR работает на CPU")
        except ImportError:
            lines.append("<b>GPU:</b> torch не установлен (HTR недоступен)")

        lines.append("")  # blank
        if settings is not None:
            lines.append(f"<b>Воркеров:</b> {settings.parallel_workers}")
            lines.append(
                "<b>Кэш OCR:</b> "
                + (
                    "отключён"
                    if settings.ocr_cache_max_mb == 0
                    else f"{settings.ocr_cache_max_mb} МБ"
                )
            )
            lines.append(
                f"<b>Автосохранение:</b> каждые {settings.autosave_interval_pages} стр."
                if settings.autosave_interval_pages
                else "<b>Автосохранение:</b> отключено"
            )

        if host.low_memory:
            lines.append("")
            lines.append(
                "<i>Режим экономии памяти активен автоматически "
                "(RAM < 6 ГБ).</i>"
            )

        QMessageBox.information(
            self,
            APP_NAME,
            "<br>".join(lines),
        )

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
        # ``check_async`` spawns a plain ``threading.Thread`` to hit
        # GitHub; its callback lands on that non-Qt thread. Route the
        # result through the ``_job_bridge.update_info`` signal — a
        # QueuedConnection signal carries Python objects natively and
        # avoids the ``qArgDataFromPyType: Unable to find a QMetaType
        # for 'object'`` error that ``QMetaObject.invokeMethod`` with
        # ``Q_ARG(object, ...)`` raises on PySide6 for arbitrary Python
        # payloads.
        from src.application.update_checker import check_async

        def _report(info) -> None:  # noqa: ANN001
            self._job_bridge.update_info.emit(info, quiet)

        check_async(_report)

    @Slot(object, bool)
    def _show_update_info_slot(self, info: object, quiet: bool) -> None:
        """Queued wrapper so ``check_async`` callbacks reach the GUI thread."""
        self._show_update_info(info, quiet=quiet)

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
