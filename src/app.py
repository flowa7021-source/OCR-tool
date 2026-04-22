"""Application bootstrap: constructs services and the main window."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from src.shared.constants import APP_NAME, APP_ORGANIZATION, APP_VERSION, ensure_user_dirs

if TYPE_CHECKING:
    from src.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def create_application(argv: list[str]) -> tuple[QApplication, MainWindow]:
    """Create the Qt application and main window, wiring all services.

    Args:
        argv: Command-line arguments (typically ``sys.argv``).

    Returns:
        A tuple of ``(QApplication, MainWindow)`` ready to show and exec.
    """
    ensure_user_dirs()

    from src.infrastructure.logger import setup_logging

    log_path = setup_logging()
    logger.info("Starting %s %s, logging to %s", APP_NAME, APP_VERSION, log_path)

    # Stale temp cleanup is deferred to a background thread AFTER the
    # main window is shown (see ``MainWindow._schedule_temp_cleanup``).
    # Doing it inline here made startup block on a filesystem walk of
    # ``%LOCALAPPDATA%/OCRStudio/temp/`` which could easily take 500 ms
    # on an HDD with many leftover job directories from prior crashes.

    # setHighDpiScaleFactorRoundingPolicy must be called BEFORE any
    # QApplication is instantiated. If a probe QApplication already
    # exists (src/main.py spawns one so the single-instance guard's
    # QLocalSocket has an event loop), this call is a no-op in practice
    # but also perfectly legal on Qt 6.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    # Reuse any existing QApplication rather than creating a second one.
    # Qt 6 / PySide6 raise ``RuntimeError: libshiboken: Please destroy
    # the QApplication singleton before creating a new QApplication
    # instance.`` if we try to construct two. ``main.py`` always creates
    # the first one as a probe for the single-instance guard, so in the
    # installed-app startup path we should always land here with one
    # already live.
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_ORGANIZATION)

    # Catch-all for unhandled exceptions in the main (UI) thread.
    # Without this hook a stray exception in a slot kills the event
    # loop silently: `--windowed` PyInstaller builds have no stderr,
    # so the user would just see the window disappear.
    _install_excepthook()

    # Apply theme — honour the persisted choice if available.
    try:
        from src.infrastructure.config_storage import SettingsStorage
        from src.ui.theme import apply_theme

        theme_kind = "dark"
        try:
            theme_kind = (SettingsStorage().load().theme or "dark").lower()
        except Exception:  # noqa: BLE001
            theme_kind = "dark"
        apply_theme(app, theme_kind)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to apply theme: %s", exc)

    # Set application-wide icon
    try:
        from src.ui.icons import app_icon

        app.setWindowIcon(app_icon())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not set app icon: %s", exc)

    # EasyOCR availability is verified by ``MainWindow._check_engine_sync``
    # right after the window is built — it's just an import probe (no
    # subprocess), so a few ms on the GUI thread is fine and a missing
    # torch/easyocr install must surface BEFORE the user clicks "Start
    # OCR".

    # Services
    from src.application.export_manager import ExportManager
    from src.application.parallel_processor import ParallelProcessor
    from src.application.profile_manager import ProfileManager
    from src.application.queue_manager import QueueManager
    from src.infrastructure.config_storage import ProfileStorage, SettingsStorage
    from src.ui.main_window import MainWindow

    profile_storage = ProfileStorage()
    profile_manager = ProfileManager(profile_storage)
    try:
        profile_manager.initialize_builtins()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize builtin profiles: %s", exc)

    settings_storage = SettingsStorage()
    settings = settings_storage.load()

    # Push the stored Anthropic API key (if any) into ``ANTHROPIC_API_KEY``
    # BEFORE any worker / orchestrator is wired. This is the single place
    # the credential crosses from settings.json into the process
    # environment; downstream code (parser's ``llm_fallback.improve_row``)
    # only sees the env var, never the storage object, so the key cannot
    # leak through log lines or profile exports. Failure is non-fatal —
    # an offline install without the key is still fully functional,
    # just without LLM rescue.
    try:
        from src.infrastructure.llm_credentials import apply_to_environment

        apply_to_environment(settings)
    except Exception as exc:  # noqa: BLE001
        logger.debug("LLM credential propagation failed: %s", exc)

    # Adaptive defaults on low-spec machines: 2 parallel workers at
    # 300 DPI peak at ~1.4 GB RAM, which pushes 4 GB laptops into swap.
    # Drop the worker count + cache budget in that case so the app
    # stays usable out of the box on entry-level hardware.
    try:
        from src.infrastructure.host_resources import (
            adjust_settings_for_host,
            detect,
        )

        host = detect()
        if adjust_settings_for_host(settings, host):
            settings_storage.save(settings)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Host-resource adaptation failed: %s", exc)

    queue_manager = QueueManager()
    parallel_processor = ParallelProcessor(max_workers=settings.parallel_workers)
    export_manager = ExportManager()

    window = MainWindow(
        profile_manager=profile_manager,
        queue_manager=queue_manager,
        parallel_processor=parallel_processor,
        export_manager=export_manager,
        settings_storage=settings_storage,
    )
    return app, window


def _install_excepthook() -> None:
    """Route unhandled exceptions to the log file + a UI error dialog.

    Qt swallows exceptions raised inside slot callbacks and only prints
    them to stderr — which doesn't exist in a ``--windowed`` PyInstaller
    build. Without this hook the user sees their click apparently do
    nothing, or the window vanishes on a more serious bug. With it:

      1. ``logger.critical`` writes the full traceback to the rotating
         log file, so we always have a record even when the user can't
         copy the message.
      2. A :class:`QMessageBox.critical` dialog tells the user what
         went wrong and where to find the log.

    KeyboardInterrupt passes through so Ctrl-C in a dev run still
    terminates the process cleanly.
    """
    import sys
    import traceback

    original_hook = sys.excepthook

    def _handler(exc_type, exc_value, exc_tb) -> None:  # noqa: ANN001
        if issubclass(exc_type, KeyboardInterrupt):
            original_hook(exc_type, exc_value, exc_tb)
            return
        logger.critical(
            "Unhandled exception in main thread",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        # Build a short tail of the traceback for the UI.
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
        tail = "".join(tb_lines[-3:])
        try:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(
                None,
                APP_NAME,
                (
                    "Произошла непредвиденная ошибка:\n\n"
                    f"{exc_type.__name__}: {exc_value}\n\n"
                    f"{tail}\n"
                    "Полный стек — в лог-файле (Помощь → Открыть лог)."
                ),
            )
        except Exception:  # noqa: BLE001 — dialog may fail during teardown
            original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _handler
