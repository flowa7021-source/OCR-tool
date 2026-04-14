"""Application bootstrap: constructs services and the main window."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

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

    # Garbage-collect stale temp workdirs left by previous sessions.
    # closeEvent runs the same cleanup on graceful exit, but a hard
    # kill / crash / BSOD never gets there — over weeks the temp/
    # directory can accumulate many GB of pre-processed PNGs. Doing
    # it here too bounds the on-disk cost to "one app run" even under
    # the worst shutdown path.
    try:
        from src.infrastructure.file_utils import cleanup_temp_dir
        from src.shared.constants import TEMP_DIR

        removed = cleanup_temp_dir(TEMP_DIR, older_than_hours=24)
        if removed:
            logger.info(
                "Startup: cleaned %d stale temp files from %s",
                removed,
                TEMP_DIR,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Startup temp cleanup failed: %s", exc)

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
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

    # Soft-verify Tesseract (non-fatal)
    try:
        from src.infrastructure.tesseract_wrapper import TesseractWrapper

        wrapper = TesseractWrapper()
        ok, message = wrapper.verify()
        if not ok:
            logger.warning("Tesseract verification reported: %s", message)
            QMessageBox.warning(
                None,
                APP_NAME,
                (
                    f"Tesseract не прошёл проверку:\n\n{message}\n\n"
                    "Приложение запустится, но OCR будет недоступен до установки Tesseract 5.x."
                ),
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tesseract verification raised: %s", exc)
        QMessageBox.warning(
            None,
            APP_NAME,
            f"Ошибка инициализации Tesseract: {exc}",
        )

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
