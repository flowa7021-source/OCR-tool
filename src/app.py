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

    # Apply theme
    try:
        from src.ui.theme import apply_theme

        apply_theme(app)
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
