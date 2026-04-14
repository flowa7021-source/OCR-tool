"""Application-wide logging configuration.

Configures a root logger with a rotating file handler and an optional
stream handler. Designed to be idempotent so it can safely be called
from both the main entry point and individual test setups.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
from pathlib import Path

from src.shared.constants import (
    LOG_BACKUP_COUNT,
    LOG_FILE_NAME,
    LOG_MAX_BYTES,
    LOGS_DIR,
    ensure_user_dirs,
)

_LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"


def setup_logging(level: int = logging.INFO, log_to_console: bool = True) -> Path:
    """Configure the root logger with rotating file + optional stream handlers.

    The configuration is fully idempotent: existing handlers on the root
    logger are removed before new ones are attached. User-level directories
    are created via :func:`ensure_user_dirs` if they do not yet exist.

    Args:
        level: Logging level for the root logger (e.g. ``logging.DEBUG``).
        log_to_console: When True a :class:`logging.StreamHandler` is also
            attached to stderr.

    Returns:
        Absolute path to the active rotating log file.
    """
    ensure_user_dirs()
    log_file: Path = LOGS_DIR / LOG_FILE_NAME

    root_logger = logging.getLogger()
    # Idempotent setup: clear previous handlers so re-calling is safe.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        with contextlib.suppress(Exception):  # pragma: no cover - best-effort
            handler.close()

    root_logger.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    if log_to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root_logger.addHandler(stream_handler)

    root_logger.debug("Logging configured. File=%s level=%s", log_file, level)
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
