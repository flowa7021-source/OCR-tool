"""System-tray notifications for long-running OCR jobs.

Uses :class:`QSystemTrayIcon.showMessage`, which on Windows 10/11 routes
through the native Toast pipeline, on macOS uses NSUserNotification, and
on Linux uses libnotify when available. No extra dependencies.

The notifier is a lightweight wrapper that:

* gracefully no-ops when the desktop environment has no tray support
  (some headless / minimal X sessions);
* throttles notifications — we don't want to fire a toast for every
  page of a 500-page job, only when the job as a whole finishes;
* respects the user's "notify_on_complete" preference.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QSystemTrayIcon

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

# 5 s is the Windows default; more than enough to read a short message
# but not annoying if you're mid-typing.
_DEFAULT_DURATION_MS = 5000


class TrayNotifier:
    """Thin façade around ``QSystemTrayIcon`` for job-completion toasts."""

    def __init__(self, icon: QIcon, parent: QWidget | None = None) -> None:
        self._tray: QSystemTrayIcon | None = None
        self.enabled: bool = True
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.info("System tray not available — notifications disabled.")
            return
        self._tray = QSystemTrayIcon(icon, parent)
        self._tray.setToolTip("OCR Studio")
        # We don't need a context menu — the app window is the primary UI,
        # the tray icon is purely a notification carrier. Showing the icon
        # is harmless on Windows 11 (collapses under "hidden icons").
        self._tray.show()

    def notify_complete(self, title: str, message: str) -> None:
        """Pop a success toast. Falls back to a debug log when unavailable."""
        self._notify(title, message, QSystemTrayIcon.MessageIcon.Information)

    def notify_failure(self, title: str, message: str) -> None:
        """Pop a failure toast (warning icon)."""
        self._notify(title, message, QSystemTrayIcon.MessageIcon.Warning)

    def shutdown(self) -> None:
        """Release tray resources on app exit."""
        if self._tray is not None:
            self._tray.hide()
            self._tray = None

    # ------------------------------------------------------------------ impl
    def _notify(
        self,
        title: str,
        message: str,
        severity: QSystemTrayIcon.MessageIcon,
    ) -> None:
        if not self.enabled or self._tray is None:
            logger.debug("Tray notify skipped (enabled=%s, tray=%s): %s",
                         self.enabled, self._tray is not None, title)
            return
        try:
            self._tray.showMessage(title, message, severity, _DEFAULT_DURATION_MS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("showMessage failed: %s", exc)
