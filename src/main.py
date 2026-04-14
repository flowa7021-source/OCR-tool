"""Entry point for OCR Studio."""

from __future__ import annotations

import sys


def main() -> int:
    """Start the Qt event loop and return its exit code."""
    # Every launch FIRST consults the single-instance guard: if another
    # OCR Studio is already running, the current process forwards its
    # CLI args (a double-click on a PDF file → opens in the existing
    # window) and exits 0. Only the primary instance goes on to build
    # QApplication + MainWindow.
    from pathlib import Path

    from PySide6.QtWidgets import QApplication as _QApp

    from src.app import create_application

    _probe_app = _QApp.instance() or _QApp(sys.argv)  # need an event loop for QLocalSocket
    from src.ui.single_instance import SingleInstanceGuard

    guard = SingleInstanceGuard()
    if not guard.is_primary():
        # Forward PDF paths from argv; ignore other flags.
        pdf_args = [a for a in sys.argv[1:] if a.lower().endswith(".pdf")]
        if guard.forward_and_exit(pdf_args):
            return 0
        # If forwarding failed we fall through and open a second window
        # rather than silently doing nothing.

    # If we got here via the probe QApplication, hand it over to the
    # real bootstrap so styling/organisation stay consistent.
    app, window = create_application(sys.argv)

    guard.start_listening()
    guard.args_received.connect(
        lambda paths: window._open_paths_from_secondary([Path(p) for p in paths])
    )
    window.show()
    # Offer to resume any jobs abandoned by a previous crash / hard shutdown.
    # Defer so the main window is visible first.
    from PySide6.QtCore import QTimer

    QTimer.singleShot(50, window.prompt_recovery)
    # Quiet auto-update check five seconds after the window is shown,
    # so it never blocks startup UI and doesn't bug users with false
    # "can't reach GitHub" errors during normal offline use.
    QTimer.singleShot(5000, lambda: window._on_check_updates(quiet=True))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
