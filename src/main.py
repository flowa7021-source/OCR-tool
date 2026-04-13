"""Entry point for OCR Studio."""

from __future__ import annotations

import sys


def main() -> int:
    """Start the Qt event loop and return its exit code."""
    from src.app import create_application

    app, window = create_application(sys.argv)
    window.show()
    # Offer to resume any jobs abandoned by a previous crash / hard shutdown.
    # Defer so the main window is visible first.
    from PySide6.QtCore import QTimer

    QTimer.singleShot(50, window.prompt_recovery)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
