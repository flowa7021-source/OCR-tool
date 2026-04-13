"""Entry point for OCR Studio."""

from __future__ import annotations

import sys


def main() -> int:
    """Start the Qt event loop and return its exit code."""
    from src.app import create_application

    app, window = create_application(sys.argv)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
