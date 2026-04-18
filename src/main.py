"""Entry point for OCR Studio."""

from __future__ import annotations

import multiprocessing
import sys

# CLI-signalling argv flags. If ANY of these appears in sys.argv we
# route straight to ``src.cli.main()`` instead of starting the Qt
# event loop. A bare ``OCRStudio.exe file.pdf`` still opens the GUI
# (the GUI accepts a PDF path as an open-on-launch argument); the
# CLI takes over only when the user asks for it explicitly.
#
# Keeping this list short and flag-based (rather than "any non-PDF
# argv") avoids accidentally breaking file-association double-click
# where Windows passes paths that might contain dashes.
_CLI_FLAGS: frozenset[str] = frozenset({
    "--cli",
    "-o", "--output",
    "-p", "--profile",
    "--list-profiles",
    "--workers",
    "--txt",
    "--docx",
    "--version",
    "-h", "--help",
})


def _should_route_to_cli(argv: list[str]) -> bool:
    """Return True when argv contains any CLI-only flag."""
    return any(a in _CLI_FLAGS for a in argv[1:])


def main() -> int:
    """Start the Qt event loop and return its exit code.

    When invoked with CLI-only flags (``--cli``, ``--profile``,
    ``-o``, etc.) we skip the Qt bootstrap entirely and dispatch to
    ``src.cli.main`` — this lets the single ``OCRStudio.exe`` binary
    serve both GUI double-click launches AND headless CI / scripting
    invocations without needing two separate PyInstaller bundles.
    """
    # CLI path first: no QApplication, no single-instance guard, no
    # window — just argparse + pipeline. This is what the install
    # smoke-test + any user-written batch script hits.
    if _should_route_to_cli(sys.argv):
        # Strip a bare ``--cli`` so src.cli's argparse doesn't choke
        # on it. Other flags pass through verbatim.
        cli_argv = [a for a in sys.argv[1:] if a != "--cli"]
        from src import cli as _cli

        return _cli.main(cli_argv)

    # Suppress the brief console windows Tesseract/Ghostscript would
    # otherwise flash on Windows ``--windowed`` builds. No-op on
    # POSIX. Must run before anything else imports ``subprocess``-
    # using code so every Popen site inherits the patched default.
    from src.infrastructure.subprocess_hygiene import (
        install_windows_console_hide,
    )

    install_windows_console_hide()

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
    # CRITICAL: must be the FIRST thing in __main__ on Windows.
    #
    # Without this call, PyInstaller-frozen executables that use
    # ``multiprocessing`` (our ProcessPoolExecutor) recursively spawn
    # copies of the app every time a worker is created — because
    # Windows ``spawn`` starts each worker by re-executing the main
    # script, which hits this block, which starts another pool, which
    # spawns more workers, etc. Each of those "workers" actually
    # reaches ``create_application`` briefly, then the OS kills the
    # runaway process, which surfaces as:
    #
    #   A child process terminated abruptly, the process pool is not
    #   usable anymore
    #
    # ``freeze_support()`` detects the "I am a spawned worker" case
    # and short-circuits: the worker runs its payload and exits.
    # No-op on non-frozen Python.
    multiprocessing.freeze_support()
    sys.exit(main())
