"""Windows-only: hide console windows from child subprocesses.

On Windows, the PyInstaller ``--windowed`` build launches OCR Studio
from a GUI-subsystem ``OCRStudio.exe`` with no console attached. When
any library inside the app calls :class:`subprocess.Popen` to spawn a
console-subsystem child, Windows gives that child a fresh console
window that flashes for the duration of the subprocess.

The fix is a targeted monkey-patch of :class:`subprocess.Popen` that
defaults ``creationflags`` to ``CREATE_NO_WINDOW`` (``0x08000000``)
whenever the caller did not pass explicit creation flags or a
``startupinfo`` object.

The monkey-patch is scoped to the current process only. Non-Windows
platforms are a no-op.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Windows constant — duplicated from the Win32 API so we don't need to
# import ``ctypes``/``_winapi`` on non-Windows platforms. Value is
# stable since Windows 2000 and documented at
# https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags
_CREATE_NO_WINDOW: int = 0x08000000

# Guard against double-installation. Installing the patch twice would
# wrap the wrapper and leak stack frames on every Popen() call.
_INSTALLED: bool = False


def install_windows_console_hide() -> bool:
    """Install the ``subprocess.Popen`` monkey-patch on Windows.

    Returns:
        ``True`` if the patch was installed by this call, ``False`` if
        it was already installed or the platform is non-Windows.

    Idempotent: subsequent calls short-circuit after the first
    install. Safe to invoke from multiple entry points (``main.py``,
    worker init) without additional bookkeeping.
    """
    global _INSTALLED
    if _INSTALLED:
        return False
    if sys.platform != "win32":
        # On Linux / macOS there is no console-allocation step — Popen
        # inherits the parent's stdio and ``creationflags`` is ignored.
        # Mark installed so a future callsite doesn't keep retrying.
        _INSTALLED = True
        return False

    original_init = subprocess.Popen.__init__

    def _patched_init(self: subprocess.Popen, *args: Any, **kwargs: Any) -> None:
        # Only inject our flag when the caller did NOT already ask for
        # something specific. Two signals indicate "hands off":
        #   1. explicit ``creationflags`` — any non-zero value means
        #      the caller is configuring Windows process creation
        #      deliberately (e.g. ``CREATE_NEW_PROCESS_GROUP`` for
        #      signal isolation in ``ProcessPoolExecutor``).
        #   2. ``startupinfo`` — caller is controlling window-show
        #      behaviour themselves (e.g. ``STARTF_USESHOWWINDOW``).
        caller_flags = kwargs.get("creationflags", 0) or 0
        if caller_flags == 0 and kwargs.get("startupinfo") is None:
            kwargs["creationflags"] = caller_flags | _CREATE_NO_WINDOW
        original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _patched_init  # type: ignore[method-assign]
    _INSTALLED = True
    logger.debug(
        "subprocess_hygiene: CREATE_NO_WINDOW default installed on Popen"
    )
    return True
