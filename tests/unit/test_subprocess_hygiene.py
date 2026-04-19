"""Unit tests for :mod:`src.infrastructure.subprocess_hygiene`.

Verifies that the Windows-only Popen monkey-patch:

1. Is idempotent (double-install is a no-op).
2. Defaults ``creationflags`` to ``CREATE_NO_WINDOW`` when the caller
   passed neither ``creationflags`` nor ``startupinfo``.
3. Does NOT override an explicit ``creationflags`` kwarg (e.g. a
   caller asking for ``CREATE_NEW_PROCESS_GROUP`` for signal isolation
   in ``ProcessPoolExecutor``).
4. Does NOT override an explicit ``startupinfo`` (the caller is
   already managing window visibility themselves).
5. No-op on POSIX.

Tests use ``patch`` to replace the real ``subprocess.Popen.__init__``
with a spy so we can inspect the kwargs the patched wrapper forwards
without actually spawning a process.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def _reset_install_flag():
    """Reset the module-level install guard between tests."""
    import src.infrastructure.subprocess_hygiene as sh

    original = sh._INSTALLED
    sh._INSTALLED = False
    # Also restore the original Popen.__init__ so a previous install
    # doesn't leak patches across tests.
    original_init = subprocess.Popen.__init__
    yield
    sh._INSTALLED = original
    subprocess.Popen.__init__ = original_init


class TestWindowsBehaviour:
    """Only meaningful when we pretend to be on Windows."""

    def test_install_returns_true_once_on_windows(
        self, _reset_install_flag, monkeypatch
    ) -> None:
        """First install returns True, second returns False."""
        monkeypatch.setattr(sys, "platform", "win32")
        from src.infrastructure.subprocess_hygiene import (
            install_windows_console_hide,
        )

        assert install_windows_console_hide() is True
        # Idempotent — second call is a no-op.
        assert install_windows_console_hide() is False

    def test_default_creationflags_get_create_no_window(
        self, _reset_install_flag, monkeypatch
    ) -> None:
        """When caller passes nothing, Popen gets CREATE_NO_WINDOW."""
        monkeypatch.setattr(sys, "platform", "win32")

        # Replace the original __init__ with a spy BEFORE installing the
        # monkey-patch. The patch wraps whatever __init__ was current
        # at install time — our spy sees the wrapped-through kwargs.
        spy = MagicMock(return_value=None)
        subprocess.Popen.__init__ = spy  # type: ignore[method-assign]

        from src.infrastructure.subprocess_hygiene import (
            _CREATE_NO_WINDOW,
            install_windows_console_hide,
        )

        install_windows_console_hide()

        # Trigger Popen (__new__ returns a bare object, __init__ is our
        # patched wrapper wrapping the spy).
        dummy = subprocess.Popen.__new__(subprocess.Popen)
        subprocess.Popen.__init__(dummy, ["cmd"])

        assert spy.called
        _, kwargs = spy.call_args
        assert kwargs.get("creationflags") == _CREATE_NO_WINDOW

    def test_explicit_creationflags_are_preserved(
        self, _reset_install_flag, monkeypatch
    ) -> None:
        """If the caller set creationflags, the patch must not touch them."""
        monkeypatch.setattr(sys, "platform", "win32")

        spy = MagicMock(return_value=None)
        subprocess.Popen.__init__ = spy  # type: ignore[method-assign]

        from src.infrastructure.subprocess_hygiene import (
            install_windows_console_hide,
        )

        install_windows_console_hide()

        explicit = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        dummy = subprocess.Popen.__new__(subprocess.Popen)
        subprocess.Popen.__init__(dummy, ["cmd"], creationflags=explicit)

        _, kwargs = spy.call_args
        # Caller's exact value survives — we do NOT OR in CREATE_NO_WINDOW
        # because a non-zero creationflags is the caller's signal that
        # they're managing Windows process creation deliberately.
        assert kwargs.get("creationflags") == explicit

    def test_explicit_startupinfo_is_preserved(
        self, _reset_install_flag, monkeypatch
    ) -> None:
        """A caller-provided startupinfo is the opt-out signal."""
        monkeypatch.setattr(sys, "platform", "win32")

        spy = MagicMock(return_value=None)
        subprocess.Popen.__init__ = spy  # type: ignore[method-assign]

        from src.infrastructure.subprocess_hygiene import (
            install_windows_console_hide,
        )

        install_windows_console_hide()

        startup = object()  # opaque — the patch must not inspect it
        dummy = subprocess.Popen.__new__(subprocess.Popen)
        subprocess.Popen.__init__(dummy, ["cmd"], startupinfo=startup)

        _, kwargs = spy.call_args
        # creationflags must not have been auto-set when startupinfo is
        # present — the caller has already chosen how to handle the
        # console window.
        assert "creationflags" not in kwargs or kwargs.get("creationflags") == 0
        assert kwargs.get("startupinfo") is startup


class TestPosixBehaviour:
    """On Linux / macOS the patch is a no-op (marked installed + False)."""

    def test_install_returns_false_on_posix(
        self, _reset_install_flag, monkeypatch
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        from src.infrastructure.subprocess_hygiene import (
            install_windows_console_hide,
        )

        # POSIX: returns False (nothing installed), but the flag still
        # flips so subsequent calls short-circuit.
        assert install_windows_console_hide() is False

    def test_popen_init_is_not_patched_on_posix(
        self, _reset_install_flag, monkeypatch
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        from src.infrastructure.subprocess_hygiene import (
            install_windows_console_hide,
        )

        snapshot = subprocess.Popen.__init__
        install_windows_console_hide()
        # No wrapping happened — the class method is bit-for-bit the
        # same function object it was before the install call.
        assert subprocess.Popen.__init__ is snapshot

    def test_patch_does_not_inject_flags_on_posix(
        self, _reset_install_flag, monkeypatch
    ) -> None:
        """Even if a caller somehow invokes the installer, no side-effects.

        Belt-and-braces: if a plugin were to call the installer on
        Linux, ``subprocess.run`` must still work exactly as stdlib
        documents it — ``creationflags`` is ignored outside of Windows
        but passing a stray value can surprise other tooling.
        """
        monkeypatch.setattr(sys, "platform", "linux")
        from src.infrastructure.subprocess_hygiene import (
            install_windows_console_hide,
        )

        install_windows_console_hide()

        # Use a dummy ``Popen.__init__`` replacement just for the
        # duration of this test — we don't want to actually spawn
        # anything, and stdlib Popen touches fd inheritance on Linux
        # which confuses CI sandboxing.
        with patch.object(subprocess.Popen, "__init__", return_value=None) as spy:
            dummy = subprocess.Popen.__new__(subprocess.Popen)
            subprocess.Popen.__init__(dummy, ["/bin/true"])

        _, kwargs = spy.call_args
        assert "creationflags" not in kwargs
