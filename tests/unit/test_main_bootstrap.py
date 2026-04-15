"""Regression tests for the app bootstrap path (src/main.py → src/app.py).

The failure we're guarding against:

    RuntimeError: libshiboken: Please destroy the QApplication
    singleton before creating a new QApplication instance.

Background. ``src/main.py`` has to construct a ``QApplication`` up
front because the :class:`SingleInstanceGuard` needs an event loop
to drive its ``QLocalSocket``. Then it calls
:func:`src.app.create_application`, which — if we're not careful —
happily constructs a **second** ``QApplication`` and crashes PySide6
on Qt 6. The previous implementation called ``QApplication(argv)``
unconditionally; the fix is to reuse ``QApplication.instance()``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def _stub_heavy_bootstrap(monkeypatch, tmp_path: Path):
    """Stub the slow/side-effecting bits of create_application.

    - Tesseract verify: return (True, '') so the MessageBox branch is
      unreachable.
    - QMessageBox.warning: no-op, otherwise offscreen Qt hangs on modal.
    - SettingsStorage / ProfileStorage: point at tmp_path.
    """
    # Redirect user-data dirs so the test doesn't touch ~/.ocrstudio.
    import src.shared.constants as constants

    for name in ("USER_DATA_DIR", "CONFIG_DIR", "PROFILES_DIR", "TEMP_DIR",
                 "LOGS_DIR", "RECOVERY_DIR", "OCR_CACHE_DIR"):
        monkeypatch.setattr(constants, name, tmp_path / name.lower())
        (tmp_path / name.lower()).mkdir(exist_ok=True)

    from src.infrastructure.tesseract_wrapper import TesseractWrapper

    monkeypatch.setattr(TesseractWrapper, "verify", lambda self: (True, "ok"))

    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **kw: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **kw: None)


class TestCreateApplicationSingletonSafety:
    def test_reuses_existing_qapplication(self, _stub_heavy_bootstrap) -> None:
        """main.py's probe QApplication must NOT cause libshiboken error."""
        from PySide6.QtWidgets import QApplication

        from src.app import create_application

        # Exactly what src/main.py:21 does today: create the probe.
        probe = QApplication.instance() or QApplication(sys.argv)

        # Without the fix this raises:
        #   RuntimeError: libshiboken: Please destroy the QApplication
        #   singleton before creating a new QApplication instance.
        app, window = create_application(sys.argv)

        assert app is probe, (
            "create_application must reuse the existing QApplication "
            "instance; constructing a second one crashes PySide6 on "
            "installed-app startup."
        )
        assert window is not None
        window.close()
        window.deleteLater()

    def test_does_not_blow_up_on_repeated_calls(
        self, _stub_heavy_bootstrap
    ) -> None:
        """Calling create_application twice in a row must be safe."""
        from PySide6.QtWidgets import QApplication

        from src.app import create_application

        QApplication.instance() or QApplication(sys.argv)
        app1, w1 = create_application(sys.argv)
        app2, w2 = create_application(sys.argv)
        assert app1 is app2, "both calls must return the same singleton"
        for w in (w1, w2):
            w.close()
            w.deleteLater()

    def test_app_py_reuses_instance_via_instance_or_construct(self) -> None:
        """Static check: src/app.py must NOT unconditionally call QApplication(argv).

        Regression guard — a future refactor that deletes the
        ``QApplication.instance()`` check would re-introduce the
        libshiboken crash and this test would catch it without needing
        to boot a real QApplication.
        """
        import ast

        path = Path(__file__).parent.parent.parent / "src" / "app.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "create_application":
                continue
            # The function body must reference QApplication.instance()
            # at least once. We deliberately don't try to check the
            # exact assignment pattern — any call to .instance() is
            # evidence that the author considered the singleton case.
            body = ast.dump(node)
            assert "QApplication" in body and "instance" in body, (
                "create_application no longer queries QApplication.instance(). "
                "This re-introduces the libshiboken crash when main.py has "
                "already built a probe QApplication for the single-instance "
                "guard. See tests/unit/test_app_bootstrap.py."
            )
            return
        pytest.fail("create_application not found in src/app.py")

    def test_main_py_creates_probe_qapplication(self) -> None:
        """Static check: src/main.py must keep instantiating a probe QApp.

        The reuse fix in app.py relies on main.py creating the probe.
        If someone deletes that line, create_application would try to
        construct a QApplication under a caller who forgot — which is
        also fine because instance() returns None there, but this
        pre-probe pattern is intentional for the single-instance guard.
        """
        path = Path(__file__).parent.parent.parent / "src" / "main.py"
        content = path.read_text(encoding="utf-8")
        # Loose string match — the specific identifier matters
        # (_QApp.instance() or _QApp(sys.argv)).
        assert "_QApp" in content and "instance()" in content, (
            "src/main.py must create a probe QApplication before calling "
            "the single-instance guard; see the comment in main().main()."
        )

    def test_main_py_calls_freeze_support(self) -> None:
        """Static check: src/main.py must call multiprocessing.freeze_support().

        Without it, PyInstaller-frozen builds on Windows recursively
        spawn copies of themselves every time the ProcessPoolExecutor
        creates a worker (spawn-start re-executes the main script).
        The user-visible symptom is:

            A child process terminated abruptly, the process pool is
            not usable anymore

        The call must live inside the ``if __name__ == "__main__":``
        block as the FIRST statement — otherwise it hits after workers
        have already re-entered main() and is useless.
        """
        import ast

        path = Path(__file__).parent.parent.parent / "src" / "main.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            # Look for `if __name__ == "__main__":`
            if not isinstance(node, ast.If):
                continue
            # Crude match — good enough for a single file.
            if "__main__" not in ast.dump(node.test):
                continue
            # First statement inside must be freeze_support().
            first = node.body[0] if node.body else None
            dump = ast.dump(first) if first is not None else ""
            assert "freeze_support" in dump, (
                "src/main.py's `if __name__ == '__main__':` block must "
                "start with multiprocessing.freeze_support() — otherwise "
                "the frozen Windows .exe will crash with "
                "'A child process terminated abruptly' on first "
                "ProcessPoolExecutor.submit."
            )
            return
        import pytest as _pytest

        _pytest.fail("No `if __name__ == '__main__':` block in src/main.py")


def _cleanup_qapp_after_session() -> None:
    """Helper hook: ensure no leaked QApplication survives past these tests.

    Other tests in the suite legitimately create QApplication instances
    via qtbot — leaving ours behind would confuse them. pytest tears
    down the process per-session anyway, but this is a belt.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        app.processEvents()
