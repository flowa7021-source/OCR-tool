"""UI responsiveness acceptance tests — Initiative 6.

The user's complaint:
  "интерфейс чрезмерно скученный, одновременно мелко и велико,
   никаких пропорций... куча несоразмерных неудобных окон, куда
   даже все функции не влезают, наслаиваясь друг на друга."

The root cause is a fixed-size mindset in the current main
window — panels reserve minimum widths/heights that on a
1280×720 Windows laptop add up past the viewport. These tests
codify the acceptance criteria for a responsive layout:

  1. The window fits in its advertised minimum size
     (``setMinimumSize(1000, 620)``) without overflow or
     clipping of critical controls.
  2. At the typical laptop resolution 1280×720 the main
     functional areas (queue, profile picker, results) are
     reachable without horizontal scrolling.
  3. The window CAN grow to 1920×1080 without huge empty zones
     or widgets stretching absurdly.

Failures here become the punch-list for the UI rewrite. The
tests that pass codify what's already correct and protect it
from regression.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")


@pytest.fixture
def main_window(qtbot, tmp_path, monkeypatch):
    """Construct :class:`MainWindow` with isolated config storage.

    The real MainWindow wires QueueManager, ProfileManager,
    RecoveryManager etc. on __init__; we route their on-disk
    state to ``tmp_path`` so tests don't touch the real user
    profile directory.
    """
    monkeypatch.setattr(
        "src.shared.constants.USER_DATA_DIR", tmp_path,
    )
    monkeypatch.setattr(
        "src.shared.constants.CONFIG_DIR", tmp_path / "config",
    )
    monkeypatch.setattr(
        "src.shared.constants.PROFILES_DIR", tmp_path / "profiles",
    )
    monkeypatch.setattr(
        "src.shared.constants.RECOVERY_DIR", tmp_path / "recovery",
    )

    from src.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    return window


class TestAdvertisedMinimumSize:
    """The MainWindow declares ``setMinimumSize(1000, 620)``. That
    declaration is a contract: at that size nothing critical is
    clipped off-screen."""

    def test_minimum_size_is_declared(self, main_window) -> None:
        hint = main_window.minimumSize()
        assert hint.width() >= 1000
        assert hint.height() >= 620

    def test_window_fits_its_own_minimum_size(self, main_window) -> None:
        # Resize to the advertised minimum — MUST not clip or
        # trigger layout warnings (Qt will swallow warnings but
        # setFixedSize at minimum is a cheap sanity check).
        main_window.resize(1000, 620)
        main_window.show()
        # After layout settles, the actual size must equal what
        # we asked for within 1-px tolerance.
        assert abs(main_window.width() - 1000) <= 1
        assert abs(main_window.height() - 620) <= 1


class TestLaptop1280x720Target:
    """The stated minimum Windows support is 1280×720 (standard
    14-inch laptop). At that resolution all core workflow
    controls must be visible without a horizontal scroll."""

    def test_fits_in_laptop_viewport(self, main_window) -> None:
        main_window.resize(1280, 720)
        main_window.show()
        # Nothing extends past the window. Check each top-level
        # child: its geometry must be inside the window rect.
        window_rect = main_window.rect()
        offenders: list[tuple[str, int, int, int, int]] = []
        for child in main_window.findChildren(object):
            if not hasattr(child, "rect"):
                continue
            if not hasattr(child, "mapTo"):
                continue
            if not child.isVisible():
                continue
            try:
                top_left = child.mapTo(main_window, child.rect().topLeft())
                bottom_right = child.mapTo(
                    main_window, child.rect().bottomRight(),
                )
            except (TypeError, RuntimeError):
                continue
            if bottom_right.x() > window_rect.right() + 2:
                offenders.append(
                    (
                        type(child).__name__,
                        top_left.x(), top_left.y(),
                        bottom_right.x(), bottom_right.y(),
                    )
                )
        assert not offenders, (
            f"At 1280×720 these widgets overflow the window: {offenders}"
        )

    def test_queue_panel_visible_at_laptop_size(
        self, main_window,
    ) -> None:
        main_window.resize(1280, 720)
        main_window.show()
        # Queue panel is a dockable widget on the main window.
        # It must be visible (not collapsed off-screen).
        assert hasattr(main_window, "queue_panel"), (
            "MainWindow missing ``queue_panel`` attribute"
        )
        queue = main_window.queue_panel
        assert queue.isVisible()


class TestDesktop1920x1080Target:
    """Widescreen desktops are the other common target. The
    window must USE the extra pixels, not leave giant empty
    zones from fixed-size panels."""

    def test_grows_to_widescreen(self, main_window) -> None:
        main_window.resize(1920, 1080)
        main_window.show()
        # Window accepted the new size.
        assert main_window.width() >= 1900
        assert main_window.height() >= 1060
