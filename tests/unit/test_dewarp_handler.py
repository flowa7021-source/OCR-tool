"""Unit tests for :class:`src.core.dewarp_handler.DewarpHandler`.

``page-dewarp`` is an optional runtime dependency — not bundled with
the standard install — so real dewarping is rarely exercised in CI.
These tests focus on the fallback paths that keep the pipeline
running when page-dewarp is unavailable, misbehaving, or absent.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.core.dewarp_handler import DewarpHandler
from src.core.models import DewarpConfig
from src.shared.validators import ValidationError


def _img(shape: tuple[int, int, int] = (60, 90, 3)) -> np.ndarray:
    return np.full(shape, 200, dtype=np.uint8)


class TestValidation:
    def test_rejects_non_ndarray(self) -> None:
        with pytest.raises(ValidationError):
            DewarpHandler().dewarp("not array", DewarpConfig())  # type: ignore[arg-type]

    def test_rejects_empty_image(self) -> None:
        with pytest.raises(ValidationError):
            DewarpHandler().dewarp(np.array([], dtype=np.uint8), DewarpConfig())

    def test_rejects_wrong_config_type(self) -> None:
        with pytest.raises(ValidationError):
            DewarpHandler().dewarp(_img(), "not a DewarpConfig")  # type: ignore[arg-type]


class TestDisabled:
    def test_disabled_returns_input_unchanged(self) -> None:
        """With config.enabled=False the handler is a no-op."""
        image = _img()
        out = DewarpHandler().dewarp(image, DewarpConfig(enabled=False))
        assert out is image  # identity — no copy, no temp dir


class TestFallbackToOriginal:
    """Every failure mode must return the input image, never raise."""

    def test_no_page_dewarp_installed_returns_original(self) -> None:
        """If page_dewarp isn't importable and subprocess fails → identity."""
        image = _img()
        cfg = DewarpConfig(enabled=True)

        # Force both strategies to fail.
        with patch.object(
            DewarpHandler, "_invoke_page_dewarp", return_value=None
        ):
            out = DewarpHandler().dewarp(image, cfg)

        assert out.shape == image.shape
        assert np.array_equal(out, image)

    def test_invoker_returns_nonexistent_path(self, tmp_path: Path) -> None:
        """Invoker claims success but output file doesn't exist → identity."""
        image = _img()
        cfg = DewarpConfig(enabled=True)

        # Return a path that doesn't exist. dewarp must handle it.
        with patch.object(
            DewarpHandler,
            "_invoke_page_dewarp",
            return_value=tmp_path / "ghost.png",
        ):
            out = DewarpHandler().dewarp(image, cfg)

        assert np.array_equal(out, image)

    def test_invoker_returns_corrupt_file(self, tmp_path: Path) -> None:
        """Output file exists but isn't a valid image → identity."""
        image = _img()
        cfg = DewarpConfig(enabled=True)

        # Create a garbage file at a plausible output location.
        bogus = tmp_path / "garbage.png"
        bogus.write_bytes(b"not a real png")

        with patch.object(
            DewarpHandler, "_invoke_page_dewarp", return_value=bogus
        ):
            out = DewarpHandler().dewarp(image, cfg)

        assert np.array_equal(out, image)


class TestSuccessfulDewarp:
    """The happy path: invoker returns a real PNG that we decode back."""

    def test_decodes_valid_output(self, tmp_path: Path) -> None:
        import cv2

        image = _img()
        cfg = DewarpConfig(enabled=True)

        # Build a valid but distinctive PNG so we can tell it came from
        # the invoker, not a stray bytewise copy of the input.
        distinctive = np.full((40, 60, 3), 50, dtype=np.uint8)
        target = tmp_path / "dewarped.png"
        ok, buf = cv2.imencode(".png", distinctive)
        assert ok
        target.write_bytes(buf.tobytes())

        with patch.object(
            DewarpHandler, "_invoke_page_dewarp", return_value=target
        ):
            out = DewarpHandler().dewarp(image, cfg)

        # Output is the DISTINCTIVE image, not the input.
        assert out.shape == distinctive.shape
        assert np.mean(out) < 100  # input was 200, distinctive was 50


class TestSubprocessStrategy:
    def test_subprocess_nonzero_exit_returns_none(self, tmp_path: Path) -> None:
        """A crashing page_dewarp subprocess → ``_try_subprocess`` returns None.

        Guard against the real user symptom from earlier logs
        ("page-dewarp не создал выходной файл"). This is the fallback
        strategy; if it silently returns None on error, the outer
        ``dewarp()`` correctly gives up and returns the original image.
        """
        import subprocess

        fake_proc = subprocess.CompletedProcess(
            args=["python"], returncode=1, stdout="", stderr="boom"
        )
        with patch("subprocess.run", return_value=fake_proc):
            result = DewarpHandler._try_subprocess(
                tmp_path / "in.png", tmp_path, DewarpConfig(enabled=True)
            )
        assert result is None

    def test_subprocess_oserror_returns_none(self, tmp_path: Path) -> None:
        """If the subprocess can't even launch we quietly degrade."""
        with patch(
            "subprocess.run", side_effect=OSError("no such executable")
        ):
            result = DewarpHandler._try_subprocess(
                tmp_path / "in.png", tmp_path, DewarpConfig(enabled=True)
            )
        assert result is None


class TestLocateOutput:
    """``_locate_output`` covers the various output filename conventions."""

    def test_suffix_thresh(self, tmp_path: Path) -> None:
        input_path = tmp_path / "foo.png"
        input_path.write_bytes(b"")
        expected = tmp_path / "foo_thresh.png"
        expected.write_bytes(b"ok")
        assert DewarpHandler._locate_output(input_path, tmp_path) == expected

    def test_suffix_dewarp(self, tmp_path: Path) -> None:
        input_path = tmp_path / "bar.png"
        input_path.write_bytes(b"")
        expected = tmp_path / "bar_dewarp.png"
        expected.write_bytes(b"ok")
        assert DewarpHandler._locate_output(input_path, tmp_path) == expected

    def test_any_png_in_work_dir_fallback(self, tmp_path: Path) -> None:
        """If the expected names are missing but *something* .png was
        produced, use that."""
        input_path = tmp_path / "baz.png"
        input_path.write_bytes(b"original-bytes")
        unexpected = tmp_path / "result_42.png"
        unexpected.write_bytes(b"ok")
        out = DewarpHandler._locate_output(input_path, tmp_path)
        assert out is not None
        assert out.name == "result_42.png"

    def test_nothing_produced_returns_none(self, tmp_path: Path) -> None:
        input_path = tmp_path / "quux.png"
        input_path.write_bytes(b"")
        assert DewarpHandler._locate_output(input_path, tmp_path) is None
