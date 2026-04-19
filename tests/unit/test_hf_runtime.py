"""Tests for ``src.infrastructure.hf_runtime``.

These guard the HTR/GOT-OCR 2.0 engine against the runtime footguns
documented in the module: Unicode cache paths, surprise online lookups,
and telemetry pings from a frozen desktop app.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.exercise_preflight


def test_configure_hf_runtime_sets_offline_flags(monkeypatch) -> None:
    """``HF_HUB_OFFLINE`` + ``TRANSFORMERS_OFFLINE`` must be set.

    Regression guard: without these, ``from_pretrained`` of a locally-
    bundled model still contacts the Hub to "check for updates" and
    hangs on firewalled networks.
    """
    from src.infrastructure.hf_runtime import configure_huggingface_runtime

    for var in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    ):
        monkeypatch.delenv(var, raising=False)

    configure_huggingface_runtime()

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"


def test_configure_hf_runtime_is_setdefault(monkeypatch) -> None:
    """User overrides must survive a subsequent ``configure`` call."""
    from src.infrastructure.hf_runtime import configure_huggingface_runtime

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")  # user override: allow online
    monkeypatch.setenv("HF_HOME", "/custom/cache")

    configure_huggingface_runtime()

    # Must NOT be clobbered.
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["HF_HOME"] == "/custom/cache"


def test_configure_hf_runtime_redirects_cache_into_user_dir(
    monkeypatch, tmp_path: Path
) -> None:
    """HF_HOME should point somewhere we control, not ``~/.cache``.

    On Windows ``~/.cache/huggingface`` lives under the user profile,
    which may contain Cyrillic characters that trip up safetensors.
    We redirect to ``USER_DATA_DIR/hf-cache`` — still potentially
    non-ASCII, but at least every read goes through Python's
    Unicode-aware ``pathlib`` rather than the default cache path.
    """
    import src.shared.constants as const

    monkeypatch.setattr(const, "USER_DATA_DIR", tmp_path / "user-data")
    for var in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(var, raising=False)

    from src.infrastructure.hf_runtime import configure_huggingface_runtime

    configure_huggingface_runtime()

    hf_home = os.environ.get("HF_HOME", "")
    assert str(tmp_path / "user-data" / "hf-cache") == hf_home, (
        f"HF_HOME={hf_home!r} did not get redirected into USER_DATA_DIR"
    )
