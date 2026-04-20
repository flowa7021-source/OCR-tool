"""Pytest-qt smoke for the LLM-fallback controls in PreferencesDialog.

The dialog already has its own persistence tests for workers / cache /
theme; this file only covers the additions from PR #8:

* The ``Anthropic API key`` input exists and uses a non-plaintext echo
  mode so screen-shared support sessions don't accidentally reveal it.
* Saving a non-empty key writes it to disk AND pushes it into the
  process environment via :func:`apply_to_environment`.
* Clearing the key in the dialog clears the env var too (managed
  sentinel path — shell-set CLI vars are already covered in
  ``test_llm_credentials``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

from src.infrastructure.config_storage import SettingsStorage  # noqa: E402
from src.infrastructure.llm_credentials import _ENV_VAR  # noqa: E402
from src.ui.preferences_dialog import PreferencesDialog  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh env on every test so sentinel tracking behaves deterministically."""
    monkeypatch.delenv(_ENV_VAR, raising=False)
    monkeypatch.delenv(f"{_ENV_VAR}__managed", raising=False)


# ---------------------------------------------------------------------------
# Dialog renders the field
# ---------------------------------------------------------------------------


def test_dialog_renders_api_key_field(qapp: QApplication, tmp_path: Path) -> None:
    """The API key field exists and uses a masked echo mode."""
    storage = SettingsStorage(config_dir=tmp_path)
    dialog = PreferencesDialog(storage)
    assert dialog.llm_api_key is not None
    # Password-echo-on-edit keeps the key out of screen-share
    # recordings when the user loads a saved profile.
    assert dialog.llm_api_key.echoMode() == QLineEdit.EchoMode.PasswordEchoOnEdit


def test_dialog_preloads_stored_key(qapp: QApplication, tmp_path: Path) -> None:
    """Existing key in settings.json is shown in the field on open."""
    storage = SettingsStorage(config_dir=tmp_path)
    s = storage.load()
    s.anthropic_api_key = "sk-ant-api03-preloaded"
    storage.save(s)

    dialog = PreferencesDialog(storage)
    assert dialog.llm_api_key.text() == "sk-ant-api03-preloaded"


# ---------------------------------------------------------------------------
# Save flow pushes the key to settings + env
# ---------------------------------------------------------------------------


def test_save_persists_key_and_updates_env(
    qapp: QApplication, tmp_path: Path,
) -> None:
    """OK → key lands in settings.json AND in ANTHROPIC_API_KEY."""
    storage = SettingsStorage(config_dir=tmp_path)
    dialog = PreferencesDialog(storage)
    dialog.llm_api_key.setText("  sk-ant-api03-dialog-save  ")  # with whitespace

    dialog._on_accept()

    on_disk = json.loads(
        (tmp_path / "settings.json").read_text(encoding="utf-8"),
    )
    # Stored trimmed — clipboard-pasted keys routinely come with
    # trailing newlines that would silently invalidate the credential.
    assert on_disk["anthropic_api_key"] == "sk-ant-api03-dialog-save"
    assert os.environ[_ENV_VAR] == "sk-ant-api03-dialog-save"


def test_clearing_key_in_dialog_clears_env(
    qapp: QApplication, tmp_path: Path,
) -> None:
    """Existing key + dialog-cleared → env cleared (managed path)."""
    storage = SettingsStorage(config_dir=tmp_path)
    s = storage.load()
    s.anthropic_api_key = "sk-ant-api03-to-be-removed"
    storage.save(s)

    # First dialog: load → immediately save → env populated.
    dialog1 = PreferencesDialog(storage)
    dialog1._on_accept()
    assert os.environ[_ENV_VAR] == "sk-ant-api03-to-be-removed"

    # Second dialog: user clears the field and saves.
    dialog2 = PreferencesDialog(storage)
    dialog2.llm_api_key.setText("")
    dialog2._on_accept()

    on_disk = json.loads(
        (tmp_path / "settings.json").read_text(encoding="utf-8"),
    )
    assert on_disk["anthropic_api_key"] == ""
    assert _ENV_VAR not in os.environ


def test_cancel_does_not_write_env(
    qapp: QApplication, tmp_path: Path,
) -> None:
    """Reject (Cancel) → env left untouched even if user typed a key."""
    storage = SettingsStorage(config_dir=tmp_path)
    dialog = PreferencesDialog(storage)
    dialog.llm_api_key.setText("sk-ant-api03-not-saved")
    dialog._on_reject()

    assert _ENV_VAR not in os.environ
    on_disk_path = tmp_path / "settings.json"
    if on_disk_path.exists():
        on_disk = json.loads(on_disk_path.read_text(encoding="utf-8"))
        assert on_disk.get("anthropic_api_key", "") == ""
