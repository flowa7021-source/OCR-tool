"""Tests for :mod:`src.infrastructure.llm_credentials`.

The helper is narrow but safety-critical — it's the only bridge
between the user's saved API key and the parser's ``improve_row``
client. Contracts:

* Non-empty key → ``ANTHROPIC_API_KEY`` set + an internal sentinel
  marks the env as managed by the app.
* Empty key + prior sentinel → env cleared (and sentinel cleared).
* Empty key + NO prior sentinel → env left alone (shell-set env
  var from a CLI user survives).
* ``is_active`` reflects the post-sync state of the env var.
* API key NEVER appears in log records (the helper must stay
  silent on support-ticket uploads).
"""

from __future__ import annotations

import logging
import os

import pytest

from src.infrastructure.config_storage import AppSettings
from src.infrastructure.llm_credentials import (
    _ENV_VAR,
    apply_to_environment,
    is_active,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the LLM env-var and its sentinel before each test.

    ``apply_to_environment`` mutates process-global state; without
    this fixture a test would leak state into the next one and the
    "shell-set var survives" assertion would flip depending on
    execution order.
    """
    monkeypatch.delenv(_ENV_VAR, raising=False)
    monkeypatch.delenv(f"{_ENV_VAR}__managed", raising=False)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_empty_key_returns_false_and_does_not_set_env() -> None:
    """Offline-first default: empty key → env stays unset → False."""
    settings = AppSettings()
    assert apply_to_environment(settings) is False
    assert _ENV_VAR not in os.environ
    assert is_active() is False


def test_non_empty_key_sets_env_var() -> None:
    """Non-empty key → env populated, sentinel set, returns True."""
    settings = AppSettings(anthropic_api_key="sk-ant-api03-testvalue")
    assert apply_to_environment(settings) is True
    assert os.environ[_ENV_VAR] == "sk-ant-api03-testvalue"
    assert os.environ[f"{_ENV_VAR}__managed"] == "ocr-studio"
    assert is_active() is True


def test_whitespace_only_key_is_treated_as_empty() -> None:
    """Users routinely paste a key with trailing newline; strip it."""
    settings = AppSettings(anthropic_api_key="   \n\t  ")
    assert apply_to_environment(settings) is False
    assert _ENV_VAR not in os.environ


def test_apply_overwrites_prior_managed_value() -> None:
    """Second save with a different key replaces the env var."""
    apply_to_environment(AppSettings(anthropic_api_key="first"))
    assert os.environ[_ENV_VAR] == "first"
    apply_to_environment(AppSettings(anthropic_api_key="second"))
    assert os.environ[_ENV_VAR] == "second"


# ---------------------------------------------------------------------------
# Shell-set env preservation
# ---------------------------------------------------------------------------


def test_empty_key_leaves_shell_set_env_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing shell-set env var (no sentinel) survives a clear.

    CLI user: ``export ANTHROPIC_API_KEY=...`` before launching the
    app. The Preferences dialog leaves the field empty. We MUST NOT
    wipe the shell-set value — it's the user's explicit opt-in via
    a different channel.
    """
    monkeypatch.setenv(_ENV_VAR, "shell-set-value")
    # Note: no sentinel — this is how a shell-export gets observed
    # when app code first touches the env.
    settings = AppSettings()  # empty key
    assert apply_to_environment(settings) is False
    assert os.environ[_ENV_VAR] == "shell-set-value"


def test_empty_key_clears_managed_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing env with sentinel → clears on empty save.

    User wrote a key, then deleted it in Preferences. The helper
    must clean up after itself.
    """
    monkeypatch.setenv(_ENV_VAR, "previous-key")
    monkeypatch.setenv(f"{_ENV_VAR}__managed", "ocr-studio")
    settings = AppSettings()
    assert apply_to_environment(settings) is False
    assert _ENV_VAR not in os.environ
    assert f"{_ENV_VAR}__managed" not in os.environ


# ---------------------------------------------------------------------------
# Log hygiene
# ---------------------------------------------------------------------------


def test_key_never_appears_in_log_records(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Support-ticket logs must not carry the user's API key."""
    settings = AppSettings(anthropic_api_key="sk-ant-api03-SECRETSHOULDNEVERLEAK")
    with caplog.at_level(logging.DEBUG):
        apply_to_environment(settings)
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "SECRETSHOULDNEVERLEAK" not in joined
    # The log line IS allowed to show the key length (useful support
    # breadcrumb) — the test above verifies the ACTUAL value is absent.
