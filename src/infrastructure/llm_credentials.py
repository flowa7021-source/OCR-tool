"""Bridge between :class:`AppSettings` and the LLM-fallback env var.

The ТН / УПД parser's :func:`src.tn_parser.llm_fallback.improve_row`
uses the stock ``anthropic.Anthropic()`` client, which reads the
``ANTHROPIC_API_KEY`` env var on construction. Rather than passing
the key through every call site — orchestrator, pipeline, CLI —
we push it into the process environment once at app startup (and
again whenever the user edits it in Preferences) and let the
bundled parser pick it up with zero additional plumbing.

This module is the single place that writes to that env var. Keeping
the translation "settings → env" in one helper means:

* The secret never travels through function signatures where a
  misplaced ``repr()`` could leak it into the application log.
* Downstream code (orchestrator, pipeline tests, ``improve_row``
  itself) stays decoupled from :mod:`src.infrastructure.config_storage`.
* Tests that want to exercise the LLM path without leaking a real
  key to the environment can call :func:`apply_to_environment` with
  an ``AppSettings`` whose key is an explicit placeholder.
"""

from __future__ import annotations

import logging
import os

from src.infrastructure.config_storage import AppSettings

logger = logging.getLogger(__name__)


_ENV_VAR = "ANTHROPIC_API_KEY"


def apply_to_environment(settings: AppSettings) -> bool:
    """Sync ``settings.anthropic_api_key`` into the process environment.

    Invariants:
      * Non-empty ``settings.anthropic_api_key`` → ``os.environ``
        gets set to that value (overwriting any pre-existing value —
        user-edited key wins over a shell export).
      * Empty ``settings.anthropic_api_key`` → the env var is
        *removed* iff the application set it in a prior call.
        An env var that came from the user's shell before startup
        is left intact so users who prefer the CLI-env workflow
        over the Preferences dialog aren't surprised.

    The prior-call tracking is implemented via a module-level
    sentinel attribute on :data:`os.environ` (``_ocr_studio_managed``)
    — a benign string marker that lets us distinguish "the dialog
    put this here" from "the shell put this here".

    Returns ``True`` when the env var ends up populated (the caller
    can use this to log a concise "LLM active / inactive" line once
    at startup), ``False`` when empty.
    """
    key = (settings.anthropic_api_key or "").strip()
    previously_managed = os.environ.get(f"{_ENV_VAR}__managed") == "ocr-studio"

    if key:
        os.environ[_ENV_VAR] = key
        os.environ[f"{_ENV_VAR}__managed"] = "ocr-studio"
        # Log at DEBUG (never INFO) and NEVER log the key itself —
        # application logs get uploaded in support tickets.
        logger.debug(
            "LLM credentials applied to environment (len=%d)", len(key),
        )
        return True

    if previously_managed and _ENV_VAR in os.environ:
        del os.environ[_ENV_VAR]
        del os.environ[f"{_ENV_VAR}__managed"]
        logger.debug("LLM credentials cleared from environment")
    return False


def is_active() -> bool:
    """Return True if the LLM env var is currently populated.

    Convenience for UI code that wants to render a badge / status
    indicator without re-reading settings. Mirrors what
    :func:`src.tn_parser.llm_fallback.improve_row` sees at the
    moment of call.
    """
    return bool(os.environ.get(_ENV_VAR, "").strip())
