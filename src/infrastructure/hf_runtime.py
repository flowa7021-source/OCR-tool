"""HuggingFace runtime configuration for the HTR (GOT-OCR 2.0) engine.

Prevents the three known footguns that would otherwise bite end users
after the GOT-OCR 2.0 weights are bundled:

1. **Cache path with non-ASCII characters.** By default HuggingFace
   transformers writes its cache to ``~/.cache/huggingface``. On
   Windows that's ``C:\\Users\\<profile>\\.cache\\huggingface`` — and
   any user with Cyrillic / accented / spaced characters in their
   profile name (e.g. ``Т.Н. 020``) triggers random Unicode-path
   failures inside ``safetensors`` and ``tokenizers`` (both are C++
   extensions with historically shaky non-ASCII handling). Redirecting
   to our own dir under ``USER_DATA_DIR`` doesn't make the path
   ASCII either, but keeps every read funnelled through Python's
   ``pathlib``, which IS Unicode-safe.

2. **Online lookups** for a locally-bundled model. By default
   ``AutoModel.from_pretrained`` will still try to contact the Hub to
   check for a newer revision / metadata, even when you pass a local
   directory. In a frozen Windows install this surfaces as a 30-60 s
   startup hang on firewalls that block ``huggingface.co``, then
   cryptic ``LocalEntryNotFoundError`` warnings. ``HF_HUB_OFFLINE=1``
   + ``TRANSFORMERS_OFFLINE=1`` cut that path entirely.

3. **Telemetry.** ``HF_HUB_DISABLE_TELEMETRY=1`` stops the background
   analytics ping — not a correctness issue but it's a GUI desktop
   app, not a notebook, and the ping is just noise in logs.

Call ``configure_huggingface_runtime()`` once at process startup
(both host and worker) BEFORE any ``import transformers`` anywhere.
Subsequent calls are no-ops because we use ``setdefault``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def configure_huggingface_runtime() -> None:
    """Set ``HF_*`` / ``TRANSFORMERS_*`` env vars for offline, local-only use.

    Uses ``setdefault`` so a power user who manually exported an
    override (e.g. pointing at a shared team cache) keeps control.
    """
    # Redirect the cache into our own user-data area. Created lazily
    # by transformers on first use, so we don't mkdir here.
    try:
        from src.shared.constants import USER_DATA_DIR

        cache_dir = str(USER_DATA_DIR / "hf-cache")
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        # HUGGINGFACE_HUB_CACHE is the newer name in huggingface_hub 0.20+;
        # setting both keeps us compatible across transformers versions.
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)
    except Exception:  # noqa: BLE001 — constants import failure is pathological
        logger.debug("Could not resolve USER_DATA_DIR for HF cache", exc_info=True)

    # Offline mode: never hit huggingface.co. The GOT-OCR 2.0 weights
    # we use are bundled with the installer and loaded from a local
    # path; we don't want transformers pinging the Hub "just to check".
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Disable background telemetry pings (desktop app, not a notebook).
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Stop the "symlink warning" banner on Windows — we don't need
    # symlink-backed caches (we load directly from the bundled dir).
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    logger.debug(
        "HF runtime configured: HF_HOME=%s offline=%s",
        os.environ.get("HF_HOME"),
        os.environ.get("HF_HUB_OFFLINE"),
    )
