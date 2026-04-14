"""Lightweight check for a newer OCR Studio release on GitHub.

Pulls ``/releases/latest`` from the GitHub REST API, parses the
``tag_name``, and compares it against the current :data:`APP_VERSION`.
Runs in a daemon thread so the main event loop is never blocked; the
result is reported via a callback, never via return.

No authentication required — anonymous GitHub API allows 60 requests
per hour per IP, and we use at most one per 24 hours.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from src.shared.constants import APP_VERSION

logger = logging.getLogger(__name__)

#: Default repository to query. Overridable for forks / tests.
DEFAULT_REPO: str = "flowa7021-source/ocr-tool"

#: UTC seconds between automatic checks.
CHECK_INTERVAL_SEC: int = 24 * 3600


@dataclass
class UpdateInfo:
    """Result of an update probe."""

    current_version: str
    latest_version: str
    is_newer: bool
    release_url: str
    published_at: str  # ISO-8601 string from GitHub
    body: str  # release notes; may be empty


# Callback receives ``UpdateInfo`` on success, ``None`` on any failure
# (network error, parse error, no tags yet). Designed so the UI can
# simply ignore the no-op case.
UpdateCallback = Callable[["UpdateInfo | None"], None]


def _parse_version(tag: str) -> tuple[int, ...]:
    """Turn ``v1.2.3`` or ``1.2.3-dev.abc123`` into a comparable tuple.

    Non-numeric trailers (``-rc1``, ``-dev.abc123``) sort BEFORE the
    corresponding stable release, matching PEP-440 / semver ordering.
    """
    cleaned = tag.lstrip("vV").strip()
    # Strip a PEP440-ish suffix after the first non-"0-9." character.
    head: list[str] = []
    for ch in cleaned:
        if ch.isdigit() or ch == ".":
            head.append(ch)
        else:
            break
    parts = [p for p in "".join(head).split(".") if p]
    return tuple(int(p) for p in parts) if parts else (0,)


def is_newer(current: str, candidate: str) -> bool:
    """Return True if ``candidate`` represents a release newer than ``current``."""
    try:
        return _parse_version(candidate) > _parse_version(current)
    except ValueError:
        return False


def check_once(
    repo: str = DEFAULT_REPO,
    *,
    timeout: float = 10.0,
) -> UpdateInfo | None:
    """Synchronous probe. Returns ``None`` on any failure.

    Call :func:`check_async` from UI code — this function blocks on I/O.
    """
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(url, headers={"User-Agent": "ocr-studio-update/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                logger.info("Update check: HTTP %s", resp.status)
                return None
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — network / JSON / SSL all acceptable
        logger.info("Update check failed: %s", exc)
        return None

    tag = str(data.get("tag_name", ""))
    if not tag:
        return None
    info = UpdateInfo(
        current_version=APP_VERSION,
        latest_version=tag,
        is_newer=is_newer(APP_VERSION, tag),
        release_url=str(data.get("html_url", f"https://github.com/{repo}/releases")),
        published_at=str(data.get("published_at", "")),
        body=str(data.get("body", "")),
    )
    logger.info(
        "Update check: current=%s latest=%s newer=%s",
        APP_VERSION, info.latest_version, info.is_newer,
    )
    return info


def check_async(
    callback: UpdateCallback,
    *,
    repo: str = DEFAULT_REPO,
    timeout: float = 10.0,
) -> threading.Thread:
    """Run :func:`check_once` on a daemon thread.

    Returns the started thread so callers that care about lifecycle
    (tests, graceful shutdown) can join it. The callback receives
    ``None`` on every failure path so downstream code doesn't need to
    differentiate between "no network", "malformed JSON" and "hit
    rate limit" — for the UI they are all the same "no update info
    right now".
    """

    def _run() -> None:
        info = check_once(repo=repo, timeout=timeout)
        try:
            callback(info)
        except Exception:  # noqa: BLE001 — never crash the thread
            logger.exception("Update-check callback raised")

    thread = threading.Thread(target=_run, name="ocr-update-check", daemon=True)
    thread.start()
    return thread
