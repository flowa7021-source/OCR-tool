"""Persistent OCR result cache keyed by (input file, profile) hash.

Running the same PDF through the same profile is a *very* common user
workflow: tweak regex rules, bump a denoise knob, re-run. Each of
those iterations re-rasterises every page, re-invokes Tesseract on
every page, and rebuilds the searchable PDF from scratch — even
though only the text post-processing differs. Caching the whole
job result (output PDF + text + metadata) gives us an essentially
free second run.

Cache key:

    sha256(input_bytes) + sha256(serialised_profile) + app_version

Both SHA-256s are truncated to 16 hex chars (128 bits, plenty of
collision resistance for this workload) and joined by ``--``.

Entry layout:

    OCR_CACHE_DIR/<key>/
        output.pdf       the searchable PDF the pipeline would have produced
        result.json      the JobResult (minus the transient job_id)

Eviction: LRU by directory mtime. On every *write* we check whether
the cache is over budget (``OCR_CACHE_MAX_BYTES``) and drop the
oldest entries until it fits. Reads don't trigger eviction —
touching the dir mtime on hit is enough to keep popular entries
from being evicted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from src.shared.constants import (
    APP_VERSION,
    OCR_CACHE_DIR,
    OCR_CACHE_MAX_BYTES,
)

if TYPE_CHECKING:
    from src.core.models import JobResult, ProfileData

logger = logging.getLogger(__name__)


_KEY_PREFIX_LEN = 16  # hex chars from each sha256


def _short_sha256(payload: bytes) -> str:
    """SHA-256 truncated to :data:`_KEY_PREFIX_LEN` hex chars."""
    return hashlib.sha256(payload).hexdigest()[:_KEY_PREFIX_LEN]


def _hash_file(path: Path) -> str:
    """Streaming SHA-256 of a file's contents (handles multi-GB PDFs).

    Returns the empty string if the file is missing / unreadable — the
    caller should treat that as a cache miss rather than crash.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()[:_KEY_PREFIX_LEN]
    except OSError as exc:
        logger.debug("SHA-256 of %s failed: %s", path, exc)
        return ""


def _hash_profile(profile: ProfileData) -> str:
    """SHA-256 of a canonicalised JSON dump of the profile.

    The profile's identity / metadata fields are excluded — renaming a
    profile, flipping its builtin flag, or re-importing it (which
    resets ``created_at``) shouldn't invalidate cache entries.
    """
    data = profile.to_dict()
    for noise in ("name", "description", "builtin", "created_at"):
        data.pop(noise, None)
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return _short_sha256(payload)


def cache_key(input_path: Path, profile: ProfileData) -> str:
    """Build the cache key for a given input + profile pair.

    Empty string on hashing failure — callers should treat as miss.
    """
    input_hash = _hash_file(input_path)
    if not input_hash:
        return ""
    profile_hash = _hash_profile(profile)
    # Bake APP_VERSION into the key so a new binary (possibly with a
    # different OCRmyPDF / Tesseract upgrade) invalidates all entries.
    return f"{input_hash}--{profile_hash}--{APP_VERSION}"


def _entry_dir(key: str, root: Path | None = None) -> Path:
    base = root if root is not None else OCR_CACHE_DIR
    return base / key


def lookup(
    input_path: Path,
    profile: ProfileData,
    *,
    cache_root: Path | None = None,
) -> tuple[Path, dict] | None:
    """Return ``(cached_pdf_path, result_dict)`` on hit, else ``None``."""
    key = cache_key(input_path, profile)
    if not key:
        return None
    entry = _entry_dir(key, cache_root)
    pdf = entry / "output.pdf"
    meta = entry / "result.json"
    if not (pdf.exists() and meta.exists()):
        return None
    try:
        result = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Cache meta unreadable at %s: %s", meta, exc)
        return None
    # Touch the entry so LRU eviction keeps popular ones.
    try:
        now = time.time()
        os.utime(entry, (now, now))
    except OSError:
        pass
    logger.info("OCR cache HIT for %s (key=%s)", input_path.name, key)
    return pdf, result


def store(
    input_path: Path,
    profile: ProfileData,
    *,
    output_pdf: Path,
    job_result: JobResult,
    cache_root: Path | None = None,
    max_bytes: int = OCR_CACHE_MAX_BYTES,
) -> None:
    """Save a successful job into the cache. Best-effort, never raises."""
    key = cache_key(input_path, profile)
    if not key:
        return
    base = cache_root if cache_root is not None else OCR_CACHE_DIR
    entry = _entry_dir(key, base)
    try:
        entry.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_pdf, entry / "output.pdf")
        # Strip the job_id so reusing the result for a new job doesn't
        # leak the old ID back into the UI (the pipeline will re-bind
        # the id for the current job).
        meta = asdict(job_result)
        meta.pop("job_id", None)
        (entry / "result.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("OCR cache STORE for %s (key=%s)", input_path.name, key)
    except OSError as exc:
        logger.debug("Cache store failed: %s", exc)
        return
    # Prune outside the critical-path so a slow FS walk doesn't stall
    # the user (but still bounded — worst-case the cache grows by one
    # entry over budget before the next write prunes it).
    try:
        _prune(base, max_bytes)
    except OSError as exc:
        logger.debug("Cache prune failed: %s", exc)


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _entry_stats(entry: Path) -> tuple[int, float] | None:
    """Return ``(size_bytes, mtime)`` for ``entry``, or None on any OSError.

    Robustness: another process may delete the entry between iterdir
    and stat, and the user may manually ``rm -rf`` an entry while we
    iterate. Returning None lets callers transparently skip such
    entries rather than aborting the whole prune pass with an OSError.
    """
    try:
        return _dir_size(entry), entry.stat().st_mtime
    except OSError as exc:
        logger.debug("_entry_stats skipping %s: %s", entry.name, exc)
        return None


def _prune(base: Path, max_bytes: int) -> None:
    if not base.exists():
        return
    try:
        entries = [p for p in base.iterdir() if p.is_dir()]
    except OSError as exc:
        logger.debug("_prune: base dir iterdir failed: %s", exc)
        return
    if not entries:
        return

    # Snapshot size + mtime per entry atomically. If an entry disappears
    # (concurrent manual cleanup, second worker finishing a store), skip
    # it — don't let its absence corrupt the running total or crash the
    # sort comparator on a missing ``stat`` call.
    stats: dict[Path, tuple[int, float]] = {}
    total = 0
    for entry in entries:
        got = _entry_stats(entry)
        if got is None:
            continue
        stats[entry] = got
        total += got[0]

    if total <= max_bytes:
        return

    # Sort oldest-first from the snapshot so concurrent writes to
    # entry.mtime don't destabilise the comparison mid-sort.
    ordered = sorted(stats.items(), key=lambda kv: kv[1][1])
    for entry, (size, _mtime) in ordered:
        if total <= max_bytes:
            break
        try:
            shutil.rmtree(entry, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001 — prune is best-effort
            logger.debug("_prune rmtree failed for %s: %s", entry, exc)
            continue
        total -= size
        logger.info("Cache evicted %s (%.1f MB)", entry.name, size / 1024 / 1024)
