"""Download GOT-OCR 2.0 weights into resources/models/got_ocr2/.

Invoked by .github/workflows/build-installer.yml when HTR bundling is
on. Reuses the same manifest (``GOT_OCR2_SPEC``) that the runtime
downloader uses so the two can never drift.

Skip semantics: if every file is already present at the target path
(size-checked) the script prints a reuse notice and exits 0 without
re-downloading — useful for iterative CI runs with a cached artifact.
"""

from __future__ import annotations

import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("download_got_ocr2")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _target_dir() -> Path:
    return _repo_root() / "resources" / "models" / "got_ocr2"


def _stream_download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` streaming in 1 MB chunks."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc
    tmp.replace(dest)


def _already_complete(spec, target: Path) -> bool:
    if not target.is_dir():
        return False
    for fobj in spec.files:
        path = target / fobj.name
        if not path.is_file():
            return False
        # size_bytes=0 means "any size accepted" per manifest convention.
        if fobj.size_bytes and path.stat().st_size < max(fobj.size_bytes * 0.5, 1024):
            log.warning(
                "%s too small (%d B, expected ~%d B) — will redownload",
                path, path.stat().st_size, fobj.size_bytes,
            )
            return False
    return True


def main() -> int:
    # Add the repo root so `src.infrastructure.model_manager` imports cleanly.
    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))

    try:
        from src.infrastructure.model_manager import GOT_OCR2_SPEC
    except ImportError as exc:
        log.error("Could not import GOT_OCR2_SPEC: %s", exc)
        return 1

    target = _target_dir()
    if _already_complete(GOT_OCR2_SPEC, target):
        log.info("Every GOT-OCR 2.0 file already present at %s — skipping", target)
        return 0

    target.mkdir(parents=True, exist_ok=True)
    for fobj in GOT_OCR2_SPEC.files:
        dest = target / fobj.name
        if dest.is_file() and (
            not fobj.size_bytes or dest.stat().st_size >= fobj.size_bytes * 0.5
        ):
            log.info("Reusing %s (%d bytes)", dest, dest.stat().st_size)
            continue
        log.info("Downloading %s → %s", fobj.url, dest)
        _stream_download(fobj.url, dest)
        log.info("  wrote %d bytes", dest.stat().st_size)

    log.info("GOT-OCR 2.0 ready at %s", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
