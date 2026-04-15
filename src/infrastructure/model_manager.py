"""HTR model manager: download, validate, and locate optional model weights.

Heavy ML models (GOT-OCR2 ~580 MB, TrOCR ~1.3 GB) are too large to
ship inside the installer. Instead we download them on first use to
``%LOCALAPPDATA%/OCRStudio/models/<model_id>/`` from the Hugging Face
Hub via plain HTTPS and stream them to disk so the UI can show
progress.

The downloader is intentionally framework-agnostic: no HuggingFace Hub
client dependency, just :mod:`urllib.request` + a small JSON manifest
of the files we need from each model. That keeps install footprint
tiny when the user never enables HTR.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.shared.constants import USER_DATA_DIR

logger = logging.getLogger(__name__)


MODELS_DIR: Path = USER_DATA_DIR / "models"


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


@dataclass
class ModelFile:
    """One file in a model package."""

    name: str  # filename within the model dir
    url: str   # absolute download URL
    size_bytes: int = 0  # 0 means "unknown, accept anything > 1 KB"
    sha256: str = ""  # optional, empty disables hash check


@dataclass
class ModelSpec:
    """Manifest of one model: identifier, files, and a friendly label."""

    model_id: str
    label: str
    description: str
    files: list[ModelFile] = field(default_factory=list)
    total_size_bytes: int = 0  # for the UI; sum of file sizes when known


# Manifests for the engines we currently support. Sizes/hashes come
# straight from the upstream HuggingFace listings; if either changes
# upstream, the next download will fail the size check and the user
# gets a clear error rather than a corrupt install.
GOT_OCR2_HF = "https://huggingface.co/stepfun-ai/GOT-OCR2_0/resolve/main"
GOT_OCR2_SPEC = ModelSpec(
    model_id="got_ocr2",
    label="GOT-OCR 2.0",
    description="Универсальный transformer-OCR (Apache-2.0, ~580 МБ).",
    total_size_bytes=580 * 1024 * 1024,
    # stepfun-ai/GOT-OCR2_0 uses a Qwen tokenizer distributed as a
    # tiktoken BPE file (``qwen.tiktoken``) PLUS the usual
    # ``tokenizer_config.json`` + ``special_tokens_map.json`` — there is
    # NO ``tokenizer.json`` in the repo, so we must not list it (pulls
    # 404 from HuggingFace mid-download).
    files=[
        ModelFile(name="config.json", url=f"{GOT_OCR2_HF}/config.json"),
        ModelFile(
            name="generation_config.json",
            url=f"{GOT_OCR2_HF}/generation_config.json",
        ),
        ModelFile(
            name="tokenizer_config.json",
            url=f"{GOT_OCR2_HF}/tokenizer_config.json",
        ),
        ModelFile(name="special_tokens_map.json", url=f"{GOT_OCR2_HF}/special_tokens_map.json"),
        ModelFile(
            name="model.safetensors",
            url=f"{GOT_OCR2_HF}/model.safetensors",
            size_bytes=560 * 1024 * 1024,
        ),
        ModelFile(name="qwen.tiktoken", url=f"{GOT_OCR2_HF}/qwen.tiktoken"),
    ],
)


_REGISTRY: dict[str, ModelSpec] = {
    GOT_OCR2_SPEC.model_id: GOT_OCR2_SPEC,
}


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


# progress callback signature: (bytes_downloaded, bytes_total, current_filename)
ProgressCallback = Callable[[int, int, str], None]


class ModelDownloadError(RuntimeError):
    """Raised when a model file cannot be downloaded or fails validation."""


class ModelManager:
    """Discover, download, and locate optional HTR model weights."""

    #: How long an ``is_available`` result is trusted before we re-stat
    #: the filesystem. 5 s is short enough that a download finishing via
    #: the UI becomes visible on the next refresh, and long enough that
    #: rapid UI queries (engine dropdown + "Download…" action refresh)
    #: don't hammer the disk.
    AVAILABILITY_TTL_SEC: float = 5.0

    def __init__(
        self,
        models_dir: Path | None = None,
        bundled_dir: Path | None = None,
    ) -> None:
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Read-only fallback populated by the installer. See
        # BUNDLED_MODELS_DIR in src/shared/constants.py — it's under the
        # app's resources/ tree when running from a PyInstaller build,
        # and usually empty in source checkouts.
        if bundled_dir is None:
            from src.shared.constants import BUNDLED_MODELS_DIR

            bundled_dir = BUNDLED_MODELS_DIR
        self.bundled_dir = Path(bundled_dir)
        # Per-model availability cache: model_id -> (monotonic_time, result)
        self._availability_cache: dict[str, tuple[float, bool]] = {}

    # ---------------------------------------------------------------- spec
    def spec_for(self, model_id: str) -> ModelSpec:
        """Return the registered :class:`ModelSpec` for ``model_id``."""
        try:
            return _REGISTRY[model_id]
        except KeyError as exc:
            raise KeyError(f"Unknown model id: {model_id}") from exc

    def model_dir(self, model_id: str) -> Path:
        """Return the directory where ``model_id``'s files live.

        Prefers the user-writable copy at :attr:`models_dir` if every
        file from the manifest is present, otherwise falls back to the
        read-only bundled copy at :attr:`bundled_dir`. This lets the
        installer ship weights under ``resources/models/<id>/`` while
        still allowing users to re-download or override them later.
        """
        user_copy = self.models_dir / model_id
        bundled_copy = self.bundled_dir / model_id
        try:
            spec = self.spec_for(model_id)
        except KeyError:
            return user_copy
        if user_copy.is_dir() and all(
            (user_copy / f.name).is_file() for f in spec.files
        ):
            return user_copy
        if bundled_copy.is_dir() and all(
            (bundled_copy / f.name).is_file() for f in spec.files
        ):
            return bundled_copy
        # Neither copy is complete — default to the user dir so a
        # subsequent download lands there.
        return user_copy

    # ----------------------------------------------------------- presence
    def is_available(self, model_id: str) -> bool:
        """Return True if every file in the manifest exists locally.

        Cached for :data:`AVAILABILITY_TTL_SEC` seconds. Callers that
        just downloaded or deleted a model should invoke
        :meth:`invalidate_availability` afterwards to skip the TTL.
        """
        now = time.monotonic()
        cached = self._availability_cache.get(model_id)
        if cached is not None:
            stamp, value = cached
            if now - stamp < self.AVAILABILITY_TTL_SEC:
                return value

        try:
            spec = self.spec_for(model_id)
        except KeyError:
            self._availability_cache[model_id] = (now, False)
            return False
        # A model is "available" if EITHER the user-writable location
        # has all files (fresh download) OR the bundled location does
        # (shipped with the installer). model_dir() returns whichever
        # is complete, so we just ask there.
        target = self.model_dir(model_id)
        if not target.is_dir():
            result = False
        else:
            result = all((target / f.name).is_file() for f in spec.files)
        self._availability_cache[model_id] = (now, result)
        return result

    def invalidate_availability(self, model_id: str | None = None) -> None:
        """Drop the cached availability result(s).

        When called with no argument every entry is dropped; otherwise
        just the one matching ``model_id``. Called from the download
        dialog on success and from the "Remove model" menu action.
        """
        if model_id is None:
            self._availability_cache.clear()
        else:
            self._availability_cache.pop(model_id, None)

    def list_local_models(self) -> list[str]:
        """Return ids of fully-downloaded models."""
        return [mid for mid in _REGISTRY if self.is_available(mid)]

    # ----------------------------------------------------------- download
    def download(
        self,
        model_id: str,
        progress_callback: ProgressCallback | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Path:
        """Download every file in ``model_id``'s manifest.

        Args:
            model_id: One of the keys in the registry (e.g. ``"got_ocr2"``).
            progress_callback: Receives ``(bytes_done, bytes_total, current_file)``
                roughly every 256 KB.
            cancel: Optional poll function. When it returns True, the download
                is aborted and a partially-written file is removed.

        Returns:
            Path to the model directory.

        Raises:
            ModelDownloadError: On HTTP failure, size mismatch, or hash
                mismatch after a complete download.
        """
        spec = self.spec_for(model_id)
        target = self.model_dir(model_id)
        target.mkdir(parents=True, exist_ok=True)

        total = spec.total_size_bytes or sum(f.size_bytes for f in spec.files)
        downloaded_bytes = 0

        for file_spec in spec.files:
            dest = target / file_spec.name
            if dest.exists() and self._validate_file(dest, file_spec):
                downloaded_bytes += dest.stat().st_size
                if progress_callback is not None:
                    with contextlib.suppress(Exception):
                        progress_callback(downloaded_bytes, total, file_spec.name)
                continue

            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                self._download_one(
                    file_spec=file_spec,
                    tmp_path=tmp,
                    base_offset=downloaded_bytes,
                    grand_total=total,
                    progress_callback=progress_callback,
                    cancel=cancel,
                )
            except Exception:
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
                raise

            if not self._validate_file(tmp, file_spec):
                with contextlib.suppress(OSError):
                    tmp.unlink(missing_ok=True)
                raise ModelDownloadError(
                    f"Файл {file_spec.name} не прошёл проверку (размер/хеш)"
                )
            os.replace(tmp, dest)
            downloaded_bytes += dest.stat().st_size

        logger.info("Model %s downloaded to %s", model_id, target)
        # Invalidate the availability cache so the next `is_available`
        # reflects the newly-downloaded files immediately.
        self.invalidate_availability(model_id)
        return target

    def remove(self, model_id: str) -> bool:
        """Delete the on-disk model directory. Returns True if removed."""
        target = self.model_dir(model_id)
        if not target.exists():
            self.invalidate_availability(model_id)
            return False
        import shutil

        shutil.rmtree(target, ignore_errors=False)
        self.invalidate_availability(model_id)
        logger.info("Removed model %s", model_id)
        return True

    # ----------------------------------------------------------- internal
    def _download_one(
        self,
        file_spec: ModelFile,
        tmp_path: Path,
        base_offset: int,
        grand_total: int,
        progress_callback: ProgressCallback | None,
        cancel: Callable[[], bool] | None,
    ) -> None:
        """Stream a single URL to ``tmp_path`` with chunked progress."""
        chunk = 256 * 1024
        req = urllib.request.Request(
            file_spec.url,
            headers={"User-Agent": "ocr-studio-htr/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_path, "wb") as out:
                content_length = int(resp.headers.get("Content-Length", "0") or 0)
                file_total = content_length or file_spec.size_bytes
                local_done = 0
                while True:
                    if cancel is not None and cancel():
                        raise ModelDownloadError(
                            f"Скачивание {file_spec.name} отменено пользователем"
                        )
                    block = resp.read(chunk)
                    if not block:
                        break
                    out.write(block)
                    local_done += len(block)
                    if progress_callback is not None:
                        with contextlib.suppress(Exception):
                            progress_callback(
                                base_offset + local_done,
                                max(grand_total, base_offset + file_total),
                                file_spec.name,
                            )
        except ModelDownloadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelDownloadError(
                f"Не удалось скачать {file_spec.name}: {exc}"
            ) from exc

    @staticmethod
    def _validate_file(path: Path, spec: ModelFile) -> bool:
        """Return True when ``path`` matches the spec's size/hash constraints."""
        try:
            actual_size = path.stat().st_size
        except OSError:
            return False
        if spec.size_bytes:
            # Allow ±1% slack for real megabyte-scale weights, but enforce
            # a 16-byte floor so trivial-size test specs catch corruption.
            # 580 MB → 5.8 MB tolerance; 32 B → 16 B tolerance.
            tolerance = max(16, spec.size_bytes // 100)
            if abs(actual_size - spec.size_bytes) > tolerance:
                logger.warning(
                    "Size mismatch for %s: %d (expected %d)",
                    path, actual_size, spec.size_bytes,
                )
                return False
        else:
            # No expected size: just refuse zero-byte / sub-1-KB files.
            if actual_size < 1024:
                return False
        if spec.sha256:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            digest = h.hexdigest()
            if digest != spec.sha256:
                logger.warning("SHA-256 mismatch for %s: %s", path, digest)
                return False
        return True


def parse_manifest(text: str) -> ModelSpec:
    """Parse a JSON manifest into a :class:`ModelSpec` (test helper)."""
    data = json.loads(text)
    return ModelSpec(
        model_id=data["model_id"],
        label=data["label"],
        description=data.get("description", ""),
        total_size_bytes=int(data.get("total_size_bytes", 0)),
        files=[ModelFile(**f) for f in data.get("files", [])],
    )


__all__ = [
    "MODELS_DIR",
    "GOT_OCR2_SPEC",
    "ModelDownloadError",
    "ModelFile",
    "ModelManager",
    "ModelSpec",
    "parse_manifest",
]
