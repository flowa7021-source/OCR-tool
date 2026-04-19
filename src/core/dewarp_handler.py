"""Page dewarping wrapper around the `page-dewarp` library.

`page-dewarp` is historically a CLI tool and its Python API has shifted between
versions. This module tries several known entry points and, failing that,
falls back to invoking the installed ``page-dewarp`` command as a subprocess.
On any failure the original image is returned unchanged so that the caller's
pipeline continues to function.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.core.models import DewarpConfig
from src.shared.validators import ValidationError

logger = logging.getLogger(__name__)


class DewarpHandler:
    """Apply cubic-sheet dewarping to a page image.

    The handler favours in-process invocation of the ``page_dewarp`` library
    because that keeps memory overhead small, but falls back to a subprocess
    call when the Python API is unavailable. Any failure is logged and the
    input image is returned unchanged.

    Thread safety:
        Each call writes to a fresh temporary directory and catches all
        exceptions, so instances are safe to use across threads. The
        underlying ``page_dewarp`` package is, however, single-threaded and
        relatively CPU-bound.

    Note:
        The exact ``page_dewarp`` invocation may need adjustment once the
        library version is pinned in the project's environment. The wrapper
        is defensive on purpose — page-dewarp's public API is minimal.
    """

    #: File name prefix used inside the temporary working directory.
    _TEMP_PREFIX = "ocrstudio_dewarp_"

    def dewarp(self, image: np.ndarray, config: DewarpConfig) -> np.ndarray:
        """Dewarp ``image`` according to ``config``.

        Args:
            image: Grayscale or BGR image as a numpy array.
            config: Dewarp parameters.

        Returns:
            The dewarped image, or the original image if dewarping is
            disabled or fails.

        Raises:
            ValidationError: If the inputs are obviously invalid
                (wrong type / empty array).
        """
        if not isinstance(image, np.ndarray):
            raise ValidationError(
                f"image должен быть np.ndarray, получено: {type(image).__name__}"
            )
        if image.size == 0:
            raise ValidationError("Пустое изображение для dewarp")
        if not isinstance(config, DewarpConfig):
            raise ValidationError(
                f"config должен быть DewarpConfig, получено: {type(config).__name__}"
            )

        if not config.enabled:
            return image

        tmp_dir: Path | None = None
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix=self._TEMP_PREFIX))
            input_path = tmp_dir / "input.png"
            # Unicode-safe write — ``cv2.imwrite`` routes through fopen
            # and fails on non-ASCII Windows paths (e.g. cyrillic user
            # names). Encode with cv2 then write via Python's IO layer.
            # NOTE: do NOT add a local ``import numpy as np`` in this
            # block — Python would then treat ``np`` as function-local
            # throughout, shadowing the module-level import used in the
            # type-guard above and raising UnboundLocalError.
            try:
                ok, buf = cv2.imencode(".png", image)
                if not ok or buf is None:
                    logger.warning(
                        "cv2.imencode вернул пустой буфер, пропускаем dewarp"
                    )
                    return image
                input_path.write_bytes(buf.tobytes())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Не удалось записать временный файл для dewarp (%s): %s",
                    input_path, exc,
                )
                return image

            result_path = self._invoke_page_dewarp(input_path, tmp_dir, config)
            if result_path is None or not result_path.exists():
                logger.warning("page-dewarp не создал выходной файл — возвращаем оригинал")
                return image

            # Same Unicode concern for the read — use the module-level
            # np import (see note above).
            try:
                raw = np.frombuffer(result_path.read_bytes(), dtype=np.uint8)
                result = (
                    cv2.imdecode(raw, cv2.IMREAD_UNCHANGED) if raw.size else None
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Не удалось прочитать результат page-dewarp (%s): %s",
                    result_path, exc,
                )
                return image
            if result is None:
                logger.warning(
                    "Не удалось распарсить результат page-dewarp: %s", result_path
                )
                return image
            return result
        except Exception as exc:  # noqa: BLE001 — page_dewarp can raise anything
            logger.warning("Dewarp завершился с ошибкой, возвращаем оригинал: %s", exc)
            return image
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_page_dewarp(
        self,
        input_path: Path,
        work_dir: Path,
        config: DewarpConfig,
    ) -> Path | None:
        """Try each known entry point in sequence.

        Returns:
            Path to the produced image, or ``None`` if every strategy failed.
        """
        strategies: list[Callable[[Path, Path, DewarpConfig], Path | None]] = [
            self._try_python_api,
            self._try_subprocess,
        ]
        for strategy in strategies:
            try:
                result = strategy(input_path, work_dir, config)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Стратегия %s не сработала: %s", strategy.__name__, exc)
                continue
            if result is not None and result.exists():
                return result
        return None

    @staticmethod
    def _try_python_api(
        input_path: Path, work_dir: Path, config: DewarpConfig
    ) -> Path | None:
        """Attempt the in-process Python API."""
        # Preferred: a high-level class. This is a best-effort import path
        # that may not exist on every version of the library.
        try:
            from page_dewarp.dewarp import PageDewarper  # type: ignore[import-not-found]

            dewarper_cls: Any = PageDewarper
        except ImportError:
            dewarper_cls = None

        if dewarper_cls is not None:
            logger.debug("Пробуем page_dewarp.dewarp.PageDewarper")
            try:
                dewarper: Any = dewarper_cls(str(input_path))
                # The exact method name differs between versions; try a couple.
                for method_name in ("dewarp", "run", "process"):
                    method = getattr(dewarper, method_name, None)
                    if callable(method):
                        method()
                        break
            except Exception as exc:  # noqa: BLE001
                logger.debug("PageDewarper вызов завершился неудачно: %s", exc)
                return None
            return DewarpHandler._locate_output(input_path, work_dir)

        # Fallback: module-level entry point.
        try:
            import page_dewarp  # type: ignore[import-not-found]
        except ImportError:
            return None

        logger.debug("Пробуем page_dewarp (module-level API)")
        for func_name in ("dewarp", "main", "run"):
            func = getattr(page_dewarp, func_name, None)
            if callable(func):
                try:
                    func(str(input_path))
                except TypeError:
                    # Some versions take argv-style list.
                    try:
                        func([str(input_path)])
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("page_dewarp.%s не сработала: %s", func_name, exc)
                        continue
                except Exception as exc:  # noqa: BLE001
                    logger.debug("page_dewarp.%s не сработала: %s", func_name, exc)
                    continue
                output = DewarpHandler._locate_output(input_path, work_dir)
                if output is not None:
                    return output
        return None

    @staticmethod
    def _try_subprocess(
        input_path: Path, work_dir: Path, config: DewarpConfig
    ) -> Path | None:
        """Attempt invocation via a subprocess (``python -m page_dewarp``)."""
        cmd = [sys.executable, "-m", "page_dewarp", str(input_path)]
        logger.debug("Пробуем subprocess: %s", cmd)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.debug("subprocess page_dewarp не запустился: %s", exc)
            return None
        if proc.returncode != 0:
            logger.debug(
                "page_dewarp subprocess вернул код %s: %s",
                proc.returncode,
                proc.stderr.strip(),
            )
            return None
        return DewarpHandler._locate_output(input_path, work_dir)

    @staticmethod
    def _locate_output(input_path: Path, work_dir: Path) -> Path | None:
        """Locate the dewarped file produced by page-dewarp.

        ``page-dewarp`` typically writes the result next to the input with a
        ``_thresh`` suffix (and sometimes a ``.png`` extension). We try a few
        common patterns.
        """
        stem = input_path.stem
        candidates = [
            input_path.with_name(f"{stem}_thresh.png"),
            input_path.with_name(f"{stem}_dewarp.png"),
            work_dir / f"{stem}_thresh.png",
            work_dir / f"{stem}_dewarp.png",
            Path.cwd() / f"{stem}_thresh.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        # Final heuristic: any *.png file in work_dir besides the input itself.
        for png in work_dir.glob("*.png"):
            if png.resolve() != input_path.resolve():
                return png
        return None
