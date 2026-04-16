"""Thin wrapper around :func:`ocrmypdf.ocr`.

The wrapper deliberately disables every built-in OCRmyPDF preprocessing step
(``deskew``, ``clean``, ``remove_background``, ``tesseract_thresholding``) since
our own :class:`~src.core.image_preprocessor.ImagePreprocessor` has already
produced a cleaned, binarised, deskewed PDF by the time we invoke OCRmyPDF.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.models import OCRConfig

logger = logging.getLogger(__name__)


class OCRmyPDFError(RuntimeError):
    """Raised when ``ocrmypdf.ocr()`` terminates with a non-zero exit code."""

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class OCRmyPDFOptions:
    """Parameters forwarded to :func:`ocrmypdf.ocr`.

    Attributes:
        input_file: Path to the preprocessed input PDF.
        output_file: Path where OCRmyPDF will write the OCR'd PDF.
        language: Tesseract language string (e.g. ``"rus+eng"``).
        oem: Tesseract OCR Engine Mode.
        psm: Tesseract Page Segmentation Mode.
        optimize: OCRmyPDF optimize level (0-3).
        skip_text: If True, pages with existing text are skipped.
        tesseract_timeout: Per-page timeout for Tesseract, in seconds.
        char_whitelist: Optional character whitelist string.
        char_blacklist: Optional character blacklist string.
        progress_bar: Optional ``(current, total)`` callback. Because
            ``ocrmypdf.ocr`` does not expose a Python-level progress API
            compatible with a callback, this is invoked once at start and
            once at completion.
        extra: Freeform dict merged into the ``ocrmypdf.ocr`` call for
            advanced callers.
    """

    input_file: Path
    output_file: Path
    language: str
    oem: int
    psm: int
    optimize: int
    skip_text: bool
    tesseract_timeout: int
    char_whitelist: str = ""
    char_blacklist: str = ""
    progress_bar: Callable[[int, int], None] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def map_ocr_config(
    cfg: OCRConfig,
    input_file: Path,
    output_file: Path,
    progress_bar: Callable[[int, int], None] | None = None,
) -> OCRmyPDFOptions:
    """Build :class:`OCRmyPDFOptions` from our :class:`OCRConfig`.

    Args:
        cfg: OCR configuration from the active profile.
        input_file: Path to the preprocessed PDF.
        output_file: Destination PDF path.
        progress_bar: Optional progress callback.

    Returns:
        Populated :class:`OCRmyPDFOptions` ready to pass to :func:`run_ocrmypdf`.
    """
    return OCRmyPDFOptions(
        input_file=input_file,
        output_file=output_file,
        language=cfg.tesseract_language_string,
        oem=int(cfg.oem),
        psm=int(cfg.psm),
        optimize=int(cfg.optimize_level),
        skip_text=cfg.skip_text,
        tesseract_timeout=cfg.tesseract_timeout,
        char_whitelist=cfg.char_whitelist,
        char_blacklist=cfg.char_blacklist,
        progress_bar=progress_bar,
    )


def _build_tesseract_config(options: OCRmyPDFOptions) -> list[str] | None:
    """Assemble the ``tesseract_config`` list for whitelist/blacklist.

    Returns:
        List of config strings, or ``None`` if neither whitelist nor
        blacklist is set.
    """
    extras: list[str] = []
    if options.char_whitelist:
        extras.append(f"-c tessedit_char_whitelist={options.char_whitelist}")
    if options.char_blacklist:
        extras.append(f"-c tessedit_char_blacklist={options.char_blacklist}")
    return extras or None


def run_ocrmypdf(options: OCRmyPDFOptions) -> None:
    """Invoke ``ocrmypdf.ocr()`` with our fixed 'we-already-preprocessed' settings.

    Args:
        options: Fully populated options.

    Raises:
        OCRmyPDFError: If OCRmyPDF exits with a non-zero code.
        FileNotFoundError: If the input PDF does not exist.
    """
    if not options.input_file.exists():
        raise FileNotFoundError(f"Input PDF does not exist: {options.input_file}")

    options.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import so unit tests that don't touch OCRmyPDF don't need it installed.
    import ocrmypdf  # noqa: WPS433
    from ocrmypdf.exceptions import ExitCodeException  # noqa: WPS433

    # Build the keyword-arg dict WITHOUT input/output — those have to
    # be passed positionally because OCRmyPDF 17+ renamed the first
    # parameter from ``input_file`` to ``input_file_or_options``.
    # Passing by keyword ties us to a specific version; passing
    # positionally works on every release since 14.x and will keep
    # working when they rename the slot again.
    input_file = str(options.input_file)
    output_file = str(options.output_file)

    kwargs: dict[str, Any] = {
        "language": options.language,
        # Disable OCRmyPDF preprocessing — we've done it ourselves.
        "deskew": False,
        "clean": False,
        "remove_background": False,
        "tesseract_thresholding": 0,
        # Tesseract engine settings.
        "tesseract_oem": options.oem,
        "tesseract_pagesegmode": options.psm,
        "optimize": options.optimize,
        "skip_text": options.skip_text,
        "tesseract_timeout": options.tesseract_timeout,
        "progress_bar": False,
        "use_threads": True,
    }

    tess_config = _build_tesseract_config(options)
    if tess_config is not None:
        kwargs["tesseract_config"] = tess_config

    # Merge any advanced extras (allows callers to pass e.g. ``rotate_pages``).
    for key, value in options.extra.items():
        # ``input_file`` / ``output_file`` in ``extra`` would collide with
        # our positional args — prefer the explicit options fields and
        # drop any override from ``extra`` with a warning, so a stale
        # caller can't produce two conflicting positional-vs-keyword
        # arguments for the same slot.
        if key in ("input_file", "output_file", "input_file_or_options"):
            logger.warning(
                "Ignoring `extra[%s]=%r` — use OCRmyPDFOptions.%s_file instead",
                key, value, "input" if "input" in key else "output",
            )
            continue
        kwargs[key] = value

    if options.progress_bar is not None:
        try:
            options.progress_bar(0, 1)
        except Exception:  # noqa: BLE001 - callback errors must not abort OCR
            logger.debug("progress_bar start-callback raised", exc_info=True)

    logger.info(
        "Running ocrmypdf: in=%s out=%s lang=%s psm=%s oem=%s optimize=%s skip_text=%s",
        options.input_file,
        options.output_file,
        options.language,
        options.psm,
        options.oem,
        options.optimize,
        options.skip_text,
    )

    try:
        ocrmypdf.ocr(input_file, output_file, **kwargs)
    except ExitCodeException as exc:
        exit_code = getattr(exc, "exit_code", None)
        logger.error("ocrmypdf failed with exit_code=%s: %s", exit_code, exc)
        raise OCRmyPDFError(
            f"OCRmyPDF failed (exit_code={exit_code}): {exc}",
            exit_code=exit_code,
        ) from exc
    except FileNotFoundError as exc:
        # OCRmyPDF's graft phase (``_graft._parse_hocr_pages``) stats a
        # per-page ``NNNNNN_ocr_hocr.hocr`` in its scratch dir for every
        # input page. If Tesseract exceeded ``tesseract_timeout`` on a
        # page, OCRmyPDF logs ``took too long to OCR - skipping`` and
        # quietly skips producing the hocr — then crashes here with an
        # opaque ``WinError 2`` that points at a random temp file.
        # Detect that specific shape and map it to a human-readable
        # hint instead of leaking the temp path to the user.
        filename = getattr(exc, "filename", "") or ""
        is_graft_hocr_miss = (
            filename.endswith("_hocr.hocr")
            or filename.endswith("_ocr_hocr.hocr")
            or "ocr_hocr" in filename
        )
        if is_graft_hocr_miss:
            logger.error(
                "OCRmyPDF graft failed: missing per-page HOCR at %r — "
                "Tesseract likely skipped a page after exceeding "
                "tesseract_timeout=%ds",
                filename, options.tesseract_timeout,
            )
            raise OCRmyPDFError(
                "Одна или несколько страниц не были распознаны за отведённое "
                f"время (tesseract_timeout={options.tesseract_timeout} с). "
                "Это типично для сложных сканов при высоком DPI. "
                "Попробуйте уменьшить DPI в профиле (например, 600 → 400) "
                "или увеличить tesseract_timeout в настройках OCR."
            ) from exc
        # Some other missing file (input PDF, Tesseract binary, …) —
        # wrap it uniformly but preserve the original filename for
        # diagnostics.
        logger.exception("ocrmypdf raised FileNotFoundError (not graft-hocr)")
        raise OCRmyPDFError(f"OCRmyPDF failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - wrap for consistent upstream handling
        logger.exception("ocrmypdf raised unexpected exception")
        raise OCRmyPDFError(f"OCRmyPDF failed: {exc}") from exc

    if options.progress_bar is not None:
        try:
            options.progress_bar(1, 1)
        except Exception:  # noqa: BLE001
            logger.debug("progress_bar end-callback raised", exc_info=True)

    logger.info("ocrmypdf completed: %s", options.output_file)
