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
from src.shared.constants import TESSDATA_DIR

logger = logging.getLogger(__name__)


def _resolve_user_dict_paths(primary_language: str) -> tuple[Path | None, Path | None]:
    """Return paths to bundled user-words / user-patterns for ``primary_language``.

    Looks under the module-level :data:`TESSDATA_DIR` (which defaults to
    ``resources/tessdata`` relative to the app root and is monkey-patched
    by tests). If either file is missing, logs a WARNING and returns
    ``None`` for that slot so the caller can skip the corresponding
    ``user_words=`` / ``user_patterns=`` kwarg instead of crashing.

    OCRmyPDF / Tesseract accept exactly one of each per run, so we
    key off ``primary_language`` rather than the full language list —
    picking the pair that matches what the document mostly contains.

    Args:
        primary_language: ISO 639-3 Tesseract language code, e.g.
            ``"rus"`` or ``"eng"``. Anything outside our bundled set
            results in ``(None, None)`` with a debug-level log (not a
            warning — it's a supported no-op, not a misconfiguration).

    Returns:
        ``(user_words_path, user_patterns_path)``. Either element may
        be ``None`` if the file is missing.
    """
    words_path = TESSDATA_DIR / f"user-words.{primary_language}"
    patterns_path = TESSDATA_DIR / f"user-patterns.{primary_language}"

    resolved_words: Path | None = words_path if words_path.is_file() else None
    resolved_patterns: Path | None = (
        patterns_path if patterns_path.is_file() else None
    )

    if resolved_words is None:
        logger.warning(
            "user-words file missing at %s — Tesseract will run without "
            "the custom dictionary for language %r. Accuracy on ИНН / "
            "КПП / ОГРН / month names may suffer. Check that the "
            "installer bundled resources/tessdata/user-words.%s.",
            words_path, primary_language, primary_language,
        )
    if resolved_patterns is None:
        logger.warning(
            "user-patterns file missing at %s — Tesseract will run "
            "without custom regex patterns for language %r. Accuracy on "
            "dates and tax IDs may suffer. Check that the installer "
            "bundled resources/tessdata/user-patterns.%s.",
            patterns_path, primary_language, primary_language,
        )
    return resolved_words, resolved_patterns


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
    #: Resolved paths to the bundled user-words / user-patterns files
    #: for ``primary_language``. ``None`` means "do not forward the
    #: corresponding kwarg to ocrmypdf.ocr" — either the user disabled
    #: ``OCRConfig.use_user_dictionaries`` or the file is missing from
    #: the bundle (graceful-degradation path, logged at map time).
    user_words: Path | None = None
    user_patterns: Path | None = None


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
    # Resolve user-dict paths once up front so any "file missing"
    # warning lands in the logs at pipeline-setup time, not deep
    # inside the OCR call where it's harder to associate with the
    # triggering job.
    user_words: Path | None = None
    user_patterns: Path | None = None
    if cfg.use_user_dictionaries:
        user_words, user_patterns = _resolve_user_dict_paths(cfg.primary_language)

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
        user_words=user_words,
        user_patterns=user_patterns,
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


# Cap on the escalated retry timeout. We multiply the base by 3 and
# clamp at this value so a pathologically stuck page cannot tie up a
# worker for an unbounded duration. 15 minutes is the longest the QA
# matrix has ever observed a *legitimate* 600 DPI Russian contract page
# taking to OCR on mid-tier hardware — anything beyond that is almost
# certainly a leptonica/tesseract bug that a further wait won't fix.
_MAX_RETRY_TESSERACT_TIMEOUT_SEC: int = 15 * 60


def _is_graft_hocr_miss(exc: BaseException) -> bool:
    """Return True iff ``exc`` is the ``_graft._parse_hocr_pages`` crash.

    OCRmyPDF stats a per-page ``NNNNNN_ocr_hocr.hocr`` in its scratch
    dir for every input page during its graft phase. If Tesseract
    exceeded ``tesseract_timeout`` on a page, OCRmyPDF logs
    ``took too long to OCR - skipping`` and quietly skips producing
    the hocr — then crashes in graft with an opaque ``FileNotFoundError``
    that points at a random temp file the user cannot act on.
    Detecting this specific shape lets us retry the job with a longer
    timeout instead of bubbling the raw error straight to the UI.
    """
    if not isinstance(exc, FileNotFoundError):
        return False
    filename = getattr(exc, "filename", "") or ""
    return (
        filename.endswith("_hocr.hocr")
        or filename.endswith("_ocr_hocr.hocr")
        or "ocr_hocr" in filename
    )


def _output_pdf_has_any_text(pdf_path: str) -> bool:
    """Return True if ANY page in ``pdf_path`` has non-whitespace text.

    ``ocrmypdf.ocr`` returns COMPLETED even when Tesseract silently
    skipped one or more pages for timing out — it writes an empty
    hOCR file and grafts it, so the output PDF "has a text layer"
    structurally but that layer contains zero recognised characters.
    This helper is the check we use to decide whether to retry.
    """
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            for page in doc:
                if (page.get_text("text") or "").strip():
                    return True
    except Exception as exc:  # noqa: BLE001
        # Can't open the output. Let the caller treat this as "no text"
        # so a retry kicks in; if the retry also hits this path we'll
        # log and bail.
        logger.debug("_output_pdf_has_any_text: fitz failed: %s", exc)
        return False
    return False


def _invoke_ocrmypdf_with_timeout_retry(
    ocrmypdf: Any,
    exit_code_exception: type[BaseException],
    input_file: str,
    output_file: str,
    kwargs: dict[str, Any],
    *,
    base_timeout: int,
) -> None:
    """Run ``ocrmypdf.ocr`` once, retrying once when Tesseract timed out.

    Tesseract timeouts surface in TWO distinct shapes and both need to
    be handled:

      1. **Graft FileNotFoundError** — OCRmyPDF aborts mid-pipeline
         because ``_parse_hocr_pages`` stats a ``*_ocr_hocr.hocr``
         that doesn't exist on disk. The job status is FAILED.

      2. **Empty-hOCR silent skip** — OCRmyPDF writes a valid but
         empty hOCR for the timed-out page, grafts it, and returns
         a COMPLETED output PDF with a text layer that contains no
         recognised characters. The job looks successful; the user
         sees empty text and can't tell it was a timeout.

    Both shapes are fixed by the same escalation: retry with
    ``tesseract_timeout = min(base * 3, _MAX_RETRY_TESSERACT_TIMEOUT_SEC)``
    — generally 3× the user's setting, floor 10 min, cap 15 min.

    We do NOT change ``use_threads`` on the retry: OCRmyPDF 16.x
    has an upstream bug where ``use_threads=False`` crashes with
    ``AttributeError: 'OcrOptions' object has no attribute
    'tesseract'``.

    Any other exception (``ExitCodeException``, non-graft
    FileNotFoundError) propagates unchanged to the caller.
    """
    retry_reason: str | None = None
    try:
        ocrmypdf.ocr(input_file, output_file, **kwargs)
    except exit_code_exception:
        raise
    except FileNotFoundError as exc:
        if not _is_graft_hocr_miss(exc):
            raise
        retry_reason = f"graft-hocr-miss ({exc})"
    else:
        # First attempt "succeeded" structurally — but check the
        # output actually has text. An all-empty output means
        # Tesseract silently skipped every page it was asked to OCR.
        if _output_pdf_has_any_text(output_file):
            return
        retry_reason = "all pages empty in output PDF (silent tesseract skip)"

    # First attempt produced an unusable result — escalate and retry.
    retry_timeout = min(
        max(base_timeout * 3, 600),
        _MAX_RETRY_TESSERACT_TIMEOUT_SEC,
    )
    retry_kwargs = dict(kwargs)
    retry_kwargs["tesseract_timeout"] = retry_timeout
    # We used to also set ``use_threads=False`` to reduce CPU contention,
    # but OCRmyPDF 16.x has an upstream bug where ``use_threads=False``
    # crashes with ``AttributeError: 'OcrOptions' object has no attribute
    # 'tesseract'`` inside ``generate_hocr``. Keeping the original
    # ``use_threads`` value (``True``) avoids the crash; the timeout
    # increase alone is sufficient for the retry to succeed on the
    # vast majority of timed-out pages.
    logger.warning(
        "OCRmyPDF first attempt unusable (%s) at tesseract_timeout=%ds — "
        "retrying once with tesseract_timeout=%ds. "
        "If this retry also fails the user will need to lower DPI or "
        "raise tesseract_timeout in the profile.",
        retry_reason, base_timeout, retry_timeout,
    )
    ocrmypdf.ocr(input_file, output_file, **retry_kwargs)


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
        # ``tesseract_thresholding`` in ocrmypdf's typed dict is declared
        # ``int | None``, but since 16.x the API implementation serialises
        # the int back into a CLI arg via ``str(value)`` and re-parses it
        # through argparse with ``choices=('auto','otsu','adaptive-otsu',
        # 'sauvola')``. Passing ``0`` therefore crashes with
        # "'0' must be one of: auto, otsu, adaptive-otsu, sauvola".
        # We pass ``"auto"`` explicitly (which maps back to 0) — that
        # round-trips cleanly through argparse AND matches what ocrmypdf
        # uses by default, so behaviour is unchanged.
        "tesseract_thresholding": "auto",
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

    # User-words / user-patterns forwarding. Only pass each kwarg if we
    # have a resolved path; if the file was missing,
    # ``_resolve_user_dict_paths`` already logged a warning and left
    # the slot ``None`` so we silently skip it here (graceful
    # degradation — the OCR still runs, just without the custom
    # dictionary). Kwargs are serialised as strings because OCRmyPDF
    # hands them straight to Tesseract's CLI.
    if options.user_words is not None:
        kwargs["user_words"] = str(options.user_words)
    if options.user_patterns is not None:
        kwargs["user_patterns"] = str(options.user_patterns)

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
        _invoke_ocrmypdf_with_timeout_retry(
            ocrmypdf,
            ExitCodeException,
            input_file,
            output_file,
            kwargs,
            base_timeout=options.tesseract_timeout,
        )
    except ExitCodeException as exc:
        exit_code = getattr(exc, "exit_code", None)
        logger.error("ocrmypdf failed with exit_code=%s: %s", exit_code, exc)
        raise OCRmyPDFError(
            f"OCRmyPDF failed (exit_code={exit_code}): {exc}",
            exit_code=exit_code,
        ) from exc
    except FileNotFoundError as exc:
        # Anything reaching this arm is either
        #   (a) the graft-hocr-miss that survived the retry, or
        #   (b) a genuinely missing file (input PDF, tesseract binary).
        # ``_invoke_ocrmypdf_with_timeout_retry`` already retried (a)
        # once with a longer timeout + single-threaded execution; if
        # it still came back with a ``*_ocr_hocr.hocr`` filename the
        # user has to intervene (lower DPI or set a larger
        # tesseract_timeout themselves).
        filename = getattr(exc, "filename", "") or ""
        is_graft_hocr_miss = (
            filename.endswith("_hocr.hocr")
            or filename.endswith("_ocr_hocr.hocr")
            or "ocr_hocr" in filename
        )
        if is_graft_hocr_miss:
            logger.error(
                "OCRmyPDF graft failed AFTER retry: missing per-page "
                "HOCR at %r — Tesseract still timed out on a page at "
                "the escalated timeout",
                filename,
            )
            raise OCRmyPDFError(
                "Одна или несколько страниц не были распознаны даже "
                "после автоматического повтора с увеличенным таймаутом. "
                "Это типично для сложных сканов при высоком DPI. "
                "Уменьшите DPI в профиле (например, 600 → 400) или "
                "явно увеличьте tesseract_timeout в настройках OCR."
            ) from exc
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
