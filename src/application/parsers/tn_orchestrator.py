"""Orchestrator for the ТН / УПД parser (``src.tn_parser``).

Sits between :class:`~src.application.pipeline.OCRPipeline` and the
pure-Python :mod:`src.tn_parser` package. Responsibilities:

* Gate on :class:`~src.core.models.ExtractConfig.enabled` and
  :attr:`~src.core.models.ExtractConfig.kind` so the pipeline can
  call us unconditionally.
* Assemble per-page OCR text into the ``\\f``-separated form the
  parser's ``split_documents`` stage expects.
* Import ``src.tn_parser`` lazily — the package pulls in PyMuPDF /
  rapidfuzz at import time, costly for jobs where extraction is off.
* Catch every parser exception and log it; a regression in
  ``fields.py`` or an unreadable ``user-words.rus`` catalog must not
  sink an otherwise-successful OCR job.
* Optionally invoke the LLM fallback for low-confidence rows,
  silently degrading to the regex-only path when the ``anthropic``
  package or API key is absent (CLAUDE.md: offline-first).

Returns :class:`~src.core.models.ParsedDocument` or ``None``. The
caller (``OCRPipeline.run``) stores the result on
:attr:`JobResult.parsed` and moves on.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.models import ExtractConfig, PageResult, ParsedDocument

logger = logging.getLogger(__name__)


# Dispatcher key for the only parser implemented at this stage. Future
# kinds (``"invoice"``, ``"contract"``) register themselves by adding
# another branch below without touching the pipeline.
_KIND_TN_UPD = "tn_upd"


def extract_from_pages(
    pages: list[PageResult],
    config: ExtractConfig,
    source_path: Path,
) -> ParsedDocument | None:
    """Run the post-OCR parser on a job's per-page text.

    The contract is "never raise, never re-run OCR":

    * Returns ``None`` when ``config.enabled`` is false, when the
      ``kind`` dispatcher doesn't match a known parser, when the
      joined text is shorter than ``config.low_text_threshold``, when
      the parser produced no rows, or when an internal error was
      caught and swallowed. The pipeline treats every ``None`` as
      "no structured data, carry on" and still reports the OCR job
      as COMPLETED.
    * Never raises. Every :class:`Exception` raised by the parser is
      logged at ``exception`` level and converted into ``None``.

    Args:
        pages: Per-page OCR output from the pipeline. The caller is
            expected to have populated :attr:`PageResult.text`
            already (post-processing done). Empty / error pages
            contribute an empty string to the joined text.
        config: The profile's :class:`ExtractConfig` sub-section.
            ``enabled=False`` short-circuits immediately.
        source_path: Path to the source PDF. Used purely for logging
            and the ``source`` field of the emitted rows — not
            re-opened by the orchestrator (the parser works on the
            text we assemble, not on the PDF).

    Returns:
        A :class:`ParsedDocument` with at least one row on success,
        else ``None``.
    """
    if not config.enabled:
        return None

    if config.kind != _KIND_TN_UPD:
        logger.warning(
            "Post-OCR extraction skipped: unknown kind %r "
            "(supported: %r)",
            config.kind,
            _KIND_TN_UPD,
        )
        return None

    # Assemble the parser input. ``src.tn_parser.parse_text`` treats
    # ``\f`` as a page separator (its ``splitter.split_documents``
    # uses page-aware classification for сводные УПД + реестр + ТН
    # packs); preserving that boundary here is how the parser tells
    # which page a row came from.
    #
    # Missing or errored pages contribute an empty string, NOT a
    # literal ``None`` — otherwise ``"\f".join(...)`` would raise
    # ``TypeError`` and the orchestrator's exception handler would
    # swallow an otherwise-normal job. The empty string is a no-op
    # for the parser (classified as ``blank`` by ``_classify_page``).
    raw_text = "\f".join((p.text or "") for p in pages)

    stripped_len = len(raw_text.strip())
    if stripped_len < config.low_text_threshold:
        logger.info(
            "Post-OCR extraction skipped for %s: %d chars < "
            "low_text_threshold=%d",
            source_path.name,
            stripped_len,
            config.low_text_threshold,
        )
        return None

    try:
        from src.tn_parser.core import parse_text
        from src.tn_parser.normalize import normalize_for_sections
    except ImportError as exc:
        # rapidfuzz / openpyxl / pymupdf missing from the frozen
        # build — catch here rather than at the pipeline level so
        # the OCR job completes with ``parsed=None`` instead of
        # FAILED.
        logger.error(
            "Post-OCR extraction unavailable: %s. Install the parser "
            "dependencies (openpyxl, rapidfuzz) or disable "
            "extract.enabled in the profile.",
            exc,
        )
        return None

    try:
        normalized = normalize_for_sections(raw_text)
        rows = parse_text(normalized, str(source_path))
    except Exception as exc:  # noqa: BLE001 — parser bugs must not sink the job
        logger.exception(
            "Post-OCR parser raised on %s: %s",
            source_path.name,
            exc,
        )
        return None

    if not rows:
        logger.info(
            "Post-OCR extraction produced no rows for %s",
            source_path.name,
        )
        return None

    if config.llm_fallback.enabled:
        _maybe_apply_llm_fallback(rows, normalized, config.llm_fallback)

    overall = (
        sum(r.confidence.overall() for r in rows) / len(rows)
        if rows
        else 0.0
    )
    logger.info(
        "Post-OCR extraction: %d row(s) for %s (mean conf=%.2f)",
        len(rows),
        source_path.name,
        overall,
    )
    return ParsedDocument(
        rows=[r.to_json_dict() for r in rows],
        overall_confidence=round(overall, 3),
    )


def _maybe_apply_llm_fallback(rows: list, normalized_text: str, llm_cfg) -> None:
    """Best-effort LLM rescue for rows below ``min_confidence``.

    Silent no-op when:
      * the ``anthropic`` package is absent (offline-first);
      * ``ANTHROPIC_API_KEY`` is unset;
      * any API call raises (network down, rate limit, malformed
        response, pydantic validation error, etc.).

    Mutates each qualifying row in-place — ``improve_row`` only
    fills fields that were MISSING / GARBAGE, so a row the regex
    parser got right survives unchanged even if the LLM
    hallucinates a different value.
    """
    try:
        from src.tn_parser.llm_fallback import improve_row
    except ImportError:
        logger.debug(
            "LLM fallback requested but anthropic/tn_parser.llm_fallback "
            "is unavailable; skipping (offline-safe no-op)",
        )
        return

    for row in rows:
        if row.confidence.overall() >= llm_cfg.min_confidence:
            continue
        try:
            improve_row(
                row,
                normalized_text,
                model=llm_cfg.model,
                max_text_chars=llm_cfg.max_text_chars,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM fallback failed for a row: %s", exc)
