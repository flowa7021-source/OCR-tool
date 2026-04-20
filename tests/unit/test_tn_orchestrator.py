"""Unit tests for :mod:`src.application.parsers.tn_orchestrator`.

The orchestrator is the *only* entry point the OCR pipeline uses to
reach the parser, so every branch here maps directly to a pipeline-
level guarantee:

* ``enabled=False`` → parser is not invoked and ``None`` returned.
* ``enabled=True`` + unknown ``kind`` → graceful skip, log a warning.
* Text below ``low_text_threshold`` → skip (keeps the parser off
  blank / cover pages).
* Parser succeeds → ``ParsedDocument`` with one row per document.
* Parser raises → caught, logged, returns ``None`` (OCR job still
  completes).
* LLM fallback + missing ``anthropic`` → silent degrade.

We feed the orchestrator the text of a real golden fixture
(``tn_ocr_tabular.txt``) instead of a hand-crafted minimal string —
a contract test that breaks the moment the real parser regresses on
a known-good waybill.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.application.parsers.tn_orchestrator import extract_from_pages
from src.core.models import (
    ExtractConfig,
    LlmFallbackConfig,
    PageResult,
    ParsedDocument,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "parsers" / "tn" / "fixtures"
_TABULAR_FIXTURE = _FIXTURES / "tn_ocr_tabular.txt"


def _real_pages() -> list[PageResult]:
    """Load a real OCR text fixture and wrap it as a single PageResult."""
    text = _TABULAR_FIXTURE.read_text(encoding="utf-8")
    return [PageResult(page_number=1, text=text, mean_confidence=90.0)]


# ---------------------------------------------------------------------------
# Gate behaviour: short-circuits without touching the parser
# ---------------------------------------------------------------------------


def test_disabled_config_returns_none() -> None:
    """``enabled=False`` must short-circuit — no parser import, no work."""
    cfg = ExtractConfig(enabled=False)
    assert extract_from_pages(_real_pages(), cfg, Path("x.pdf")) is None


def test_unknown_kind_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Future parser kinds must fail open, not raise."""
    cfg = ExtractConfig(enabled=True, kind="invoice_unknown_2099")
    with caplog.at_level(logging.WARNING):
        result = extract_from_pages(_real_pages(), cfg, Path("x.pdf"))
    assert result is None
    assert any("unknown kind" in rec.message.lower() for rec in caplog.records)


def test_text_below_low_text_threshold_returns_none() -> None:
    """Short text (cover pages, stamps-only) skips parsing."""
    cfg = ExtractConfig(enabled=True, low_text_threshold=200)
    short_pages = [PageResult(page_number=1, text="Всего пара слов.")]
    assert extract_from_pages(short_pages, cfg, Path("x.pdf")) is None


def test_all_empty_pages_return_none() -> None:
    """Every page errored / empty → join is empty → skip."""
    cfg = ExtractConfig(enabled=True)
    empty = [PageResult(page_number=i + 1, text="") for i in range(3)]
    assert extract_from_pages(empty, cfg, Path("x.pdf")) is None


def test_page_with_none_text_does_not_crash() -> None:
    """Defensive: a PageResult where text defaulted to '' via dataclass."""
    cfg = ExtractConfig(enabled=True)
    # dataclass default is "" — explicitly verify that path works.
    pages = [PageResult(page_number=1)]
    assert extract_from_pages(pages, cfg, Path("x.pdf")) is None


# ---------------------------------------------------------------------------
# Happy path: real fixture produces rows
# ---------------------------------------------------------------------------


def test_real_fixture_produces_parsed_document() -> None:
    """Feed a known-good waybill text; expect one row with real values."""
    cfg = ExtractConfig(enabled=True)
    result = extract_from_pages(_real_pages(), cfg, Path("tn_ocr_tabular.pdf"))
    assert isinstance(result, ParsedDocument)
    assert len(result.rows) == 1
    row = result.rows[0]
    # Contract from tests/parsers/tn/test_parsing.py: number extracted,
    # date known, shipper includes ИНН. The orchestrator just pipes to
    # the parser — if these change, the parser-level test catches it
    # first. This is a last-line gate that the *wiring* didn't lose data.
    assert row["number"] == "7145/Б"
    assert row["date"] == "23.07.2022"
    assert "Бекам" in row["shipper"]
    assert 0.0 <= result.overall_confidence <= 1.0
    assert result.overall_confidence > 0.5, (
        f"expected non-trivial confidence, got {result.overall_confidence}"
    )


def test_multi_page_joins_with_formfeed() -> None:
    """Two pages of the same text join with \\f for the parser's splitter."""
    text = _TABULAR_FIXTURE.read_text(encoding="utf-8")
    cfg = ExtractConfig(enabled=True)
    pages = [
        PageResult(page_number=1, text=text),
        PageResult(page_number=2, text=text),
    ]
    result = extract_from_pages(pages, cfg, Path("dup.pdf"))
    assert result is not None
    # The parser's splitter detects two distinct documents because \f
    # was preserved. Both will carry the same number; we just assert
    # that the splitter actually saw the boundary.
    assert len(result.rows) == 2, (
        f"expected 2 rows (one per \\f-separated page), got {len(result.rows)}"
    )


# ---------------------------------------------------------------------------
# Error handling: exceptions stay caught
# ---------------------------------------------------------------------------


def test_parser_exception_is_caught(monkeypatch: pytest.MonkeyPatch,
                                     caplog: pytest.LogCaptureFixture) -> None:
    """A parser regression must NOT sink the OCR job."""
    def boom(*_a, **_kw):
        raise RuntimeError("simulated parser bug")

    # Patch the parser's parse_text where the orchestrator imports it.
    import src.tn_parser.core as tn_core
    monkeypatch.setattr(tn_core, "parse_text", boom)

    cfg = ExtractConfig(enabled=True)
    with caplog.at_level(logging.ERROR):
        result = extract_from_pages(_real_pages(), cfg, Path("x.pdf"))
    assert result is None
    assert any("parser raised" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# LLM fallback: offline-safe degradation
# ---------------------------------------------------------------------------


def test_llm_fallback_enabled_without_anthropic_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """enabled=True but anthropic missing → silent no-op, still returns rows.

    Simulates an air-gapped Windows install that carries
    ``extract.enabled=True`` in the profile but has no
    ``anthropic`` wheel in its bundle. The orchestrator must still
    produce rows from the regex parser.
    """
    import sys

    # Make the fallback import path raise ImportError even if the
    # module is installed on this machine.
    monkeypatch.setitem(sys.modules, "src.tn_parser.llm_fallback", None)

    cfg = ExtractConfig(
        enabled=True,
        llm_fallback=LlmFallbackConfig(enabled=True, min_confidence=0.99),
    )
    with caplog.at_level(logging.DEBUG):
        result = extract_from_pages(
            _real_pages(), cfg, Path("tn_ocr_tabular.pdf"),
        )
    assert result is not None
    assert result.rows, "regex path must still produce rows"
