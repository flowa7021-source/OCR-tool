"""Unit tests for ``OCRPipeline._maybe_run_parser``.

We target the helper directly rather than running the full ``run()``
pipeline — the hook's contract is narrow and doesn't depend on any
of the preprocessor / engine / postprocessor machinery. An
``OCRPipeline`` instance is created via ``__new__`` so the test
doesn't need Tesseract / OCRmyPDF / a real workdir.

Three invariants are exercised:

1. When ``profile.extract.enabled=False`` the hook must leave
   ``result.parsed`` as ``None`` AND not import the orchestrator
   (startup cost matters on the UI thread).
2. When ``enabled=True`` the hook delegates to
   ``extract_from_pages`` and stores the return value verbatim.
3. If the orchestrator raises *despite* its own exception handler
   (defence in depth — a future refactor could regress), the hook
   swallows the exception, keeps ``result.parsed=None``, and the
   job surrounding it is not affected.
"""

from __future__ import annotations

import pytest

# OCRPipeline imports opencv / ocrmypdf / fitz at module scope. On a CI
# leg that deliberately trims those (``syntax`` job, dev laptops
# without opencv-python), this test would noisily error out rather
# than skip. ``importorskip`` keeps the suite green on those legs
# while guaranteeing the test DOES run on the real ``tests`` matrix
# where the full dep set is present.
pytest.importorskip("cv2")
pytest.importorskip("fitz")

from src.application.pipeline import OCRPipeline  # noqa: E402
from src.core.models import (  # noqa: E402
    ExtractConfig,
    JobResult,
    OCRJobConfig,
    PageResult,
    ParsedDocument,
    ProfileData,
)
from src.shared.types import JobStatus  # noqa: E402


def _bare_pipeline() -> OCRPipeline:
    """OCRPipeline without the heavy preprocessor / tesseract wiring.

    Bypasses ``__init__`` so the tests don't need Tesseract binaries
    or a real preprocessor instance. Only attributes the hook itself
    reads (none, today) are populated.
    """
    return OCRPipeline.__new__(OCRPipeline)


def _job_with_extract(enabled: bool) -> OCRJobConfig:
    profile = ProfileData(name="test_profile")
    profile.extract = ExtractConfig(enabled=enabled, kind="tn_upd")
    return OCRJobConfig(
        input_path="/tmp/in.pdf",
        output_path="/tmp/out.pdf",
        profile=profile,
    )


def _result_with_pages(text: str = "hello") -> JobResult:
    return JobResult(
        job_id="j1",
        status=JobStatus.RUNNING,
        input_path="/tmp/in.pdf",
        output_path="/tmp/out.pdf",
        pages=[PageResult(page_number=1, text=text)],
    )


# ---------------------------------------------------------------------------
# Disabled → no-op
# ---------------------------------------------------------------------------


def test_hook_forwards_disabled_config_to_orchestrator(mocker) -> None:
    """The gate lives in the orchestrator; the pipeline forwards unchanged.

    The hook doesn't short-circuit on ``enabled=False`` itself — it
    calls ``extract_from_pages`` unconditionally and lets the
    orchestrator decide (which returns ``None`` on disabled). This
    keeps the "gate lives in one place" invariant.
    """
    spy = mocker.patch(
        "src.application.parsers.tn_orchestrator.extract_from_pages",
        return_value=None,
    )
    pipeline = _bare_pipeline()
    result = _result_with_pages()
    pipeline._maybe_run_parser(result, _job_with_extract(enabled=False))
    assert result.parsed is None
    assert spy.called
    # Verify the hook passes the disabled config through verbatim.
    kwargs = spy.call_args.kwargs
    assert kwargs["config"].enabled is False


# ---------------------------------------------------------------------------
# Enabled → delegates and stores the return value
# ---------------------------------------------------------------------------


def test_hook_stores_orchestrator_result_verbatim(mocker) -> None:
    """The hook must not reinterpret the orchestrator's ParsedDocument."""
    fake_doc = ParsedDocument(
        rows=[{"number": "ТН-001", "date": "01.01.2024"}],
        overall_confidence=0.73,
    )
    mocker.patch(
        "src.application.parsers.tn_orchestrator.extract_from_pages",
        return_value=fake_doc,
    )
    pipeline = _bare_pipeline()
    result = _result_with_pages()
    pipeline._maybe_run_parser(result, _job_with_extract(enabled=True))
    assert result.parsed is fake_doc
    assert result.parsed.rows[0]["number"] == "ТН-001"
    assert result.parsed.overall_confidence == 0.73


# ---------------------------------------------------------------------------
# Defence in depth: orchestrator raises → hook swallows
# ---------------------------------------------------------------------------


def test_hook_swallows_orchestrator_exception(
    mocker, caplog: pytest.LogCaptureFixture,
) -> None:
    """Future regression: orchestrator raises → job still completes."""
    import logging

    mocker.patch(
        "src.application.parsers.tn_orchestrator.extract_from_pages",
        side_effect=RuntimeError("bad parser upgrade"),
    )
    pipeline = _bare_pipeline()
    result = _result_with_pages()
    with caplog.at_level(logging.ERROR):
        pipeline._maybe_run_parser(result, _job_with_extract(enabled=True))
    assert result.parsed is None
    assert any("parser hook" in rec.message.lower() for rec in caplog.records)


# ---------------------------------------------------------------------------
# JobResult default: ``parsed`` is None until the hook runs
# ---------------------------------------------------------------------------


def test_job_result_parsed_defaults_to_none() -> None:
    """Backwards-compat: legacy callers that build JobResult directly."""
    result = _result_with_pages()
    assert result.parsed is None
