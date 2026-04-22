"""Tests for the ``ocr-cli parser <command>`` dispatcher.

The dispatcher's contract is simple and stable:

* ``ocr-cli parser`` with no subcommand → print help to stdout, exit 2
  (argparse-style "you forgot something").
* ``ocr-cli parser -h`` / ``--help`` → print help, exit 0.
* ``ocr-cli parser unknown-cmd`` → error on stderr, exit 2.
* ``ocr-cli parser <known>`` → import ``scripts.<module>``, call
  ``main(argv)`` with the trailing args, forward its exit code.
* Import error on the backing module → exit 3 (distinct from
  validation-2 and success-0) with a support-actionable message
  mentioning the missing dependency.

Crucially the dispatch MUST NOT touch the OCR pipeline's imports —
``ocr-cli parser feedback-stats`` on a machine without EasyOCR / torch
installed should still work. We verify that by patching the OCR-path
imports to explode: the dispatcher never reaches them.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from src.cli import _PARSER_SUBCOMMANDS
from src.cli import main as cli_main

# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_registry_contains_expected_subcommands() -> None:
    """Locks the public CLI surface against accidental removals."""
    expected = {"golden", "update-golden", "collect-feedback", "feedback-stats"}
    assert set(_PARSER_SUBCOMMANDS) == expected


def test_registry_modules_are_importable() -> None:
    """Each dispatched module exports a callable ``main(argv)``."""
    import importlib

    for name, (module_name, _desc) in _PARSER_SUBCOMMANDS.items():
        mod = importlib.import_module(module_name)
        sub_main = getattr(mod, "main", None)
        assert callable(sub_main), f"{module_name}.main missing for {name!r}"


# ---------------------------------------------------------------------------
# Help surface
# ---------------------------------------------------------------------------


def test_bare_parser_prints_help_and_exits_2(capsys: pytest.CaptureFixture) -> None:
    """``ocr-cli parser`` without a subcommand — user forgot something."""
    rc = cli_main(["parser"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "Usage: ocr-cli parser" in out
    # Every known subcommand name appears in the help block.
    for name in _PARSER_SUBCOMMANDS:
        assert name in out


def test_parser_help_flag_exits_0(capsys: pytest.CaptureFixture) -> None:
    """``-h`` / ``--help`` is a user request, not an error."""
    for flag in ("-h", "--help"):
        rc = cli_main(["parser", flag])
        out = capsys.readouterr().out
        assert rc == 0, flag
        assert "Usage: ocr-cli parser" in out


def test_unknown_subcommand_exits_2(capsys: pytest.CaptureFixture) -> None:
    """Unknown subcommand → stderr error + help, exit 2."""
    rc = cli_main(["parser", "frobnicate"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Неизвестная подкоманда" in captured.err
    assert "Usage: ocr-cli parser" in captured.out


# ---------------------------------------------------------------------------
# Dispatch: every known subcommand forwards argv + return code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subcmd,module",
    [
        ("golden", "scripts.run_golden"),
        ("update-golden", "scripts.update_golden"),
        ("collect-feedback", "scripts.collect_feedback"),
        ("feedback-stats", "scripts.feedback_stats"),
    ],
)
def test_dispatch_forwards_argv_and_return_code(
    subcmd: str, module: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each subcommand imports its module and passes through argv + rc."""
    fake = MagicMock(return_value=42)
    # Replace the backing module's ``main`` in sys.modules so
    # importlib.import_module(module) returns a stub with our ``main``.
    stub = MagicMock(spec=["main"])
    stub.main = fake
    monkeypatch.setitem(sys.modules, module, stub)

    rc = cli_main(["parser", subcmd, "--flag", "value", "positional"])
    assert rc == 42
    fake.assert_called_once_with(["--flag", "value", "positional"])


def test_dispatch_propagates_zero_rc(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subcommand that returns 0 makes the whole CLI return 0."""
    stub = MagicMock(spec=["main"])
    stub.main = MagicMock(return_value=0)
    monkeypatch.setitem(sys.modules, "scripts.feedback_stats", stub)

    assert cli_main(["parser", "feedback-stats"]) == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_import_error_returns_3(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """Backing module can't be imported → exit 3 with actionable hint."""
    import importlib

    original_import = importlib.import_module

    def boom(name: str, *args, **kwargs):
        if name == "scripts.run_golden":
            raise ImportError("No module named 'openpyxl'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", boom)
    rc = cli_main(["parser", "golden"])
    err = capsys.readouterr().err
    assert rc == 3
    assert "golden" in err
    assert "openpyxl" in err


def test_module_without_main_returns_3(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """Backing module loaded but has no ``main`` → exit 3."""
    bad_module = MagicMock(spec=[])  # nothing exported
    # Explicitly set ``main`` to None so getattr(..., "main", None) hits.
    bad_module.main = None
    monkeypatch.setitem(sys.modules, "scripts.run_golden", bad_module)

    rc = cli_main(["parser", "golden"])
    err = capsys.readouterr().err
    assert rc == 3
    assert "main" in err


# ---------------------------------------------------------------------------
# Pipeline isolation: OCR deps not touched by parser subcommand
# ---------------------------------------------------------------------------


def test_parser_subcommand_does_not_import_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ocr-cli parser feedback-stats`` must not pull in easyocr / torch / cv2.

    Guarantees that a Windows dev without the OCR engine installed
    can still run the parser-dev subcommands from a fresh checkout.
    We prove it by blocking imports of pipeline-adjacent modules via
    ``sys.meta_path`` and verifying the dispatch still returns zero.
    """
    import importlib.abc
    import importlib.machinery

    blocked_prefixes = (
        "src.application.pipeline",
        "src.application.engines",
        "src.core.image_preprocessor",
        "easyocr",
        "torch",
    )

    class _BlockFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):  # noqa: ANN001
            if any(name == p or name.startswith(p + ".") for p in blocked_prefixes):
                raise RuntimeError(
                    f"pipeline module {name!r} imported by parser subcommand",
                )
            return None

    # Purge any already-imported pipeline modules so the finder has
    # a fresh chance to block on re-import during the subcommand run.
    for mod_name in list(sys.modules):
        if any(
            mod_name == p or mod_name.startswith(p + ".")
            for p in blocked_prefixes
        ):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    finder = _BlockFinder()
    sys.meta_path.insert(0, finder)
    try:
        # A minimal stub for the subcommand's backing module.
        stub = MagicMock(spec=["main"])
        stub.main = MagicMock(return_value=0)
        monkeypatch.setitem(sys.modules, "scripts.feedback_stats", stub)

        rc = cli_main(["parser", "feedback-stats"])
    finally:
        sys.meta_path.remove(finder)

    assert rc == 0
