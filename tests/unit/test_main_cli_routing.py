"""Tests for ``src.main._should_route_to_cli``.

The GUI launcher and the headless CLI share a single
``OCRStudio.exe`` binary. Routing between them happens in
:func:`src.main.main` via :func:`_should_route_to_cli`, which
scans ``sys.argv`` for a curated set of CLI-only sentinels.

A missing sentinel doesn't crash anything — it silently opens the
GUI when the user expected CLI. That's the exact failure mode
this test locks down: every flag the argparse parser recognises
AND the ``parser`` subcommand MUST trigger the CLI branch, so a
user typing ``OCRStudio.exe input.pdf --excel`` gets an xlsx, not
a GUI window.
"""

from __future__ import annotations

import pytest

from src.main import _CLI_FLAGS, _should_route_to_cli

# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv_tail",
    [
        ["--cli"],
        ["--excel"],
        ["--txt"],
        ["--docx"],
        ["--profile", "tn_upd"],
        ["-p", "default"],
        ["-o", "out.pdf"],
        ["--output", "out.pdf"],
        ["--workers", "4"],
        ["--list-profiles"],
        ["--check-engine", "tesseract"],
        ["-v"],
        ["-vv"],
        ["-vvv"],
        ["--verbose"],
        ["--version"],
        ["-h"],
        ["--help"],
        ["parser"],
        ["parser", "golden"],
        ["parser", "feedback-stats"],
        # Mixed positional + CLI flag — the flag still triggers CLI.
        ["input.pdf", "--excel"],
        ["input.pdf", "--profile", "tn_upd", "--excel"],
    ],
)
def test_route_to_cli_fires_for_known_sentinels(argv_tail: list[str]) -> None:
    """Every CLI flag + the ``parser`` subcommand must switch to CLI."""
    # argv[0] is the program name — the real launcher sees it but
    # _should_route_to_cli ignores it, so any placeholder works.
    argv = ["OCRStudio.exe"] + argv_tail
    assert _should_route_to_cli(argv) is True, argv


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv_tail",
    [
        [],                              # bare double-click
        ["input.pdf"],                   # file association (open-on-launch)
        ["input.pdf", "out.pdf"],        # two positional paths, no flags
        ["C:\\Users\\test\\scan.pdf"],   # Windows-style path
    ],
)
def test_route_stays_on_gui_without_cli_sentinels(argv_tail: list[str]) -> None:
    """Bare paths / zero args keep the GUI path. File association relies on this."""
    argv = ["OCRStudio.exe"] + argv_tail
    assert _should_route_to_cli(argv) is False, argv


# ---------------------------------------------------------------------------
# Coverage sanity: the set MUST include every sentinel the argparse sees
# ---------------------------------------------------------------------------


def test_excel_flag_present_in_routing_set() -> None:
    """Regression guard: PR #9 added ``--excel`` but forgot to wire it here."""
    assert "--excel" in _CLI_FLAGS


def test_parser_subcommand_present_in_routing_set() -> None:
    """Regression guard: PR #6 added ``ocr-cli parser`` dispatcher."""
    assert "parser" in _CLI_FLAGS


def test_verbose_short_forms_present() -> None:
    """``-v`` / ``-vv`` / ``-vvv`` are common CLI invocations."""
    for flag in ("-v", "-vv", "-vvv", "--verbose"):
        assert flag in _CLI_FLAGS, flag
