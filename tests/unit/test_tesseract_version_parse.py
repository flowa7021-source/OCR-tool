"""Regression guard for Tesseract ``--version`` output parsing.

End-user logs showed "Tesseract version detected: <unknown>" even on
freshly-packaged Windows installs with bundled Tesseract 5.5.0. Root
cause: the UB Mannheim Windows build emits ``tesseract v5.5.0.20241111``
(with a ``v`` prefix), and the regex only accepted ``tesseract 5.5.0``
(Linux/apt format). The resulting empty version string:

  * Makes ``verify()`` skip the "major.minor matches expected" check.
  * Fills the status bar with "Версия: <не определено>" — alarming but
    not functionally broken.

This test asserts every documented ``--version`` output variant
parses to the underlying x.y.z string.
"""

from __future__ import annotations

import pytest

from src.infrastructure.tesseract_wrapper import _VERSION_RE


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Linux apt / macOS brew:
        ("tesseract 5.3.4", "5.3.4"),
        ("tesseract 4.1.1", "4.1.1"),
        # UB Mannheim Windows build — the original production failure:
        ("tesseract v5.5.0.20241111", "5.5.0.20241111"),
        ("tesseract v5.4.1.20240606", "5.4.1.20240606"),
        # Multi-line output (we search only the first non-empty line):
        ("tesseract v5.5.0.20241111\n leptonica-1.82.0", "5.5.0.20241111"),
        # Case insensitivity — matches the IGNORECASE flag on the regex.
        ("Tesseract 5.3.4", "5.3.4"),
        ("TESSERACT V5.5.0", "5.5.0"),
    ],
)
def test_version_regex_parses_known_formats(raw: str, expected: str) -> None:
    first_line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    match = _VERSION_RE.search(first_line)
    assert match is not None, (
        f"Regex failed to match the first line of:\n{raw!r}\n"
        "The UB Mannheim 'tesseract v5.5.0.20241111' case is the one "
        "the production parser missed — update the regex if this test "
        "fails on that line specifically."
    )
    assert match.group(1) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "no tesseract here",
        "leptonica-1.82.0",
        "tesseract  ",  # no version number at all
    ],
)
def test_version_regex_rejects_garbage(raw: str) -> None:
    first_line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    assert _VERSION_RE.search(first_line) is None
