"""Static checks on the build-installer workflow.

The full installer build takes ~30 min, so we want cheap pytest-level
tripwires for classes of regressions we've seen burn us in CI:

* The installer-size floor in the integrity-check step must exist and
  be tight enough to catch a degenerate build where Tesseract or
  tessdata got dropped.
* The installer-integrity streaming scan must not OOM on a large
  installer (a prior revision used ``ReadAllBytes`` which blew up
  PowerShell's single-object size limit).
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).parent.parent.parent / ".github" / "workflows" / "build-installer.yml"


def _read() -> str:
    assert WORKFLOW.exists(), f"Missing workflow: {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


class TestInstallerSizeFloor:
    """A bundle with no Tesseract / tessdata would be <20 MB (just
    the Python / Qt shell). The floor must safely reject that while
    leaving comfortable headroom above a real compressed build.
    """

    def test_floor_exists(self) -> None:
        """``$minMB`` assignment must exist in the workflow."""
        content = _read()
        assert re.search(
            r"\$minMB\s*=\s*\d+",
            content,
        ), "The `$minMB = ...` size-floor assignment disappeared."

    def test_floor_rejects_degenerate_bundle(self) -> None:
        """Floor must be above a degenerate bundle but below a real one."""
        content = _read()
        m = re.search(r"\$minMB\s*=\s*(\d+)", content)
        assert m, "Couldn't parse the size floor"
        floor = int(m.group(1))
        # Real Tesseract+tessdata+Python+Qt+Ghostscript compresses to
        # ~90-120 MB; degenerate (no Tesseract) is <20 MB.
        assert 50 <= floor <= 200, (
            f"Size floor {floor} MB is outside the [50, 200] sanity "
            f"range — either too tight (will reject real builds) or "
            f"too loose (won't catch a dropped bundle)."
        )


class TestIntegrityCheckMemoryFriendly:
    """Guard the installer-integrity step against an OOM regression.

    Loading a big installer via ``[System.IO.File]::ReadAllBytes +
    GetString`` throws ``System.OutOfMemoryException`` inside
    PowerShell because the resulting string allocation exceeds the
    runtime's single-object size limit. The step now uses a
    StreamReader + 4 MB chunking, which this test locks in.
    """

    def _integrity_section(self) -> str:
        """Return just the ``Sanity-check installer integrity`` step body.

        We scope all the assertions to this one step so unrelated
        PowerShell blocks elsewhere in the workflow don't false-match.
        """
        content = _read()
        start = content.find("Sanity-check installer integrity")
        assert start >= 0, "integrity step name not found"
        end_marker = "Installer integrity check passed"
        end = content.find(end_marker, start)
        assert end > start, "end of integrity step body not found"
        return content[start:end + len(end_marker)]

    def test_integrity_step_does_not_readallbytes(self) -> None:
        """No actual call to ReadAllBytes — comments mentioning it are fine."""
        code_lines = [
            line for line in self._integrity_section().splitlines()
            # Drop pure-comment lines (ignoring leading indentation).
            if not line.lstrip().startswith("#")
        ]
        code = "\n".join(code_lines)
        assert "ReadAllBytes" not in code, (
            "Installer integrity step calls [System.IO.File]::ReadAllBytes "
            "— OOMs PowerShell on 1 GB+ HTR installers. Use a chunked "
            "StreamReader scan instead."
        )

    def test_integrity_step_streams_with_readerloop(self) -> None:
        section = self._integrity_section()
        # Must use StreamReader + chunk loop.
        assert "StreamReader" in section, (
            "Integrity step must stream the file via StreamReader "
            "to avoid OOM on large installers."
        )
        assert "$sr.Read" in section or "sr.Read(" in section, (
            "Integrity step must call the Read(buffer, 0, size) "
            "overload in a loop to process the file in chunks."
        )
