"""Static checks on installer/setup.iss.

The Inno Setup script only gets exercised by the full Windows build
(~30 min), so syntax errors like "Unknown identifier 'INFOAFTERLABEL'"
slip through until the nightly installer job runs. These tests
enforce a couple of cheap invariants on every CI run:

* No references to :class:`TWizardForm`-only properties
  (``InfoAfterLabel``, ``InfoBeforeLabel``, ``NextButton`` …) on the
  ``UninstallProgressForm`` object — those are compile errors.
* ``InitializeUninstallProgressForm`` and ``CurUninstallStepChanged``
  exist in the ``[Code]`` section (so the data-cleanup checkbox and
  its handler aren't accidentally dropped).
"""

from __future__ import annotations

import re
from pathlib import Path

SETUP_ISS = Path(__file__).parent.parent.parent / "installer" / "setup.iss"


def _read() -> str:
    assert SETUP_ISS.exists(), f"Missing installer script: {SETUP_ISS}"
    return SETUP_ISS.read_text(encoding="utf-8")


class TestUninstallFormPropertyUse:
    # These are TWizardForm-only — referencing them on
    # UninstallProgressForm yields `Unknown identifier` at compile time.
    _WIZARD_ONLY_PROPERTIES = (
        "InfoAfterLabel",
        "InfoBeforeLabel",
        "WelcomeLabel1",
        "WelcomeLabel2",
        "LicenseAcceptedRadio",
        "LicenseNotAcceptedRadio",
        "NextButton",
        "BackButton",
        "DirEdit",
        "GroupEdit",
        "TypesCombo",
        "ComponentsList",
        "TasksList",
        "ReadyMemo",
        "FinishedLabel",
        "FinishedHeadingLabel",
        "RunList",
    )

    def test_no_wizard_only_props_on_uninstall_form(self) -> None:
        content = _read()
        for prop in self._WIZARD_ONLY_PROPERTIES:
            pattern = rf"UninstallProgressForm\s*\.\s*{prop}\b"
            matches = list(re.finditer(pattern, content, flags=re.IGNORECASE))
            assert not matches, (
                f"setup.iss references UninstallProgressForm.{prop} — "
                f"that property only exists on TWizardForm (setup), not "
                f"TUninstallProgressForm. This will fail Inno Setup "
                f"compilation with 'Unknown identifier {prop.upper()}'."
            )


class TestUninstallCodeSectionShape:
    def test_initialize_uninstall_progress_form_exists(self) -> None:
        content = _read()
        assert "procedure InitializeUninstallProgressForm" in content

    def test_cur_uninstall_step_changed_exists(self) -> None:
        content = _read()
        assert "procedure CurUninstallStepChanged" in content

    def test_clean_user_data_checkbox_declared(self) -> None:
        """The data-wipe opt-in must still be present."""
        content = _read()
        assert "CleanUserDataCheckbox" in content
        # Should be bound to the uninstall form's InnerPage.
        assert re.search(
            r"CleanUserDataCheckbox\.Parent\s*:=\s*UninstallProgressForm\.InnerPage",
            content,
        ), "Checkbox must parent to UninstallProgressForm.InnerPage"


class TestScaleDirectiveUsage:
    """Position offsets must go through ScaleX/ScaleY for HiDPI correctness."""

    def test_no_raw_integer_offsets_on_checkbox_positioning(self) -> None:
        content = _read()
        # Extract the InitializeUninstallProgressForm body.
        m = re.search(
            r"procedure\s+InitializeUninstallProgressForm.*?end;",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        assert m, "InitializeUninstallProgressForm not found"
        body = m.group(0)
        # Positioning offsets in pixel space must go through ScaleX() /
        # ScaleY() so the checkbox layout stays sane on HiDPI displays.
        # Catches "+ 16" style regressions where raw integer offsets
        # creep back in.
        assert "ScaleX(" in body or "ScaleY(" in body, (
            "Positioning offsets in InitializeUninstallProgressForm "
            "must use ScaleX()/ScaleY() for HiDPI correctness."
        )
