"""Static checks on the build-installer workflow.

The full installer build takes ~30 min, so we want cheap pytest-level
tripwires for classes of regressions we've seen burn us in CI:

* Installer-size floor in the integrity-check step must be achievable
  under realistic Inno Setup compression. A previous revision set it
  to 600 MB for HTR builds while actual compressed bundles land in
  the 300-450 MB range, so every HTR release failed the sanity check.
* Both the no-HTR and HTR floors must exist (the ternary can't drop
  one branch by accident).
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).parent.parent.parent / ".github" / "workflows" / "build-installer.yml"


def _read() -> str:
    assert WORKFLOW.exists(), f"Missing workflow: {WORKFLOW}"
    return WORKFLOW.read_text(encoding="utf-8")


class TestInstallerSizeFloor:
    # Realistic observed ranges for the lzma2/ultra64-compressed installer.
    # Tighten these only after observing several real CI runs — the whole
    # point is to NOT be a false-positive gate.
    _NO_HTR_OBSERVED_MIN_MB = 80      # no-HTR build floor
    _NO_HTR_OBSERVED_MAX_MB = 120     # no-HTR build ceiling
    # HTR build now bundles ~580 MB of GOT-OCR 2.0 safetensors +
    # torch + transformers. Safetensors don't compress much, so the
    # compressed installer lands at ~700-900 MB.
    _HTR_OBSERVED_MIN_MB = 700        # HTR build floor
    _HTR_OBSERVED_MAX_MB = 1000       # HTR build ceiling

    def test_minmb_ternary_exists(self) -> None:
        """The size check must keep distinct floors for HTR vs no-HTR."""
        content = _read()
        assert re.search(
            r"\$minMB\s*=\s*if\s*\(.+with_htr.+?\)\s*\{\s*\d+\s*\}\s*else\s*\{\s*\d+\s*\}",
            content,
            flags=re.DOTALL,
        ), "The HTR/no-HTR size-floor ternary disappeared or changed shape."

    def test_htr_floor_is_achievable(self) -> None:
        """HTR floor must leave headroom below observed real build size."""
        content = _read()
        m = re.search(
            r"\$minMB\s*=\s*if\s*\(.+with_htr.+?\)\s*\{\s*(\d+)\s*\}\s*else\s*\{\s*(\d+)\s*\}",
            content,
            flags=re.DOTALL,
        )
        assert m, "Couldn't parse the HTR floor"
        htr_floor = int(m.group(1))
        assert htr_floor <= self._HTR_OBSERVED_MIN_MB, (
            f"HTR installer floor is {htr_floor} MB but actual compressed "
            f"HTR builds land around {self._HTR_OBSERVED_MIN_MB}–"
            f"{self._HTR_OBSERVED_MAX_MB} MB. Reduce the floor or the "
            f"integrity check will fail every release."
        )

    def test_no_htr_floor_catches_missing_bundle(self) -> None:
        """No-HTR floor must be above a degenerate bundle (everything dropped)."""
        content = _read()
        m = re.search(
            r"\$minMB\s*=\s*if\s*\(.+with_htr.+?\)\s*\{\s*\d+\s*\}\s*else\s*\{\s*(\d+)\s*\}",
            content,
            flags=re.DOTALL,
        )
        assert m, "Couldn't parse the no-HTR floor"
        no_htr_floor = int(m.group(1))
        # A bundle with no Tesseract / tessdata would be <20 MB (just
        # the Python / Qt shell). The floor must safely reject that.
        assert no_htr_floor >= 50, (
            f"No-HTR floor {no_htr_floor} MB is too low — a degenerate "
            f"build without Tesseract/tessdata (<20 MB) would sneak through."
        )

    def test_htr_floor_is_above_no_htr_floor(self) -> None:
        """HTR floor must be strictly higher than the no-HTR floor."""
        content = _read()
        m = re.search(
            r"\$minMB\s*=\s*if\s*\(.+with_htr.+?\)\s*\{\s*(\d+)\s*\}\s*else\s*\{\s*(\d+)\s*\}",
            content,
            flags=re.DOTALL,
        )
        assert m
        htr, no_htr = int(m.group(1)), int(m.group(2))
        assert htr > no_htr, (
            f"HTR floor ({htr} MB) must be > no-HTR floor ({no_htr} MB) — "
            f"otherwise a silently-dropped HTR bundle wouldn't be caught."
        )
