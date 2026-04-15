"""Static lint: bash heredocs inside PowerShell workflow steps.

We've repeatedly lost ~15 minutes of CI wall-clock to the same failure:

    ParserError: Missing file specification after redirection operator.

    python - <<'EOF'
              ~

GitHub Actions steps run under whichever shell the `shell:` key names.
A ``python - <<'EOF'`` heredoc is valid bash but crashes PowerShell
(which is the default on ``windows-latest`` AND is explicitly pinned
in our build workflows for lossless path handling). These tests
scan the workflow YAML for the pattern so the author sees it
immediately, instead of after pushing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS_DIR = Path(__file__).parent.parent.parent / ".github" / "workflows"


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yml")) + sorted(WORKFLOWS_DIR.glob("*.yaml"))


def _iter_pwsh_run_scripts(path: Path):
    """Yield (step_name, line_number, script_text) for every step that
    explicitly opts into PowerShell."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return
    jobs = data.get("jobs") or data.get(True) or {}
    if not isinstance(jobs, dict):
        return
    for _job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            shell = step.get("shell", "")
            run = step.get("run")
            if not isinstance(run, str):
                continue
            if "pwsh" in shell or "powershell" in shell:
                yield step.get("name", f"step #{i}"), i, run


_HEREDOC_RE = re.compile(r"<<\s*['\"]?\w+['\"]?\s*$", re.MULTILINE)


class TestNoBashHeredocsInPwsh:
    def test_no_heredoc_redirection_in_pwsh_steps(self) -> None:
        offenders: list[tuple[Path, str, str]] = []
        for wf in _workflow_files():
            for name, _idx, script in _iter_pwsh_run_scripts(wf):
                if _HEREDOC_RE.search(script):
                    # Pick the first offending line for a helpful message.
                    first = next(
                        line for line in script.splitlines() if _HEREDOC_RE.search(line)
                    )
                    offenders.append((wf, name, first.strip()))
        assert not offenders, (
            "Bash heredoc(s) found inside pwsh steps — PowerShell treats "
            "`<<` as a broken redirection and fails the whole step. "
            "Move the payload to a file under .github/scripts/ and invoke "
            "it as `python .github/scripts/<name>.py`.\nOffenders:\n"
            + "\n".join(f"  {wf.name} :: {name} :: {line}" for wf, name, line in offenders)
        )

    def test_workflow_yaml_is_valid(self) -> None:
        """Fail fast if a workflow file stops parsing as YAML."""
        for wf in _workflow_files():
            try:
                yaml.safe_load(wf.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                pytest.fail(f"{wf.name} is not valid YAML: {exc}")


class TestGenerateIcoScript:
    def test_script_exists_and_handles_missing_deps(self) -> None:
        """The extracted script must not crash if cairosvg is absent —
        the workflow relies on the soft-skip behaviour."""
        import subprocess
        import sys

        script = (
            Path(__file__).parent.parent.parent
            / ".github" / "scripts" / "generate_ico.py"
        )
        assert script.exists(), "generate_ico.py was lost"
        # Should always exit 0 — skip is not an error.
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        assert result.returncode == 0, result.stderr

    def test_script_survives_cairosvg_oserror(self, tmp_path: Path) -> None:
        """Simulates the Windows-runner scenario: cairosvg is importable
        but the native libcairo-2.dll is missing, so ``import cairosvg``
        raises ``OSError``. The script must still exit 0 — the workflow
        step is ``continue-on-error: true``.

        We inject a shim ``cairosvg/__init__.py`` on the child's
        ``sys.path`` that raises OSError, replicating the real failure
        without needing Windows + cairocffi at test time.
        """
        import subprocess
        import sys
        import textwrap

        # Build a fake cairosvg package that raises OSError on import.
        shim_root = tmp_path / "shim"
        pkg = shim_root / "cairosvg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text(
            textwrap.dedent(
                """\
                raise OSError(
                    "no library called 'libcairo-2' was found — "
                    "simulated Windows runner state"
                )
                """
            ),
            encoding="utf-8",
        )
        script = (
            Path(__file__).parent.parent.parent
            / ".github" / "scripts" / "generate_ico.py"
        )
        env = {
            **__import__("os").environ,
            # Prepend the shim so our fake cairosvg is found first.
            "PYTHONPATH": str(shim_root)
            + __import__("os").pathsep
            + __import__("os").environ.get("PYTHONPATH", ""),
        }
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
        )
        assert result.returncode == 0, (
            f"generate_ico.py must soft-skip on cairosvg OSError but "
            f"exited {result.returncode}. stderr:\n{result.stderr}"
        )
        assert "skipped" in result.stdout.lower()
