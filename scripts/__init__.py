"""Developer / ops scripts that live outside the shipped package.

Exposing these as a regular package (``scripts.run_golden``,
``scripts.collect_feedback``, …) lets ``src.cli`` import and
dispatch them as subcommands without resorting to ``runpy`` or
``subprocess.Popen``. The directory is intentionally NOT listed
in ``pyproject.toml::tool.setuptools.packages.find.include`` —
these are repo-local utilities, not end-user APIs.
"""
