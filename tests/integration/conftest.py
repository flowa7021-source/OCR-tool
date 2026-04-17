"""Integration-level conftest: re-export fixtures from _real_ocr_helpers.

pytest discovers fixtures in conftest.py automatically so test modules
don't need to ``import real_tesseract_wrapper`` at module scope — that
import pattern triggers ruff F811 (redefinition) because the fixture
name reappears as a test-method parameter.
"""

from tests.integration._real_ocr_helpers import (  # noqa: F401
    real_tesseract_wrapper,
)
