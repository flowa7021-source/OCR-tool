"""Integration-level conftest: re-export fixtures from _real_ocr_helpers
and register plugin hooks that must live in conftest.py (pytest refuses
to register ``pytest_addoption`` from a test module).

pytest discovers fixtures in conftest.py automatically so test modules
don't need to ``import real_tesseract_wrapper`` at module scope — that
import pattern triggers ruff F811 (redefinition) because the fixture
name reappears as a test-method parameter.
"""

from tests.integration._real_ocr_helpers import (  # noqa: F401
    real_tesseract_wrapper,
)


def pytest_addoption(parser) -> None:
    """Register custom CLI flags for integration tests.

    ``--regenerate-baseline`` is used by the accuracy benchmark
    (``tests/integration/test_accuracy_benchmark.py``) to overwrite
    ``baseline.json`` with the current measurements. The hook MUST
    live in conftest.py, not in the test module: pytest discovers
    add-option hooks during plugin loading, which happens before
    test modules are imported.
    """
    parser.addoption(
        "--regenerate-baseline",
        action="store_true",
        default=False,
        help=(
            "Run the accuracy benchmark and OVERWRITE "
            "tests/fixtures/accuracy_corpus/baseline.json with the "
            "measured values. Use after an intentional accuracy "
            "improvement."
        ),
    )
