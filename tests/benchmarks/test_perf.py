"""Performance benchmarks for hot-path operations.

Run with::

    pytest tests/benchmarks --benchmark-only

Used to detect regressions in:

  * ImagePreprocessor full pipeline on a synthetic A4 page
  * Per-stage preprocessing operations (binarisation, denoise, etc.)
  * TextPostprocessor on a one-page Russian document
  * Profile JSON serialisation / deserialisation

CI runs them in non-blocking mode (``--benchmark-disable-gc
--benchmark-warmup-iterations=2 --benchmark-disable``) — they're not
asserted against absolute thresholds because runner perf varies wildly.
Compare against the previous run via ``--benchmark-compare``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")
pytest.importorskip("cv2")

import numpy as np  # noqa: E402

from src.core.image_preprocessor import ImagePreprocessor, preview_step  # noqa: E402
from src.core.models import (  # noqa: E402
    BinarizationConfig,
    ContrastConfig,
    DenoiseConfig,
    DenoiseStep,
    DeskewConfig,
    PostprocessConfig,
    PreprocessConfig,
    ProfileData,
    RegexRule,
)
from src.core.text_postprocessor import TextPostprocessor  # noqa: E402
from src.shared.types import BinarizationMethod, DenoiseMethod  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_page() -> np.ndarray:
    """Generate an A4-sized greyscale-ish image with text-like stripes.

    ~2480x3508 px = 300 DPI A4. Forced to 1240x1754 here to keep the
    benchmark suite under a few seconds total.
    """
    import cv2

    h, w = 1754, 1240
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    # ~40 dark horizontal "lines" simulating body text
    for y in range(80, h - 80, 35):
        cv2.rectangle(img, (60, y), (w - 60, y + 12), (15, 15, 15), thickness=-1)
    # Salt-and-pepper noise
    rng = np.random.default_rng(42)
    for _ in range(2000):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        img[y, x] = (0, 0, 0)
    # Gradient shadow in lower-right corner
    gradient = np.tile(
        np.linspace(0, 80, w, dtype=np.float32), (h, 1)
    )
    img = np.clip(img.astype(np.float32) - gradient[..., None], 0, 255).astype(np.uint8)
    return img


@pytest.fixture(scope="module")
def russian_page_text() -> str:
    """A typical OCR'd one-page result with realistic noise."""
    base = (
        "Московский Государ-\n"
        "ственный Университет имени М.В.Ломоносова.\n\n"
        "Договор № 0001/24 от 15 марта 2024 года\n"
        "Стороны:\n"
        "Заказчик: ООО «Ромашка», ИНН 7700000000\n"
        "Исполнитель: ИП Иванов И.И., ИНН 770000000001\n\n"
        "1. Предмет договора\n"
        "Исполнитель обязуется оказать услуги по разработке программного "
        "обеспечения, а Заказчик — принять и оплатить эти услуги.\n\n"
        "2. Стоимость и порядок расчётов\n"
        "Общая стоимость работ составляет 1 000 000 (один миллион) рублей.\n"
    )
    return base * 5  # 5 pages of text


# ---------------------------------------------------------------------------
# ImagePreprocessor
# ---------------------------------------------------------------------------


class TestPreprocessorBenchmarks:
    def test_full_default_pipeline(self, benchmark, synthetic_page) -> None:
        """End-to-end: deskew + contrast + denoise + binarise."""
        cfg = PreprocessConfig()
        cfg.deskew = DeskewConfig(enabled=True, auto_detect=False, manual_angle=0.0)
        cfg.contrast = ContrastConfig(clahe_enabled=True, clahe_clip=2.0)
        cfg.denoise = DenoiseConfig(
            enabled=True,
            steps=[DenoiseStep(method=DenoiseMethod.MEDIAN, ksize=3)],
        )
        cfg.binarization = BinarizationConfig(method=BinarizationMethod.OTSU)
        pre = ImagePreprocessor()
        benchmark(lambda: pre.process(synthetic_page, cfg))

    def test_only_otsu_binarisation(self, benchmark, synthetic_page) -> None:
        cfg = PreprocessConfig()
        cfg.deskew = DeskewConfig(enabled=False)
        cfg.contrast = ContrastConfig()
        cfg.denoise = DenoiseConfig(enabled=False)
        cfg.binarization = BinarizationConfig(method=BinarizationMethod.OTSU)
        pre = ImagePreprocessor()
        benchmark(lambda: pre.process(synthetic_page, cfg))

    def test_only_adaptive_binarisation(self, benchmark, synthetic_page) -> None:
        cfg = PreprocessConfig()
        cfg.deskew = DeskewConfig(enabled=False)
        cfg.contrast = ContrastConfig()
        cfg.denoise = DenoiseConfig(enabled=False)
        cfg.binarization = BinarizationConfig(
            method=BinarizationMethod.ADAPTIVE_GAUSSIAN
        )
        pre = ImagePreprocessor()
        benchmark(lambda: pre.process(synthetic_page, cfg))

    def test_clahe_only(self, benchmark, synthetic_page) -> None:
        cfg = PreprocessConfig()
        cfg.deskew = DeskewConfig(enabled=False)
        cfg.contrast = ContrastConfig(clahe_enabled=True, clahe_clip=2.0)
        cfg.denoise = DenoiseConfig(enabled=False)
        cfg.binarization = BinarizationConfig(method=BinarizationMethod.NONE)
        pre = ImagePreprocessor()
        benchmark(lambda: pre.process(synthetic_page, cfg))

    def test_nlm_denoise(self, benchmark, synthetic_page) -> None:
        """NLM is the slowest denoise variant — separately tracked."""
        cfg = PreprocessConfig()
        cfg.deskew = DeskewConfig(enabled=False)
        cfg.contrast = ContrastConfig()
        cfg.denoise = DenoiseConfig(
            enabled=True,
            steps=[DenoiseStep(method=DenoiseMethod.NLM, h=7)],
        )
        cfg.binarization = BinarizationConfig(method=BinarizationMethod.NONE)
        pre = ImagePreprocessor()
        benchmark.pedantic(
            lambda: pre.process(synthetic_page, cfg),
            iterations=1,
            rounds=3,
        )

    def test_preview_binarisation_step(self, benchmark, synthetic_page) -> None:
        """preview_step is called on every UI tick — must stay snappy."""
        cfg = PreprocessConfig()
        benchmark(lambda: preview_step(synthetic_page, "binarization", cfg))


# ---------------------------------------------------------------------------
# TextPostprocessor
# ---------------------------------------------------------------------------


class TestPostprocessorBenchmarks:
    def test_default_postprocess(self, benchmark, russian_page_text) -> None:
        cfg = PostprocessConfig()
        post = TextPostprocessor()
        benchmark(lambda: post.process(russian_page_text, cfg))

    def test_postprocess_with_50_custom_rules(self, benchmark, russian_page_text) -> None:
        """Many users build up large rule lists over time."""
        rules = [
            RegexRule(
                pattern=f"foo{i}",
                replacement=f"bar{i}",
                is_regex=False,
                enabled=True,
            )
            for i in range(50)
        ]
        cfg = PostprocessConfig(custom_rules=rules)
        post = TextPostprocessor()
        benchmark(lambda: post.process(russian_page_text, cfg))


# ---------------------------------------------------------------------------
# Profile JSON
# ---------------------------------------------------------------------------


class TestProfileSerialisation:
    def test_to_dict(self, benchmark) -> None:
        profile = ProfileData(name="bench")
        benchmark(profile.to_dict)

    def test_from_dict(self, benchmark) -> None:
        data = ProfileData(name="bench").to_dict()
        benchmark(lambda: ProfileData.from_dict(data))
