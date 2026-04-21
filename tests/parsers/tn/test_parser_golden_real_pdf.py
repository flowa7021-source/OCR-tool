"""Golden regression на 5 реальных ТН/УПД PDF.

Каждый тест:

  1. Берёт PDF из :file:`inputs/`.
  2. Прогоняет через реальный pipeline (Tesseract + Ghostscript +
     ImagePreprocessor + TextPostprocessor + tn_parser).
  3. Сравнивает первую ``ParsedRow`` с golden-spec из
     :file:`tests/parsers/tn/golden_fixtures/<pdf_stem>.json`.
  4. Пасс — если overall-score ≥ :data:`MIN_GOLDEN_SCORE`.

**TDD-контракт:**

    После закрытия багов #1, #4, #7, #2, #3, #9 из top-10 overall-
    score должен расти. Тест не требует 100%-совпадения (OCR-noise
    легитимно портит некоторые токены) — но и не терпит регрессов:
    ``MIN_GOLDEN_SCORE = 0.55`` = 55 % взвешенного match'а на
    baseline (сейчас), и эта планка только поднимается в будущих
    PR'ах. Если PR проседает ниже baseline — тест падает с
    detail'ным отчётом по каждому field'у.

**Когда запускается:**

    Тест помечен ``@pytest.mark.parser_golden`` и skip'ается на PR-CI
    (слишком медленный — 4-10 мин per PDF × 5 PDF). Гоняется только
    в nightly workflow
    :file:`.github/workflows/parser-golden.yml` либо локально с
    явным ``pytest -m parser_golden``.

**Как обновлять baseline:**

    1. Улучшил парсер → прогнал golden локально → увидел новый
       overall-score.
    2. Поднял ``MIN_GOLDEN_SCORE`` на новый пол в этом файле.
    3. Commit с комментарием «golden: поднят порог до X% после
       закрытия idea #N».
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from tests.parsers.tn.golden_compare import compare_row, format_report

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INPUTS_DIR = REPO_ROOT / "inputs"
FIXTURES_DIR = Path(__file__).resolve().parent / "golden_fixtures"

#: Минимальный overall-score для пасса одного документа. Растёт
#: после каждой closed idea из top-10. Baseline апрель 2026 после
#: идей #4 (infra) + #5 bug-fixes = 55 %. Целевой post-#1/#2/#7:
#: 75 %. Post-#3/#9: 85 %.
MIN_GOLDEN_SCORE: float = 0.55

#: Сам список PDF -> golden fixture. Имя PDF = stem JSON.
_GOLDEN_PAIRS = [
    ("TN_k_UPD_36_ot_02.09.2022.pdf", "TN_k_UPD_36_ot_02.09.2022.json"),
    ("TN_k_UPD_41_ot_06.09.2022.pdf", "TN_k_UPD_41_ot_06.09.2022.json"),
    ("TN_k_UPD_47_ot_09.01.2023.pdf", "TN_k_UPD_47_ot_09.01.2023.json"),
    ("TN_k_UPD_48_ot_09.01.2023.pdf", "TN_k_UPD_48_ot_09.01.2023.json"),
    ("UPD_662_ot_22.10.2022.pdf",     "UPD_662_ot_22.10.2022.json"),
]


def _requires_real_ocr_and_inputs():
    """Skip-reasons собраны в один хелпер чтобы не дублировать.

    Skip если:
      1. Tesseract / Ghostscript отсутствуют (unit-host).
      2. Inputs отсутствуют (checkouts без LFS-assets).
    """
    import shutil as _shutil

    reasons = []
    if not _shutil.which("tesseract"):
        reasons.append("tesseract не на PATH")
    if not _shutil.which("gs"):
        reasons.append("ghostscript не на PATH")
    if not INPUTS_DIR.exists() or not any(INPUTS_DIR.glob("*.pdf")):
        reasons.append("inputs/*.pdf отсутствуют (нужны реальные сканы)")
    return reasons


pytestmark = [
    pytest.mark.parser_golden,
    pytest.mark.skipif(
        bool(_requires_real_ocr_and_inputs()),
        reason=f"golden requires: {', '.join(_requires_real_ocr_and_inputs()) or '(none)'}",
    ),
]


def _run_pipeline_on_pdf(pdf_path: Path, workdir: Path) -> dict:
    """Прогнать полный pipeline и вернуть first ParsedRow как dict.

    Возвращает пустой dict если парсер не вернул rows — caller
    интерпретирует это как 0% match.
    """
    from src.application.pipeline import OCRPipeline
    from src.application.profile_manager import ProfileManager
    from src.core.doc_catalog import load_default_catalog
    from src.core.image_preprocessor import ImagePreprocessor
    from src.core.models import OCRJobConfig
    from src.core.text_postprocessor import TextPostprocessor
    from src.infrastructure.config_storage import ProfileStorage
    from src.infrastructure.tesseract_wrapper import TesseractWrapper

    # OCR_DISABLE_CACHE=1 — кэш может скрыть регрессию новой версии
    # парсера, возвращая предыдущий prepros'нутый результат. Golden
    # prohibited from using cache — see Bug 7.
    os.environ["OCR_DISABLE_CACHE"] = "1"

    profile_dir = workdir / "profiles"
    storage = ProfileStorage(profiles_dir=profile_dir)
    mgr = ProfileManager(storage)
    mgr.initialize_builtins()
    profile = mgr.load("universal_accurate")
    tess = TesseractWrapper()
    tess.configure_pytesseract()

    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(catalog=load_default_catalog()),
        tesseract=tess,
        compute_confidence=True,
    )
    result = pipeline.run(OCRJobConfig(
        input_path=str(pdf_path),
        output_path=str(workdir / f"{pdf_path.stem}_ocr.pdf"),
        profile=profile,
    ))

    if not result.parsed or not result.parsed.rows:
        return {}
    return dict(result.parsed.rows[0])


@pytest.mark.parametrize(
    ("pdf_name", "fixture_name"),
    _GOLDEN_PAIRS,
    ids=[p[0] for p in _GOLDEN_PAIRS],
)
def test_parser_golden_real_pdf(
    pdf_name: str,
    fixture_name: str,
    tmp_path: Path,
    caplog,
) -> None:
    """Golden-sanity: overall-score ≥ :data:`MIN_GOLDEN_SCORE`.

    Лог содержит детальный per-field отчёт на failure, чтобы
    разработчик сразу видел какие именно поля регрессировали.
    """
    caplog.set_level(logging.WARNING)

    pdf_path = INPUTS_DIR / pdf_name
    fixture_path = FIXTURES_DIR / fixture_name
    assert pdf_path.exists(), f"Missing source PDF: {pdf_path}"
    assert fixture_path.exists(), f"Missing golden fixture: {fixture_path}"

    expected_spec = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected_row = expected_spec["expected_row"]

    actual_row = _run_pipeline_on_pdf(pdf_path, tmp_path)

    checks, overall = compare_row(actual_row, expected_row)
    report = format_report(checks, overall)

    # Логируем отчёт всегда — полезно для сравнения прогонов.
    logging.info("\n==== GOLDEN REPORT [%s] ====\n%s", pdf_name, report)

    assert overall >= MIN_GOLDEN_SCORE, (
        f"\nGolden regression on {pdf_name}: "
        f"overall {overall*100:.1f}% < threshold {MIN_GOLDEN_SCORE*100:.0f}%\n"
        f"{report}\n"
        f"Чтобы обновить baseline: улучши парсер, прогони "
        f"`pytest -m parser_golden -v`, подними MIN_GOLDEN_SCORE."
    )
