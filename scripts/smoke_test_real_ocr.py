"""Real-OCR smoke test — the fastest feedback loop between code and OCR.

Runs a one-page synthetic English PDF through the full pipeline with
the builtin ``universal_accurate`` profile and **real** EasyOCR.
Returns 0 on success, nonzero on any failure. Typical wall time
~30-60 seconds on a midrange laptop (first-run model load dominates).

Intended use:

  * **Before push**: dev runs ``python scripts/smoke_test_real_ocr.py``
    after a local change to ``pipeline.py`` / ``image_preprocessor.py`` /
    ``text_postprocessor.py`` / the engine. If it's green, the change
    hasn't broken the end-to-end happy path.

  * **In CI**: run it as an extra job alongside ``pytest``. Uses the
    same EasyOCR weights we already prefetch in CI.

Design notes:
  * No assertion that the recognised text EQUALS "HELLO" — EasyOCR
    occasionally misreads synthetic glyphs. We only require that AT
    LEAST ONE tri-gram of the input word survives through OCR +
    postprocess. That's the regression signal we care about.

  * Renders English rather than Russian: PyMuPDF's built-in
    ``helv`` font is Latin-only and produces garbled glyphs for
    Cyrillic — the test would fail for a reason unrelated to the OCR
    pipeline we're actually validating.

  * The script intentionally does NOT use ``pytest``. It's supposed to
    run fast, in isolation, with a single ``python scripts/...``
    command and a readable stdout.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Repo-root import path bootstrap (so the script works via
# ``python scripts/smoke_test_real_ocr.py`` from a fresh checkout
# without ``pip install -e``).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _check_engine_available() -> tuple[bool, str]:
    """Return ``(ok, message)`` for EasyOCR availability."""
    from src.application.engines.registry import get_engine
    from src.shared.types import OCREngineKind

    try:
        engine = get_engine(OCREngineKind.EASYOCR)
    except Exception as exc:  # noqa: BLE001
        return False, f"EasyOCR engine unavailable: {exc}"
    return engine.is_available()


def _render_sample_pdf(path: Path) -> None:
    """Render a one-page image-only PDF with a known English phrase."""
    import fitz

    txt_doc = fitz.open()
    try:
        page = txt_doc.new_page(width=612, height=792)
        page.insert_text(
            (72, 200), "HELLO WORLD OCR\nSMOKE TEST 2026",
            fontsize=40, fontname="helv",
        )
        raw = txt_doc.tobytes()
    finally:
        txt_doc.close()

    scan_doc = fitz.open()
    src_doc = fitz.open(stream=raw, filetype="pdf")
    try:
        for src_page in src_doc:
            pix = src_page.get_pixmap(dpi=200, alpha=False)
            new = scan_doc.new_page(
                width=src_page.rect.width, height=src_page.rect.height
            )
            new.insert_image(new.rect, stream=pix.tobytes("png"))
        scan_doc.save(str(path))
    finally:
        scan_doc.close()
        src_doc.close()


def _run_pipeline(
    input_pdf: Path,
    output_pdf: Path,
    profile_name: str,
    *,
    dpi_override: int | None = None,
) -> tuple[bool, str, float]:
    """Run the pipeline with ``profile_name``.

    Returns ``(ok, message, elapsed_seconds)``.
    """
    from src.application.pipeline import OCRPipeline
    from src.application.profile_manager import ProfileManager
    from src.core.image_preprocessor import ImagePreprocessor
    from src.core.models import OCRJobConfig
    from src.core.text_postprocessor import TextPostprocessor
    from src.infrastructure.config_storage import ProfileStorage
    from src.shared.types import JobStatus

    storage = ProfileStorage()
    manager = ProfileManager(storage)
    manager.initialize_builtins()
    try:
        profile = manager.load(profile_name)
    except FileNotFoundError:
        return False, f"Профиль {profile_name!r} не найден", 0.0

    if dpi_override is not None:
        profile.ocr.dpi = dpi_override
    # Force-run OCR: our sample is an image-only PDF, but an accidental
    # text-layer bypass would defeat the smoke signal.
    profile.ocr.skip_text = False

    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        compute_confidence=False,
    )

    t0 = time.time()
    result = pipeline.run(
        OCRJobConfig(
            input_path=str(input_pdf),
            output_path=str(output_pdf),
            profile=profile,
        )
    )
    elapsed = time.time() - t0

    if result.status is not JobStatus.COMPLETED:
        return False, f"Job FAILED: {result.error}", elapsed

    if not output_pdf.exists():
        return False, "COMPLETED но выходной PDF не создан", elapsed

    if not result.pages:
        return False, "COMPLETED но result.pages пуст", elapsed

    recognised = result.pages[0].text.upper()
    expected_trigrams = ("HEL", "WOR", "OCR", "SMO", "TES", "202")
    if not any(tri in recognised for tri in expected_trigrams):
        return (
            False,
            (
                f"COMPLETED, но OCR не распознал ни один ожидаемый "
                f"фрагмент. Распознанный текст: {recognised!r}"
            ),
            elapsed,
        )
    return True, f"OCR прошёл. Распознанный текст: {recognised!r}", elapsed


# Декабрь 2026: builtin профили консолидированы в
# ``universal_accurate`` + ``universal_clean``. Smoke-test прогоняет
# оба, поскольку они оба являются публичным контрактом.
_ALL_EASYOCR_PROFILES: list[str] = [
    "universal_accurate",
    "universal_clean",
]


def main() -> int:
    import contextlib
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(AttributeError, OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description=(
            "Быстрый smoke-test: прогоняет синтетический PDF через "
            "полный пайплайн с реальным EasyOCR. По умолчанию "
            "проверяет оба builtin-профиля."
        )
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Имя одного конкретного профиля. Если не задан — "
            "прогоняются оба builtin-профиля."
        ),
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Не удалять входной/выходной PDF из tmp после прогона",
    )
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else _ALL_EASYOCR_PROFILES

    print("=" * 60)
    print("OCR Studio — real-OCR smoke test (EasyOCR)")
    print(f"Профили: {', '.join(profiles)}")
    print("=" * 60)

    print("\n[1/3] Проверка доступности EasyOCR…")
    ok, msg = _check_engine_available()
    if not ok:
        print(f"  ❌ {msg}")
        return 2
    print(f"  ✅ {msg}")

    import tempfile

    work = Path(tempfile.mkdtemp(prefix="ocr-smoke-"))
    input_pdf = work / "sample.pdf"
    try:
        print(f"\n[2/3] Генерация тестового PDF ({input_pdf.name})…")
        _render_sample_pdf(input_pdf)
        print(f"  ✅ {input_pdf.stat().st_size} байт")

        failures = 0
        total = len(profiles)
        for idx, profile_name in enumerate(profiles, 1):
            output_pdf = work / f"out_{profile_name}.pdf"
            print(
                f"\n[3/3] [{idx}/{total}] Профиль {profile_name!r}…"
            )
            # Smoke test stays at 200 DPI — adequate for EasyOCR on
            # the synthetic 40pt sample and keeps wall time short.
            ok, msg, elapsed = _run_pipeline(
                input_pdf, output_pdf, profile_name,
                dpi_override=200,
            )
            status = "✅" if ok else "❌"
            print(f"  {status} За {elapsed:.1f} с: {msg}")
            if not ok:
                failures += 1

        if args.keep_artifacts:
            print(f"\nАртефакты остались в {work}")

        if failures:
            print(f"\n{'=' * 60}")
            print(f"❌ {failures}/{total} профилей упали.")
            return 1
        print(f"\n{'=' * 60}")
        print(f"✅ Все {total} профилей прошли.")
        return 0
    finally:
        if not args.keep_artifacts:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
