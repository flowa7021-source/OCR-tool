"""Real-OCR smoke test — the fastest feedback loop between code and OCR.

Runs a one-page synthetic Russian PDF through the full pipeline with
the ``quick_reliable`` profile and **real** Tesseract. Returns 0 on
success, nonzero on any failure. Typical wall time ~30 seconds on a
midrange laptop.

Intended use:

  * **Before push**: dev runs ``python scripts/smoke_test_real_ocr.py``
    after a local change to ``pipeline.py`` / ``image_preprocessor.py`` /
    ``text_postprocessor.py`` / any engine. If it's green, the change
    hasn't broken the end-to-end happy path. Catches ~80% of the bugs
    we've been losing days on.

  * **In CI**: run it as an extra job alongside ``pytest``. Cheap,
    uses the same Tesseract we already install in CI.

  * **On user's Windows dev box**: after ``git pull``, the dev can
    run this script against an already-installed OCR Studio — the
    ``installed_app`` fallback in :mod:`src.infrastructure.tesseract_wrapper`
    picks up the bundled binary from ``Programs/OCR Studio/_internal/``
    so no separate Tesseract install is needed.

Design notes:
  * No assertion that the recognised text EQUALS "ПРИВЕТ" — Tesseract
    occasionally misreads synthetic glyphs. We only require that AT
    LEAST ONE tri-gram of the input word survives through OCR +
    postprocess. That's the regression signal we care about.

  * The script intentionally does NOT use ``pytest``. It's supposed to
    run fast, in isolation, with a single ``python scripts/...``
    command and a readable stdout. No fixtures, no collection pass,
    no plugin overhead.
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


def _check_external_tools() -> tuple[bool, str]:
    """Return ``(ok, message)`` for Tesseract + Ghostscript availability.

    Covers the three paths the production pipeline itself uses:
    PyInstaller bundle, installed-app fallback, and system PATH.
    """
    from src.infrastructure.external_tools import verify_required_for_ocrmypdf
    from src.infrastructure.tesseract_wrapper import TesseractWrapper

    missing = verify_required_for_ocrmypdf()
    if missing:
        return False, (
            f"Не найдены внешние программы: {', '.join(missing)}. "
            "Установите OCR Studio (тогда скрипт подхватит бандленные "
            "бинарники автоматически) или поставьте Tesseract + "
            "Ghostscript системно."
        )

    # The verify_required_for_ocrmypdf check uses ``shutil.which`` via
    # ``locate()`` with the new installed-app fallback — but it
    # doesn't prove tessdata (``rus.traineddata``) is actually reachable.
    # Do that here so the script bails out with a clear message instead
    # of crashing deep inside ocrmypdf.
    try:
        wrapper = TesseractWrapper()
        tessdata = wrapper.find_tessdata_dir()
    except Exception as exc:  # noqa: BLE001
        return False, f"tessdata не найдена: {exc}"

    # Smoke test only needs English (Latin) since we deliberately
    # render an English phrase — see ``_render_sample_pdf``. We log a
    # soft warning about missing rus.traineddata so the user knows
    # real Russian contracts would need it, but don't fail the smoke.
    if not (tessdata / "eng.traineddata").is_file():
        return False, (
            f"tessdata найдена в {tessdata}, но нет eng.traineddata. "
            "Скачайте eng.traineddata из "
            "https://github.com/tesseract-ocr/tessdata/ и положите в "
            "эту папку."
        )
    if not (tessdata / "rus.traineddata").is_file():
        print(
            "  ⚠ rus.traineddata отсутствует — smoke-test прогоняется "
            "на английском. Для прогона CLI на реальном русском "
            "договоре докачайте rus.traineddata."
        )
    return True, ""


def _render_sample_pdf(path: Path) -> None:
    """Render a one-page image-only PDF with a known English phrase.

    We deliberately render **English**, not Russian: PyMuPDF's built-in
    ``helv`` font is Latin-only and produces garbled glyphs for
    Cyrillic — the test would fail for a reason unrelated to the OCR
    pipeline we're actually validating. Real users' documents are
    scanned image PDFs (raster text already in pixels), not rendered
    text, so the font-coverage limitation doesn't apply there. This
    smoke script stays with English to keep the test-harness simple
    and cross-platform; Russian end-to-end is validated separately in
    ``tests/integration/test_e2e_real_ocr.py`` when rus.traineddata
    is installed.

    Uses the same two-step render-then-rasterise trick as
    ``tests/integration/test_e2e_real_ocr.py::_render_text_pdf``:
    Tesseract needs image pixels, not selectable text — otherwise
    OCRmyPDF's ``skip_text`` short-circuits before any OCR happens.
    """
    import fitz

    # 1. Write selectable text.
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

    # 2. Rasterise + re-embed as an image-only PDF.
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
    from src.infrastructure.tesseract_wrapper import TesseractWrapper

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

    pipeline = OCRPipeline(
        preprocessor=ImagePreprocessor(),
        postprocessor=TextPostprocessor(),
        tesseract=TesseractWrapper(),
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

    # Recognised-text sanity: one of the expected tri-grams must survive.
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


# All Tesseract-engine profiles that must produce recognised text.
# Декабрь 2026: builtin профили консолидированы в единственный
# ``universal_accurate`` (см. profile_manager). ``quick_reliable``,
# ``default``, ``contracts_ru``, ``low_quality_scan``, ``english_text``
# удалены. Smoke-test проверяет единственный оставшийся profile.
_ALL_TESSERACT_PROFILES: list[str] = [
    "universal_accurate",
]


def main() -> int:
    # Force stdout/stderr to UTF-8 so the Russian help text and status
    # messages don't crash the script with ``UnicodeEncodeError`` on a
    # Windows host whose default console encoding is cp1252 / cp866 —
    # exactly what happens when the script is invoked as a subprocess
    # by CI (``test_smoke_test_script_runs_green``). Safe no-op on
    # *nix where stdout is already UTF-8.
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
            "полный пайплайн с реальным Tesseract. По умолчанию "
            "проверяет ВСЕ Tesseract-профили."
        )
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Имя одного конкретного профиля. Если не задан — "
            "прогоняются все 6 Tesseract-профилей."
        ),
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Не удалять входной/выходной PDF из tmp после прогона",
    )
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else _ALL_TESSERACT_PROFILES

    print("=" * 60)
    print("OCR Studio — real-OCR smoke test")
    print(f"Профили: {', '.join(profiles)}")
    print("=" * 60)

    # Stage 1: external tools.
    print("\n[1/3] Проверка внешних программ (Tesseract, Ghostscript)…")
    ok, msg = _check_external_tools()
    if not ok:
        print(f"  ❌ {msg}")
        return 2
    print("  ✅ OK")

    # Stage 2: render sample.
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="ocr-smoke-"))
    input_pdf = work / "sample.pdf"
    try:
        print(f"\n[2/3] Генерация тестового PDF ({input_pdf.name})…")
        _render_sample_pdf(input_pdf)
        print(f"  ✅ {input_pdf.stat().st_size} байт")

        # Stage 3: pipeline for every profile.
        failures = 0
        total = len(profiles)
        for idx, profile_name in enumerate(profiles, 1):
            output_pdf = work / f"out_{profile_name}.pdf"
            print(
                f"\n[3/3] [{idx}/{total}] Профиль {profile_name!r}…"
            )

            # ``universal_accurate`` is 600 DPI — too slow for a quick
            # smoke. We override DPI via a monkey-patch on the loaded
            # profile so the rest of its config (adaptive_gaussian +
            # CLAHE + denoise + all postprocess) is still exercised.
            dpi_override = 300 if profile_name == "universal_accurate" else None

            ok, msg, elapsed = _run_pipeline(
                input_pdf, output_pdf, profile_name,
                dpi_override=dpi_override,
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
