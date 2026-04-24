"""Quick end-to-end confidence measurement on a real TN/UPD.

Loads universal_accurate profile, calls the EasyOCR engine directly
with all our new flags on (auto_upscale, structured_retry, domain_lm),
and reports per-page + aggregate mean confidence. Bypasses the heavy
preprocessor chain to measure the engine-level improvements in
isolation.

Usage:
    python scripts/measure_conf_end_to_end.py [pdf_name]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    pdf_name = sys.argv[1] if len(sys.argv) > 1 else "TN_k_UPD_36_ot_02.09.2022.pdf"
    pdf_path = REPO_ROOT / "inputs" / pdf_name
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        return 1

    # Load universal_accurate profile to get the OCRConfig with all our flags.
    from src.core.models import ProfileData
    profile_json = json.loads(
        (REPO_ROOT / "profiles" / "universal_accurate.json").read_text(
            encoding="utf-8",
        )
    )
    profile = ProfileData.from_dict(profile_json)
    cfg = profile.ocr

    print(f"[test] Profile: {profile.name}")
    print(f"[test]   dpi: {cfg.dpi}")
    print(f"[test]   min_keep_confidence: {cfg.min_keep_confidence}")
    print(f"[test]   auto_upscale_low_dpi: {cfg.auto_upscale_low_dpi}")
    print(f"[test]   structured_retry_enabled: {cfg.structured_retry_enabled}")
    print(f"[test]   domain_lm_correction_enabled: "
          f"{cfg.domain_lm_correction_enabled}")
    print(f"[test]   low_conf_retry_enabled: {cfg.low_conf_retry_enabled}")
    print()

    # Load engine + domain LM directly.
    import easyocr
    from src.shared.domain_lm import DomainLM
    from src.shared.dpi_utils import estimate_page_source_dpi

    print("[test] Loading EasyOCR reader (ru+en, CPU)…")
    reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)
    print("[test] Loading domain LM…")
    lm = DomainLM.from_default_corpus()
    print(f"[test]   LM vocab: {len(lm)} tokens")
    print()

    from src.application.low_conf_retry import retry_low_confidence
    from src.application.structured_retry import validate_and_retry
    from src.core.super_resolution import upscale_for_ocr

    allowlist = cfg.allowlist or None
    craft_kwargs = {
        "text_threshold": float(cfg.craft_text_threshold),
        "low_text": float(cfg.craft_low_text),
        "link_threshold": float(cfg.craft_link_threshold),
        "canvas_size": int(cfg.craft_canvas_size),
        "contrast_ths": float(cfg.easyocr_contrast_ths),
        "adjust_contrast": float(cfg.easyocr_adjust_contrast),
    }

    def _run(arr):
        return reader.readtext(
            arr, detail=1, paragraph=False,
            allowlist=allowlist, **craft_kwargs,
        )

    doc = fitz.open(str(pdf_path))
    print(f"[test] {pdf_name}: {doc.page_count} pages")
    print()
    page_confs: list[float] = []
    total_words = 0
    total_upscaled = 0
    total_sr_fixed = 0
    total_lm_fixed = 0

    try:
        for i in range(doc.page_count):
            page = doc[i]
            src_dpi = (
                estimate_page_source_dpi(page)
                if cfg.auto_upscale_low_dpi else None
            )
            pix = page.get_pixmap(dpi=cfg.dpi, colorspace=fitz.csGRAY)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width,
            )

            # Auto-upscale if applicable.
            if (cfg.auto_upscale_low_dpi and src_dpi is not None
                    and src_dpi < cfg.auto_upscale_threshold):
                arr_up, sr_stats = upscale_for_ocr(
                    arr, source_dpi=src_dpi,
                    target_dpi=max(cfg.dpi, 300),
                    mode="bicubic_sharpen",
                )
                if sr_stats.applied:
                    total_upscaled += 1
                    arr = arr_up
                    print(f"[page {i+1}] auto-upscaled {src_dpi}→"
                          f"{sr_stats.output_dpi} DPI ({sr_stats.scale:.1f}x)")

            raw = _run(arr)

            # Low-conf retry
            if cfg.low_conf_retry_enabled:
                raw, ret_stats = retry_low_confidence(
                    arr, raw, _run,
                    threshold=cfg.low_conf_retry_threshold,
                )
                if ret_stats.improved:
                    print(f"[page {i+1}] low-conf retry: +{ret_stats.improved} "
                          f"improved")

            # Structured retry
            if cfg.structured_retry_enabled:
                raw, sv_stats = validate_and_retry(arr, raw, _run)
                rescued = sv_stats.fixed_offline + sv_stats.fixed_via_retry
                total_sr_fixed += rescued
                if rescued:
                    print(f"[page {i+1}] structured-retry rescued {rescued} "
                          f"requisites ({sv_stats.fixed_offline} offline, "
                          f"{sv_stats.fixed_via_retry} via retry)")

            # Domain LM
            if cfg.domain_lm_correction_enabled and len(lm) > 0:
                corrected = []
                changed = 0
                for bbox, text, conf in raw:
                    if not text.strip():
                        corrected.append((bbox, text, conf))
                        continue
                    r = lm.correct(
                        text, ocr_conf=float(conf),
                        min_ocr_conf_to_skip=cfg.domain_lm_skip_conf,
                    )
                    if r.was_corrected:
                        corrected.append((bbox, r.word, conf))
                        changed += 1
                    else:
                        corrected.append((bbox, text, conf))
                raw = corrected
                total_lm_fixed += changed
                if changed:
                    print(f"[page {i+1}] domain-LM corrected {changed} words")

            confs = [
                float(c) for _b, t, c in raw
                if c >= cfg.min_keep_confidence and t.strip()
            ]
            if confs:
                mean_c = sum(confs) / len(confs) * 100
                page_confs.append(mean_c)
                total_words += len(confs)
                print(f"[page {i+1}] {len(confs)} kept words, "
                      f"mean_conf = {mean_c:.1f}%")
    finally:
        doc.close()

    print()
    print("=" * 60)
    print(f"{'DOCUMENT':>20} = {pdf_name}")
    print(f"{'pages':>20} = {len(page_confs)}")
    print(f"{'total words':>20} = {total_words}")
    print(f"{'pages upscaled':>20} = {total_upscaled}")
    print(f"{'SR fixed requisites':>20} = {total_sr_fixed}")
    print(f"{'LM fixed words':>20} = {total_lm_fixed}")
    if page_confs:
        agg = sum(page_confs) / len(page_confs)
        print(f"{'MEAN CONFIDENCE':>20} = {agg:.2f}%")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
