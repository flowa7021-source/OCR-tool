# Tier D / Deferred — accuracy roadmap

This document covers the items still on the accuracy roadmap that are
too large or too risky to land in a single commit. Each section is a
full implementation plan — scope, files touched, test plan, rollout
risk, and an estimate.

Tiers A-C (DPI-adaptive preprocessing, 400-DPI default, Tesseract -c
flags, min-cache-confidence, text-layer bypass, Cyrillic↔Latin numeric-
context heuristic, benchmark matrix, auto-orientation, adaptive
threshold, wrong-profile warning, low-conf export markers, per-word
script disambiguation) are landed on branch
``claude/ocr-app-development-Hh9xL``.

---

## D1 — Tesseract 5.3+ upgrade

Goal: pick up the improved layout analyser and the new ``tessdata_best``
Russian model (``rus.traineddata`` from Oct 2023 vs the 2021 vintage we
ship today). Expected lift: 2-5 % CER on mixed-layout Russian scans.

### Scope

  * Rebuild the Windows installer with Tesseract 5.3.3 (current stable)
  * Swap to ``tessdata_best/rus.traineddata`` (bigger, slower — use
    ``tessdata_fast`` only if the benchmark justifies the ~30 % latency
    saving on dense documents)
  * Refresh ``resources/tessdata/`` in the bundle
  * Verify our ``user-words.rus`` / ``user-patterns.rus`` still load
    (the binary format is stable but the Tesseract 5.3 launcher adds
    validation we didn't have before)
  * Re-run ``scripts/benchmark_matrix.py`` across the synthetic corpus
    and publish the CER delta in the release notes

### Files touched

  * ``.github/workflows/build-installer.yml`` — bump the Tesseract
    download URL and checksum
  * ``installer/setup.iss`` — unchanged structurally, just pulls the
    new bundle
  * ``resources/tessdata/*`` — regenerated from the new upstream
  * ``src/infrastructure/tesseract_wrapper.py`` — version check in
    ``configure_pytesseract()`` accepts 5.3.x
  * ``tests/integration/test_tesseract_version_parse.py`` — add
    5.3.3 / 5.3.4 to the parse regression suite
  * ``docs/tesseract_upgrade_notes.md`` — new file documenting the
    before/after CER numbers, rollback procedure

### Test plan

  * Smoke test on Windows installer: run the full ``benchmark_matrix``
    on the bundled synthetic corpus, CSV output attached to release
    notes
  * Nightly ``test_accuracy_benchmark.py`` regenerate-baseline, commit
    the new numbers with a ``docs/`` entry explaining the upgrade
  * Run the existing retry-tier suite (``test_e2e_retry_tiers.py``)
    unchanged — if a retry tier starts firing on documents that
    previously completed cleanly, the 5.3 layout analyser might have
    regressed on our adversarial corpus
  * ``user-words`` / ``user-patterns`` smoke: add a test PDF that
    contains ``ИНН 7701234567`` and check the exact token survives
    the upgrade

### Risks

  * ``tessdata_best`` is ~1.7× the size of ``tessdata`` — installer
    bundle grows ~40 MB. Budget check before shipping.
  * Tesseract 5.3 changed how OSD reports confidence on pages with
    fewer than ~100 glyphs — our ``MIN_ORIENTATION_CONFIDENCE`` floor
    of 1.0 might need re-tuning. Measure on the benchmark corpus.
  * Some languages-other-than-rus/eng regressed in 5.3 per upstream
    bug tracker; unlikely to affect us but worth a CHANGELOG skim.

### Estimate

  * 1 engineer-day for the bundle bump + benchmark run
  * 0.5 day for release notes and rollback rehearsal
  * **Total: ~1.5 days**

---

## D2 — Hybrid engine: Tesseract + PaddleOCR for tables

Goal: Tesseract's layout analyser is reliably worse than PaddleOCR's
on complex table-heavy pages (invoices, financial statements, balance
sheets). Wrap PaddleOCR as a second engine and dispatch per-page
based on a cheap table-detection heuristic.

### Scope

  * Add PaddleOCR (Apache-2.0) as an optional dependency under a new
    ``paddle`` extras group
  * Wrap it in a new ``PaddleEngine`` implementing
    ``src.application.engines.base.OCREngine``
  * Write a lightweight table detector (Hough lines over the
    preprocessed page; if ≥ N horizontal + ≥ M vertical long runs
    → table-heavy → route to Paddle)
  * Multi-engine dispatch logic in ``src/application/pipeline.py``:
    per-page, pick ``tesseract`` or ``paddle`` based on the detector
  * Merge: each page's result is still a ``PageResult``; the only
    difference is which engine produced it. Metadata field
    ``PageResult.engine_used: str`` so the UI / DMS consumer can
    colour-code (optional)
  * UI toggle in ``src/ui/preferences_dialog.py``: "Use PaddleOCR
    for table-heavy pages"

### Files touched

  * ``pyproject.toml`` — ``[project.optional-dependencies] paddle``
    group with ``paddlepaddle>=2.6``, ``paddleocr>=2.7``
  * ``src/application/engines/paddle_engine.py`` — new file
  * ``src/application/engines/registry.py`` — register Paddle under
    ``OCREngineKind.PADDLE``
  * ``src/shared/types.py`` — add ``OCREngineKind.PADDLE``
  * ``src/core/table_detector.py`` — new file, Hough-based detector
  * ``src/application/pipeline.py`` — per-page engine dispatch hook
    before ``engine.run()``
  * ``src/core/models.py`` — new ``OCRConfig.table_dispatch: bool``
    and ``PageResult.engine_used: str``
  * ``build.py`` — bundle Paddle binaries only when ``--with-paddle``
    (like the old ``--with-htr`` flag for GOT-OCR; we remember the
    weight of GOT's bundle and want this to stay opt-in)
  * Tests: ``tests/unit/test_paddle_engine.py``,
    ``tests/unit/test_table_detector.py``,
    ``tests/integration/test_hybrid_dispatch.py``

### Test plan

  * Unit: table-detector regressions against a fixture set of
    known-table / known-prose pages
  * Integration: synthetic 3-page PDF (prose + table + prose); the
    middle page MUST go through Paddle, the others through Tesseract
  * Accuracy: add an invoice / statement corpus to
    ``scripts/benchmark_matrix.py`` with Tesseract-only vs hybrid
    columns, publish CER delta
  * Regression: every existing test runs with
    ``table_dispatch=False`` as today

### Risks

  * **Bundle weight.** PaddleOCR + Paddle runtime ~700 MB bundled;
    bigger than the GOT-OCR 580 MB we just cut. Default installer
    stays Tesseract-only; ``--with-paddle`` opt-in only.
  * **Memory footprint.** Paddle's detection + recognition models
    want ~1.2 GB combined on CPU. Our per-page memory budget
    (500 MB/worker) has to account for this or we exhaust RAM on
    multi-file queues. Plan: halve concurrent workers when Paddle
    is in use.
  * **Licensing review.** Apache-2.0 is fine for the runtime but
    the pretrained weights on HuggingFace have their own licence
    — confirm with a repo audit before shipping.
  * **User perception.** We just removed GOT-OCR for being worse
    than Tesseract on the user's documents. Need hard CER numbers
    that Paddle wins on table-heavy pages before shipping, or we're
    adding complexity for no user-visible win.

### Estimate

  * 1.5 days for the engine wrapper + table detector + dispatch
  * 1 day for the benchmark corpus expansion + CER measurement
  * 0.5 day for bundle / installer work
  * 0.5 day for licence review + release notes
  * **Total: ~3.5 days**

---

## D3 — Post-OCR hunspell spell-check for Russian

Goal: pyhunspell against ``hunspell-ru`` dict (Apache-licensed) as a
last-pass corrector. Tesseract's built-in autocorrect (via the DAWG)
fixes 70-80 % of one-edit errors; hunspell catches the rest because
its dictionary is ~10× bigger and it knows about morphology (case /
number / gender endings our ``user-words.rus`` doesn't cover).

### Scope

  * Bundle ``hunspell`` binary + ``ru_RU`` dictionary (~6 MB combined)
    in the installer
  * Add ``cyhunspell`` (pure-Python ctypes wrapper) as a dependency
  * New ``src/core/hunspell_corrector.py`` with a ``correct_text(text)
    → (corrected, list[(original, corrected, confidence)])`` API
  * Integrate into ``TextPostprocessor`` behind a new
    ``PostprocessConfig.hunspell_correction: bool = False`` flag
  * Conservative corrector — only applies a substitution when:
    1. The original word is NOT in the custom ``user-words.rus``
       (protects product codes, ИНН, proprietary terms)
    2. Hunspell reports exactly ONE suggestion at edit distance ≤ 1
    3. The candidate is more frequent than the original (via
       dictionary ranking)

### Files touched

  * ``pyproject.toml`` — ``cyhunspell>=2.0``
  * ``resources/hunspell/{ru_RU.dic,ru_RU.aff}`` — new bundle files
  * ``build.py`` — include hunspell dict in ``resources/``
  * ``installer/setup.iss`` — reference the new resource files
  * ``src/core/hunspell_corrector.py`` — new module
  * ``src/core/text_postprocessor.py`` — wire up the flag
  * ``src/core/models.py`` — new ``PostprocessConfig.hunspell_correction``
  * Tests: ``tests/unit/test_hunspell_corrector.py``,
    regression against the accuracy benchmark corpus

### Test plan

  * Unit: 50 hand-curated (wrong, right) pairs from real OCR output
    on Russian contracts. Hunspell must fix ≥ 80 % without
    regressing any.
  * Regression: ``user-words.rus`` tokens MUST NOT be touched by
    hunspell even when hunspell "knows better" (ИНН 7701234567
    isn't in the dictionary, but it's in user-words, and the user
    wants it surfaced verbatim).
  * Benchmark: nightly CER delta on the synthetic corpus; accept
    only if CER improves and WER doesn't regress.
  * UI: the corrector reports every correction it made; the
    results-panel side-bar shows a "corrections applied" count
    with an expand-for-details affordance.

### Risks

  * **Over-correction.** Russian legal / technical text has many
    words hunspell doesn't know (ГОСТ-numbered terms, entity names,
    abbreviations). Protection: the user-words lookup above, AND a
    blocklist of patterns (``[А-Я]{2,5}-\d+``, regex IDs) that
    skip correction.
  * **Runtime cost.** Hunspell's C library is fast (~1 ms per word)
    but we spawn the corrector per page — for 500-page documents
    that's ~2 seconds, acceptable.
  * **Dictionary quality.** ``ru_RU.aff`` is hand-maintained by the
    LibreOffice project; it's good but not perfect. Worth checking
    whether ``aot.ru``'s morphological dictionary is a better fit
    (BSD-licensed, more vocabulary) — extra research before
    committing to hunspell.

### Estimate

  * 1 day for the corrector module + tests
  * 0.5 day for bundle work
  * 0.5 day for benchmark validation + dictionary decision
  * 0.5 day for UI affordances (corrections-applied counter)
  * **Total: ~2.5 days**

---

## Deferred items from Tiers A-C

These were in the original A-C plan but the work is larger than the
"quick improvement" positioning they had; they're listed here so the
roadmap is explicit.

### A3 — Ensemble preprocessing

Run 2-3 preprocessing variants (OTSU / Sauvola / raw grayscale) per
page, OCR each, and pick the variant with the highest mean confidence.
Expected: +5-10 % accuracy on ambiguous pages.

**Why deferred.** The merge path is the hard part: different
binarisations shift glyph bounding boxes by 1-2 px, so per-word
"winner takes all" needs bbox alignment that the current pipeline
doesn't do. Either pick a per-page winner (simpler, less accurate)
or build a glyph-level alignment layer (more accurate, much more
code). Before committing either way we need the Tier D benchmark
numbers to justify the complexity.

**Sketch:**
- New ``OCRConfig.ensemble_preprocessing: bool = False``.
- In pipeline, after page preprocess, run N variants concurrently.
- Each variant produces a full ``PageResult``.
- Merge rule (v1): pick the variant with the highest mean confidence,
  use its text verbatim.
- Merge rule (v2, deferred to D-tier): per-word confidence alignment
  by bbox IoU ≥ 0.7.

**Estimate:** 2 days for v1, 4 more days for v2.

### B3 — Per-block PSM dispatch

Layout-aware PSM: each Tesseract ``block_num`` from ``image_to_data``
gets its own PSM (``SINGLE_BLOCK`` for tables, ``AUTO`` for paragraphs,
``SINGLE_LINE`` for headers). Expected: +3-7 % accuracy on
table-heavy contracts.

**Why deferred.** Requires two OCR passes: a first pass with
``-c tessedit_pageseg_mode=3`` just to get the block boundaries, then
N single-block passes with the optimal PSM. Doubles or triples per-page
OCR time on pages where the layout is simple (nothing to gain). Needs
a heuristic to decide "is the layout complex enough to be worth the
two-pass cost" before it's a default-on feature.

**Sketch:**
- New ``OCRConfig.per_block_psm_dispatch: bool = False``.
- After the layout-analysis pass, enumerate blocks, classify each as
  table / paragraph / heading / line by bbox aspect + word density.
- Dispatch each block crop to a per-block OCR call with the optimal
  PSM.
- Re-assemble into the original page ``PageResult`` preserving
  block_num ordering.

**Estimate:** 2.5 days, plus 1 day for the classifier training set.

---

## Suggested order

1. **Measurement first.** Run ``scripts/benchmark_matrix.py`` on the
   user's real documents with ground-truth typed up. This tells us
   which items actually matter for the user's workload before we
   invest engineering hours in D1-D3.
2. **D1 (Tesseract 5.3) next.** Smallest change, biggest breadth of
   effect, zero runtime cost increase.
3. **A3 or B3 based on the benchmark.** If table-heavy pages dominate
   the CER → B3. If the CER is uniform across page types → A3.
4. **D2 / D3 last.** Both are expensive to ship; both need the Tier 1
   benchmark to justify.
