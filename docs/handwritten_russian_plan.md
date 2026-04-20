# Handwritten Russian OCR — implementation plan

Current state: Tesseract 5.3 + LSTM on ``rus+eng`` reads **printed**
Russian text with 85–90 % confidence on real transport invoices.
Handwriting is a different problem — Tesseract's Russian model was
not trained on handwritten samples and returns mostly-noise output
on handwritten fields.

This document plans how to add handwritten-Russian support without
repeating the GOT-OCR mistake (bundled a model, measured later,
removed it when real-world performance was 3 % vs Tesseract's
78 %).

## What handwriting looks like in the user's corpus

Scanned transport invoices (УПД / ТН) contain three kinds of
handwritten fields:

  * Signatures — recognition is out of scope; users care about
    PRESENCE/ABSENCE only ("есть подпись?").
  * Handwritten notes in margin / corners (dates, driver's
    initials, stamp overlays). Low-priority — users rarely need
    the exact text.
  * Handwritten form fields (driver's license number, delivery
    marks, cargo acceptance stamps). HIGH-priority — these are
    legally-binding accounting fields.

## Why Tesseract alone doesn't work

Empirically on the user's ``inputs/`` samples, Tesseract LSTM on
rus+eng returns gibberish for handwritten regions — 10-40 %
confidence random-character sequences (the CAPS-preservation hook
in confidence_filter correctly drops most of these as noise).

The LSTM was trained on printed type (TTF-rendered corpora); the
handwriting class has essentially zero training weight. No amount
of preprocessing will fix this — the model doesn't know what
handwritten ``д`` looks like.

## Candidate solutions

### 1. TrOCR (Microsoft, MIT License)

Transformer-based OCR (encoder-decoder Vision Transformer) with a
dedicated HTR (Handwritten Text Recognition) checkpoint. HuggingFace
hosts ``microsoft/trocr-large-handwritten`` trained on IAM, plus
community-tuned Russian-handwriting checkpoints.

**Pros:**
  * State-of-the-art HTR quality
  * Pure Python via ``transformers`` library — integrates cleanly
  * Apache-2.0 code, MIT-licensed weights
  * No GPU required for inference (slower on CPU, ~2-5 s per line)

**Cons:**
  * Model weights ~600 MB bundled — installer size impact similar
    to the GOT-OCR we just removed
  * Pinned to ``transformers`` and ``torch`` — heavy dependencies
  * Russian-handwritten checkpoints aren't officially from Microsoft;
    community-trained on fragment corpora of varying quality.
  * Needs line segmentation upstream — we can use Tesseract's
    layout analysis for that, but its bbox precision on handwritten
    regions is poor (see below).

### 2. PaddleOCR (Apache-2.0)

PaddleOCR bundles a rec model for handwritten Chinese/English by
default. Community ports exist for Russian handwriting (e.g.
``PP-OCRv4-rec-ru-handwritten``) but quality is unverified on
Cyrillic handwriting specifically.

**Pros:**
  * Production-grade detection + recognition pipeline
  * Smaller model size (~80-150 MB)
  * Handles line segmentation natively

**Cons:**
  * Same dependency cost as GOT-OCR was (PaddlePaddle ~500 MB)
  * Russian handwriting checkpoint is community-maintained, not
    officially benchmarked by Baidu
  * We evaluated PaddleOCR for printed Russian earlier — didn't
    beat Tesseract. Handwriting quality would need independent
    verification on the user's corpus.

### 3. Kraken (Apache-2.0)

Historical document OCR engine that accepts user-trained HTR
models via the community model repository (``htr-united``).

**Pros:**
  * ~50 MB bundle per model
  * Designed specifically for HTR workflows
  * Active academic-OCR community

**Cons:**
  * Requires Python + PyTorch
  * Russian models are scarce; users would often need to train one

### 4. Azure / Google / AWS Cloud HTR APIs

Out of scope for an on-premise tool. Not considered.

### 5. Detect-and-flag (no recognition)

Rather than transcribing, DETECT handwritten regions and flag them
for manual review. Users get a searchable PDF with "⟨handwritten
region, page 2 bbox (200, 400, 120, 40)⟩" markers.

**Pros:**
  * No model bundle, no new dependency, no training data
  * Plays to Tesseract's strength (it KNOWS when its output is
    noise — low confidence + high-density glyph density signals
    "probably handwritten")
  * User workflow integrates with existing manual entry
  * Measurable regressions limited to "we missed a handwritten
    region" — fail-open rather than fail-wrong

**Cons:**
  * No transcription. Users still have to type the handwritten
    content themselves.

## Recommendation

**Phase 1 (now) — Detect and flag.**
No new dependencies, no model weights. Implementation:

  * Post-OCR per-page analysis: for each Tesseract block with
    mean per-word confidence < 40 % AND more than N words, classify
    as "suspect handwritten".
  * Surface via the already-existing ``redact_noisy_blocks``
    mechanism (currently used for stamps / signatures) — just add
    a new flag like ``mark_suspect_handwritten_blocks`` that
    replaces the block text with ``⟨рукописный текст⟩``.
  * UI: results panel highlights these blocks; TXT / DOCX export
    preserves the markers.

Estimate: 0.5 day for the detection heuristic, 0.5 day for UI
wiring, 0.5 day for tests. **~1.5 days**.

**Phase 2 (later) — If users want transcription too.**
Evaluate TrOCR on the user's own handwritten samples. Measure
quality BEFORE committing to the bundle cost:

  1. User provides 10-20 handwritten fragments with ground truth.
  2. Run TrOCR + the "detect and flag" pipeline against them.
  3. Measure CER on the handwritten fragments specifically.
  4. Only if CER < 15 % do we proceed with the ~600 MB bundle
     cost (same discipline we should have applied to GOT-OCR).

Estimate: 1 day for the measurement harness, then conditional
2-3 days for full integration if the numbers justify it.
**1 day up-front + 2-3 contingent days**.

## What NOT to do (hard-won lessons)

1. Don't bundle a handwriting model before measuring it on the
   user's real samples. The GOT-OCR mistake cost us a whole
   development cycle.
2. Don't try to "fix" Tesseract for handwriting via preprocessing.
   Aggressive CLAHE / morphology makes handwriting LESS readable
   for a printed-text LSTM, not more.
3. Don't rely on cloud APIs. The app is on-premise by design.

## Honest expectation management for the user

Until Phase 1 ships, users should assume:

  * Printed text: 85-95 % confidence, high fact-recall on ИНН /
    ОГРН / amounts / dates.
  * Handwritten text: **not reliably transcribed**. Results panel
    may show garbage characters (10-40 % confidence, dropped by
    the word-conf filter in most cases).

The "detect and flag" phase closes half of this gap — the user
sees ``⟨рукописный текст⟩`` markers instead of silent gaps, so
they can decide whether to type the field manually. Phase 2
would close the other half but requires real-sample measurement
before committing engineering hours.
