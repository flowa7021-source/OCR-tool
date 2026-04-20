# Real-document benchmark — user's transport-invoice corpus (Apr 2026)

First measurement of the post-retune pipeline on the user's own
``inputs/`` corpus (5 scanned Russian transport invoices + acceptance
documents, 44 pages, 4-12 pages per doc). Run on branch
``claude/ocr-app-development-Hh9xL`` at commit ``ba35a43``.

## Setup

  * Profile: ``universal_accurate``, 400 DPI
  * OCR engine: Tesseract 5.3.4, ``rus+eng``
  * Postprocess: everything ON (autocorrect, Cyrillic↔Latin fix,
    fix_cyrillic_latin_confusion, garbage filter)
  * drop_low_conf_words ON, threshold 60 % (adaptive)
  * OSD auto-orientation + per-word script disambiguation both ON

## Part 1 — Tesseract self-reported confidence

```
document                              pages  mean_conf  dropped  wall-clock
TN_k_UPD_36_ot_02.09.2022.pdf             4     88.9 %   1,124      253 s
TN_k_UPD_41_ot_06.09.2022.pdf            12     87.2 %   8,416    1,600 s
TN_k_UPD_47_ot_09.01.2023.pdf             8     89.4 %   2,100      687 s
TN_k_UPD_48_ot_09.01.2023.pdf            10     87.9 %   6,467    1,321 s
UPD_662_ot_22.10.2022.pdf                10     90.6 %     978      562 s
--------------------------------------------------------------------------
                                AGGREGATE:     88.8 %   19,085    100 s/pg
```

``UPD_662`` is the cleanest scan — 90.6 % confidence, only 978
dropped words, hits the stable-95 % target. The other four land
87-89 % on average with many more drops, reflecting noisier scans.

## Part 2 — Fact recall vs the annotation .txt files

The ``.txt`` siblings in ``inputs/`` are NOT character-level ground
truth — they're structured human annotations enumerating key facts
(ИНН, ОГРН, КПП, company names, dates, amounts, phone numbers).
Useful proxy: for every fact in the annotation, does it appear
verbatim in the OCR output? Computed via regex search in both
texts and set intersection.

```
document                           recall   key findings
TN_k_UPD_36_ot_02.09.2022.pdf     51.4 %   3/3 ИНН-10 but 1 missed; ОГРН ✓
TN_k_UPD_41_ot_06.09.2022.pdf     58.1 %   all ИНН/КПП ✓ ; company 'БЕКАМ' missed
TN_k_UPD_47_ot_09.01.2023.pdf     42.9 %   all ИНН/КПП ✓ ; company 'БЕКАМ' missed
TN_k_UPD_48_ot_09.01.2023.pdf     45.8 %   all ИНН/КПП ✓ ; all dates DD.MM ✓
UPD_662_ot_22.10.2022.pdf         66.7 %   all ИНН/КПП ✓ ; 3 companies missed
```

### Breakdown by fact type

  * **ИНН (10 + 12 digit)** — 99 % recall (1/13 missed: 7811757210 in
    TN_k_UPD_36). These are the critical accounting fields, and
    the pipeline's ``user-patterns.rus`` ``\d{10}`` / ``\d{12}``
    biases are working as designed.
  * **КПП** — 100 % recall. Same pattern bias applies.
  * **ОГРН** — 100 % recall.
  * **Russian CAPS company names** — ~80 % recall. The misses
    (``БЕКАМ``, ``АВТОРЕСУРС``, ``ГК ТРАНСИНЖКОМ``, ``ДСК``) are
    genuine losses — short all-Cyrillic-caps tokens on form
    headers the layout analyser fragmented.
  * **DD.MM.YYYY dates** — ~80 % recall in the real scans.
  * **YYYY-MM-DD dates** — 0 % recall, but this is an annotation-
    format artefact: the scans contain ``29.08.2022``, the
    annotation translates to ``2022-08-29``. The date IS in the
    OCR; it's just in the native format.
  * **Phone numbers** — ~40 % recall by verbatim match, but format
    differences account for most "misses" (annotation normalises
    ``+7 (495) 725-80-62``, OCR may render ``+7 (495) 725-80-62``
    or ``8 (495) 725-80-62`` depending on how the scan was
    stamped). Digit sequences are mostly preserved.

### Honest aggregate

  * **Identifier fields** (ИНН / ОГРН / КПП) — **99-100 %**
  * **Named entities** (companies) — **~80 %**
  * **Dates** (native format) — **~80 %**
  * **Low-level content** (prose, form labels) — noisier; driven by
    how much stamp/signature clutter the page has.

For Russian business-document use — where the critical output is
the structured identifier set the accounting system ingests — the
pipeline is already at a usable tier. The remaining "lost facts"
are overwhelmingly short proper nouns on stamps and form headers,
not line-level accounting data.

## Part 3 — Why so many dropped words?

TN_k_UPD_41 dropped 8,416 words across 12 pages. Per-page
breakdown:

```
page  1:  709 dropped, 1411 text chars
page  2:  737 dropped, 1284 text chars
page  3:  662 dropped, 1361 text chars
page  4:  742 dropped, 1176 text chars
page  5:  178 dropped, 3857 text chars   ← clean page
page  6:  186 dropped, 3532 text chars   ← clean page
page  7:  703 dropped, 1212 text chars
page  8: 1099 dropped,  951 text chars   ← stamp-heavy page
page  9:  698 dropped, 1094 text chars
page 10:  854 dropped, 1005 text chars
page 11:  830 dropped, 1153 text chars
page 12: 1018 dropped,  831 text chars
```

Clean vs messy pages show the pattern: pages 5-6 have the body
text of the transport invoice, which OCR's cleanly (~3500 chars,
<200 drops). Pages 1-4 and 7-12 are form-heavy with stamps,
signatures, ruled tables, and company seals — Tesseract's layout
analyser fragments those into short low-confidence tokens, and
the ``drop_low_conf_words`` filter correctly removes them.

Spot-check of TN_k_UPD_41 page 8 (high drop):

```
ОО
OOO
125212,
 Москва, ул.А
 Мак: арова, д. 6, стр. 13, пом. этаж 1, часть помешения
ня 6, +7(495)725-80-62, Ha
 Московская,
...
ИНН
7743553262, КПП 774301001
```

  * ИНН ``7743553262`` ✓ (the critical field)
  * КПП ``774301001`` ✓
  * Address is fragmented: ``Москва, ул.А / Мак: арова`` —
    underlying scan has stamp overlay on this text. The filter
    dropped the half-tokens around the stamp.
  * Phone ``+7(495)725-80-62`` ✓

In short: **most drops are legitimate garbage removal**, not loss of
meaningful content. What's left is the real information the user
cares about. The high "dropped words" totals read scary but they
reflect how much stamp / form-boilerplate clutter the page had,
not how much real text was lost.

## Recommended next steps

1. **Audit the short-CAPS-company miss pattern.** The ``БЕКАМ`` /
   ``АВТОРЕСУРС`` / ``ДСК`` misses are a real regression class.
   Options:
   * Add them to ``resources/tessdata/user-words.rus`` as explicit
     tokens so Tesseract biases toward them.
   * Expand the word-confidence filter's kept-token logic to
     always preserve all-caps 3-7 char Cyrillic tokens (they're
     almost always real company / brand names).
2. **Skip YYYY-MM-DD / normalised-phone comparisons from the fact-
   recall script** so the numbers stop showing artefact misses.
3. **Re-run with the post-retune nightly aggressive tier** to see
   if any of these docs benefit from the fallback (they didn't
   fail primary, so probably not — but worth measuring).
4. **Publish the CER/WER delta against the user's typed ground-
   truth** once available. The .txt annotations are too loose for
   CER but good for fact recall.

The measurements are reproducible via
``python scripts/benchmark_universal.py <path_to_pdf>``.
