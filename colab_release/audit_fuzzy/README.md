# Fuzzy-match audit

Sampled 30 crop(s) where the bootstrap's fuzzy match replaced the raw EasyOCR output with a GT token at Levenshtein distance ≥ 2. For each row, **look at the crop** and decide whether `label_after_correction` is actually what's printed on the crop:

- If **yes** — the fuzzy match was correct (OCR misread a letter, the GT word rescued it). Training on this is fine.
- If **no** — the match hijacked the label to a similar but wrong word. This crop should be dropped (or relabelled). The more of these you see, the more false-positive noise in the training set.

| # | Crop | Raw OCR | Label | Lev | PDF |
|---|------|---------|-------|-----|-----|
| 0 | <img src="crops/000_TN_k_UPD_36_ot_02.09.2022_p0_b15.jpg" height="40"> | `Экземпляр W` | `экземпляр` | 2 | TN_k_UPD_36_ot_02.09.2022 p0 |
| 1 | <img src="crops/001_TN_k_UPD_36_ot_02.09.2022_p0_b48.jpg" height="40"> | `нестак` | `местах` | 2 | TN_k_UPD_36_ot_02.09.2022 p0 |
| 2 | <img src="crops/002_TN_k_UPD_36_ot_02.09.2022_p2_b8.jpg" height="40"> | `{Научно-исследовательскпй` | `научно-исследовательский` | 2 | TN_k_UPD_36_ot_02.09.2022 p2 |
| 3 | <img src="crops/003_TN_k_UPD_36_ot_02.09.2022_p2_b68.jpg" height="40"> | `значетшя` | `значения` | 2 | TN_k_UPD_36_ot_02.09.2022 p2 |
| 4 | <img src="crops/004_TN_k_UPD_36_ot_02.09.2022_p2_b80.jpg" height="40"> | `уебование-накладную` | `требование-накладную` | 2 | TN_k_UPD_36_ot_02.09.2022 p2 |
| 5 | <img src="crops/005_TN_k_UPD_41_ot_06.09.2022_p0_b21.jpg" height="40"> | `рузоотправнтель` | `грузоотправитель` | 2 | TN_k_UPD_41_ot_06.09.2022 p0 |
| 6 | <img src="crops/006_TN_k_UPD_41_ot_06.09.2022_p0_b50.jpg" height="40"> | `~именование` | `наименование` | 2 | TN_k_UPD_41_ot_06.09.2022 p0 |
| 7 | <img src="crops/007_TN_k_UPD_41_ot_06.09.2022_p0_b134.jpg" height="40"> | `Тягач с` | `тягач` | 2 | TN_k_UPD_41_ot_06.09.2022 p0 |
| 8 | <img src="crops/008_TN_k_UPD_41_ot_06.09.2022_p1_b81.jpg" height="40"> | `01.09.26/22` | `01.09.2022` | 2 | TN_k_UPD_41_ot_06.09.2022 p1 |
| 9 | <img src="crops/009_TN_k_UPD_41_ot_06.09.2022_p1_b136.jpg" height="40"> | `Отмстки` | `отметка` | 2 | TN_k_UPD_41_ot_06.09.2022 p1 |
| 10 | <img src="crops/010_TN_k_UPD_41_ot_06.09.2022_p2_b21.jpg" height="40"> | `22,391` | `2239` | 2 | TN_k_UPD_41_ot_06.09.2022 p2 |
| 11 | <img src="crops/011_TN_k_UPD_41_ot_06.09.2022_p2_b136.jpg" height="40"> | `Тягач c` | `тягач` | 2 | TN_k_UPD_41_ot_06.09.2022 p2 |
| 12 | <img src="crops/012_TN_k_UPD_41_ot_06.09.2022_p3_b42.jpg" height="40"> | `Кладовщнк'` | `кладовщик` | 2 | TN_k_UPD_41_ot_06.09.2022 p3 |
| 13 | <img src="crops/013_TN_k_UPD_41_ot_06.09.2022_p3_b67.jpg" height="40"> | `01,09,2022` | `01.09.2022` | 2 | TN_k_UPD_41_ot_06.09.2022 p3 |
| 14 | <img src="crops/014_TN_k_UPD_41_ot_06.09.2022_p4_b47.jpg" height="40"> | `анменование` | `наименование` | 2 | TN_k_UPD_41_ot_06.09.2022 p4 |
| 15 | <img src="crops/015_TN_k_UPD_41_ot_06.09.2022_p4_b92.jpg" height="40"> | `6 Перевозчик` | `перевозчик` | 2 | TN_k_UPD_41_ot_06.09.2022 p4 |
| 16 | <img src="crops/016_TN_k_UPD_41_ot_06.09.2022_p4_b101.jpg" height="40"> | `Перевозчика)` | `перевозчик` | 2 | TN_k_UPD_41_ot_06.09.2022 p4 |
| 17 | <img src="crops/017_TN_k_UPD_41_ot_06.09.2022_p4_b105.jpg" height="40"> | `Тягач c` | `тягач` | 2 | TN_k_UPD_41_ot_06.09.2022 p4 |
| 18 | <img src="crops/018_TN_k_UPD_41_ot_06.09.2022_p5_b109.jpg" height="40"> | `(подпнсь` | `подпись` | 2 | TN_k_UPD_41_ot_06.09.2022 p5 |
| 19 | <img src="crops/019_TN_k_UPD_41_ot_06.09.2022_p6_b10.jpg" height="40"> | `(109.2022` | `01.09.2022` | 2 | TN_k_UPD_41_ot_06.09.2022 p6 |
| 20 | <img src="crops/020_TN_k_UPD_41_ot_06.09.2022_p6_b17.jpg" height="40"> | `Экземпляр N` | `экземпляр` | 2 | TN_k_UPD_41_ot_06.09.2022 p6 |
| 21 | <img src="crops/021_TN_k_UPD_41_ot_06.09.2022_p6_b55.jpg" height="40"> | `трузон` | `грузов` | 2 | TN_k_UPD_41_ot_06.09.2022 p6 |
| 22 | <img src="crops/022_TN_k_UPD_41_ot_06.09.2022_p6_b139.jpg" height="40"> | `Автопрафит"` | `автопрофит` | 2 | TN_k_UPD_41_ot_06.09.2022 p6 |
| 23 | <img src="crops/023_TN_k_UPD_41_ot_06.09.2022_p6_b150.jpg" height="40"> | `Тягач c` | `тягач` | 2 | TN_k_UPD_41_ot_06.09.2022 p6 |
| 24 | <img src="crops/024_TN_k_UPD_41_ot_06.09.2022_p8_b17.jpg" height="40"> | `95ЗЗ/Б` | `9507/б` | 2 | TN_k_UPD_41_ot_06.09.2022 p8 |
| 25 | <img src="crops/025_TN_k_UPD_41_ot_06.09.2022_p8_b19.jpg" height="40"> | `(2,09.2022` | `02.09.2022` | 2 | TN_k_UPD_41_ot_06.09.2022 p8 |
| 26 | <img src="crops/026_TN_k_UPD_41_ot_06.09.2022_p8_b145.jpg" height="40"> | `Тягач c` | `тягач` | 2 | TN_k_UPD_41_ot_06.09.2022 p8 |
| 27 | <img src="crops/027_TN_k_UPD_41_ot_06.09.2022_p9_b15.jpg" height="40"> | `02,09,2022` | `02.09.2022` | 2 | TN_k_UPD_41_ot_06.09.2022 p9 |
| 28 | <img src="crops/028_TN_k_UPD_41_ot_06.09.2022_p9_b87.jpg" height="40"> | `(2.09.20/22` | `02.09.2022` | 2 | TN_k_UPD_41_ot_06.09.2022 p9 |
| 29 | <img src="crops/029_TN_k_UPD_41_ot_06.09.2022_p10_b9.jpg" height="40"> | `зявка)` | `заявка` | 2 | TN_k_UPD_41_ot_06.09.2022 p10 |
