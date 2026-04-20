"""Unit tests for :mod:`src.core.pdf_text_filter`.

The module redacts low-confidence word regions from a searchable
PDF's invisible text layer. The tests build small synthetic PDFs
with known text placement, hand a TSV dict with mixed confidences,
and assert that the post-filter PDF contains ONLY the high-conf
words when extracted via PyMuPDF's ``page.get_text("text")``.

We don't need real Tesseract here — the filter takes a pre-built
TSV, so ``pytesseract`` is a non-dependency. Only ``fitz`` (PyMuPDF)
is required; import-skip if absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fitz")

from src.core.pdf_text_filter import (  # noqa: E402
    filter_pdf_text_layer,
    find_noisy_blocks,
)


def _build_pdf_with_words(
    path: Path,
    *,
    words: list[tuple[str, float, float]],
    page_size: tuple[float, float] = (612.0, 792.0),
) -> Path:
    """Synthesize a 1-page PDF with every word in ``words`` laid out
    at its given ``(x, y)`` point coordinate. Returns the path.

    Simulates what OCRmyPDF's graft step produces: a PDF whose text
    layer is a sequence of positioned word strings. The layer is
    VISIBLE here (no render-mode trick) so PyMuPDF extraction cleanly
    round-trips through ``page.get_text``.
    """
    import fitz

    doc = fitz.open()
    try:
        page = doc.new_page(width=page_size[0], height=page_size[1])
        for word, x, y in words:
            page.insert_text((x, y), word, fontsize=12)
        doc.save(str(path))
    finally:
        doc.close()
    return path


def _pdf_text(path: Path) -> str:
    """Extract all visible text from page 1 of ``path``."""
    import fitz

    with fitz.open(str(path)) as doc:
        return doc.load_page(0).get_text("text") or ""


def _tsv(
    *rows: tuple[str, float, int, int, int, int],
    block: int = 1,
) -> dict[str, list]:
    """Build an ``image_to_data``-shaped dict from ``(text, conf,
    left_px, top_px, width_px, height_px)`` tuples. All rows live in
    one block by default; multi-block fixtures use :func:`_tsv_blocks`.
    """
    return {
        "text":      [r[0] for r in rows],
        "conf":      [r[1] for r in rows],
        "left":      [r[2] for r in rows],
        "top":       [r[3] for r in rows],
        "width":     [r[4] for r in rows],
        "height":    [r[5] for r in rows],
        "block_num": [block] * len(rows),
    }


def _tsv_blocks(
    *groups: tuple[int, list[tuple[str, float, int, int, int, int]]],
) -> dict[str, list]:
    """Build a multi-block TSV. Each ``(block_num, [rows...])`` group
    contributes rows tagged with that block id — lets a single test
    fixture cover "this page has two layout blocks, one noisy".
    """
    texts, confs, lefts, tops, widths, heights, blocks = (
        [], [], [], [], [], [], [],
    )
    for blk, rows in groups:
        for r in rows:
            texts.append(r[0])
            confs.append(r[1])
            lefts.append(r[2])
            tops.append(r[3])
            widths.append(r[4])
            heights.append(r[5])
            blocks.append(blk)
    return {
        "text":      texts,
        "conf":      confs,
        "left":      lefts,
        "top":       tops,
        "width":     widths,
        "height":    heights,
        "block_num": blocks,
    }


# ----------------------------------------------------------------------
# The happy-path end-to-end tests use a 612×792 pt page (US Letter).
# TSV bboxes are given in a 612×792-pixel image so the pt⇄px transform
# is the identity — keeps the test assertions readable without
# obscuring the filter's coordinate math. Non-identity transforms get
# their own dedicated test below.
# ----------------------------------------------------------------------


class TestFilterHappyPath:
    """Mixed-confidence words in one page: keep high, drop low."""

    def test_low_conf_word_is_removed(self, tmp_path: Path) -> None:
        pdf = _build_pdf_with_words(
            tmp_path / "in.pdf",
            words=[
                ("hello", 50.0, 100.0),   # high conf — keep
                ("TPAC", 200.0, 100.0),   # low conf — remove
                ("world", 300.0, 100.0),  # high conf — keep
            ],
        )
        # TSV coords parallel the PDF insert_text positions with a ~15 px
        # bbox. Pixel-space == point-space at 612×792 image size.
        tsv = _tsv(
            ("hello", 92.0,  50,  90, 40, 15),
            ("TPAC",  18.0,  200, 90, 40, 15),
            ("world", 90.0,  300, 90, 40, 15),
        )

        redacted = filter_pdf_text_layer(
            pdf,
            tsv_per_page=[tsv],
            image_sizes_px=[(612, 792)],
            min_confidence=60.0,
        )

        assert redacted == 1, "exactly one low-conf word should be redacted"
        text = _pdf_text(pdf)
        assert "hello" in text
        assert "world" in text
        assert "TPAC" not in text, (
            f"low-conf 'TPAC' leaked through the filter: {text!r}"
        )

    def test_returns_zero_when_nothing_below_threshold(
        self, tmp_path: Path
    ) -> None:
        pdf = _build_pdf_with_words(
            tmp_path / "clean.pdf",
            words=[("alpha", 50.0, 100.0), ("beta", 150.0, 100.0)],
        )
        original = _pdf_text(pdf)

        redacted = filter_pdf_text_layer(
            pdf,
            tsv_per_page=[
                _tsv(
                    ("alpha", 95.0, 50, 90, 40, 15),
                    ("beta",  97.0, 150, 90, 40, 15),
                )
            ],
            image_sizes_px=[(612, 792)],
            min_confidence=60.0,
        )
        assert redacted == 0
        # Text layer unchanged byte-for-byte.
        assert _pdf_text(pdf) == original


class TestFilterCoordinateTransform:
    """Pixel-space TSV coords → PDF user-space points transform."""

    def test_half_size_image_still_aligns(self, tmp_path: Path) -> None:
        """TSV coords are in a 306×396-pixel image (half the 612×792
        PDF page dimensions). The filter must scale bboxes ×2 in both
        axes before issuing the redaction."""
        pdf = _build_pdf_with_words(
            tmp_path / "half.pdf",
            words=[
                ("keep", 50.0, 100.0),
                ("noise", 300.0, 500.0),
            ],
        )
        # TSV coords are HALF the PDF coords.
        tsv = _tsv(
            ("keep",  90.0,   25,  45, 20,  8),  # → 50,90,40,16 pt
            ("noise", 15.0,  150, 245, 20,  8),  # → 300,490,40,16 pt
        )

        redacted = filter_pdf_text_layer(
            pdf,
            tsv_per_page=[tsv],
            image_sizes_px=[(306, 396)],
            min_confidence=50.0,
        )

        assert redacted == 1
        text = _pdf_text(pdf)
        assert "keep" in text
        assert "noise" not in text


class TestFilterSafety:
    """Edge cases that must not crash or silently misalign."""

    def test_length_mismatch_raises(self, tmp_path: Path) -> None:
        pdf = _build_pdf_with_words(
            tmp_path / "x.pdf", words=[("a", 50.0, 100.0)],
        )
        with pytest.raises(ValueError, match="length"):
            filter_pdf_text_layer(
                pdf,
                tsv_per_page=[_tsv(), _tsv()],  # 2 entries
                image_sizes_px=[(612, 792)],     # 1 entry — mismatch
                min_confidence=50.0,
            )

    def test_page_count_mismatch_raises(self, tmp_path: Path) -> None:
        """A TSV list longer than the PDF page count is a programmer
        bug — we refuse to silently mis-align rather than apply the
        wrong TSV to each page."""
        pdf = _build_pdf_with_words(
            tmp_path / "one.pdf", words=[("only", 50.0, 100.0)],
        )
        with pytest.raises(ValueError, match="page"):
            filter_pdf_text_layer(
                pdf,
                tsv_per_page=[_tsv(), _tsv()],   # 2 pages
                image_sizes_px=[(612, 792), (612, 792)],
                min_confidence=50.0,
            )

    def test_empty_tsv_is_a_noop(self, tmp_path: Path) -> None:
        pdf = _build_pdf_with_words(
            tmp_path / "x.pdf", words=[("survives", 50.0, 100.0)],
        )
        before = _pdf_text(pdf)
        redacted = filter_pdf_text_layer(
            pdf,
            tsv_per_page=[_tsv()],
            image_sizes_px=[(612, 792)],
            min_confidence=60.0,
        )
        assert redacted == 0
        assert _pdf_text(pdf) == before

    def test_negative_conf_placeholder_rows_ignored(
        self, tmp_path: Path
    ) -> None:
        """Tesseract emits conf=-1 for level/block/par/line marker
        rows that carry no word text. Those must never be treated as
        "low-confidence words to redact" regardless of bbox."""
        pdf = _build_pdf_with_words(
            tmp_path / "x.pdf", words=[("keep", 50.0, 100.0)],
        )
        tsv = _tsv(
            ("marker", -1.0, 50, 90, 40, 15),  # placeholder row
            ("keep",   95.0, 50, 90, 40, 15),
        )
        redacted = filter_pdf_text_layer(
            pdf,
            tsv_per_page=[tsv],
            image_sizes_px=[(612, 792)],
            min_confidence=60.0,
        )
        assert redacted == 0
        assert "keep" in _pdf_text(pdf)

    def test_zero_image_dimensions_skips_page_gracefully(
        self, tmp_path: Path
    ) -> None:
        """An erroring page's placeholder slot has (0, 0) dims. The
        filter must skip that page, not divide-by-zero."""
        pdf = _build_pdf_with_words(
            tmp_path / "x.pdf", words=[("keep", 50.0, 100.0)],
        )
        redacted = filter_pdf_text_layer(
            pdf,
            tsv_per_page=[_tsv(("noise", 10.0, 0, 0, 10, 10))],
            image_sizes_px=[(0, 0)],
            min_confidence=50.0,
        )
        assert redacted == 0
        assert "keep" in _pdf_text(pdf)

    def test_module_exports_public_surface(self) -> None:
        """Tripwire: ``__all__`` must contain exactly the public helpers
        documented in the module docstring + tested here. If someone
        adds a third public function without updating this list, this
        fails loudly."""
        import src.core.pdf_text_filter as m

        assert set(m.__all__) == {"filter_pdf_text_layer", "find_noisy_blocks"}


class TestMultiPage:
    """TSV + image_sizes parallel lists must apply per-page correctly."""

    def test_different_tsv_per_page(self, tmp_path: Path) -> None:
        """Two-page PDF: page 1 has low-conf "foo", page 2 has low-
        conf "bar". After filtering, neither survives but the
        high-conf words on each page do."""
        import fitz

        p = tmp_path / "two.pdf"
        doc = fitz.open()
        try:
            p1 = doc.new_page(width=612, height=792)
            p1.insert_text((50, 100), "keep1", fontsize=12)
            p1.insert_text((200, 100), "foo", fontsize=12)
            p2 = doc.new_page(width=612, height=792)
            p2.insert_text((50, 100), "keep2", fontsize=12)
            p2.insert_text((200, 100), "bar", fontsize=12)
            doc.save(str(p))
        finally:
            doc.close()

        redacted = filter_pdf_text_layer(
            p,
            tsv_per_page=[
                _tsv(
                    ("keep1", 95.0, 50,  90, 40, 15),
                    ("foo",   20.0, 200, 90, 40, 15),
                ),
                _tsv(
                    ("keep2", 95.0, 50,  90, 40, 15),
                    ("bar",   15.0, 200, 90, 40, 15),
                ),
            ],
            image_sizes_px=[(612, 792), (612, 792)],
            min_confidence=50.0,
        )
        assert redacted == 2

        with fitz.open(str(p)) as d:
            page1_text = d.load_page(0).get_text("text") or ""
            page2_text = d.load_page(1).get_text("text") or ""
        assert "keep1" in page1_text and "foo" not in page1_text
        assert "keep2" in page2_text and "bar" not in page2_text


# ---------------------------------------------------------------------------
# find_noisy_blocks — unit tests for the pure helper
# ---------------------------------------------------------------------------


class TestFindNoisyBlocks:
    """Block-level "majority noise" detection from the TSV."""

    def test_empty_tsv_returns_empty_list(self) -> None:
        assert find_noisy_blocks({}, min_confidence=60.0) == []

    def test_block_below_min_words_is_ignored(self) -> None:
        """A 2-word block is too small-sample to trust its ratio —
        page numbers / captions often hit 100 % noise by chance."""
        tsv = _tsv_blocks(
            (1, [
                ("junk1", 10.0, 0, 0, 10, 10),
                ("junk2", 15.0, 0, 15, 10, 10),
            ]),
        )
        assert find_noisy_blocks(
            tsv, min_confidence=60.0, min_block_words=3,
        ) == []

    def test_noisy_block_returned_bbox_encloses_all_words(self) -> None:
        """Majority-noise block: bbox spans from the leftmost/top-most
        word to the rightmost/bottom-most."""
        tsv = _tsv_blocks(
            (1, [
                ("real",  90.0, 100, 100, 40, 15),
                ("junk1", 10.0, 200, 100, 40, 15),
                ("junk2", 15.0, 300, 120, 40, 15),
                ("junk3", 20.0, 400, 110, 40, 15),
            ]),
        )
        noisy = find_noisy_blocks(tsv, min_confidence=60.0)
        # 3 / 4 words noisy = 75 % > 0.5 threshold
        assert len(noisy) == 1
        left, top, width, height = noisy[0]
        # Encloses all words: left=100 (from "real"), right=440
        # (from "junk3"), top=100, bottom=135.
        assert left == 100
        assert top == 100
        assert width == 340   # 440 - 100
        assert height == 35   # 135 - 100

    def test_clean_block_not_returned(self) -> None:
        tsv = _tsv_blocks(
            (1, [
                ("body1", 95.0, 100, 100, 40, 15),
                ("body2", 92.0, 150, 100, 40, 15),
                ("body3", 90.0, 200, 100, 40, 15),
                ("body4", 15.0, 250, 100, 40, 15),  # 1/4 noise, < 50%
            ]),
        )
        assert find_noisy_blocks(tsv, min_confidence=60.0) == []

    def test_multi_block_only_noisy_ones_returned(self) -> None:
        """Page has two blocks. Block 1 clean, block 2 is a stamp
        zone. Only block 2 should be flagged."""
        tsv = _tsv_blocks(
            (1, [
                ("Hello", 95.0, 50,  50, 60, 15),
                ("World", 95.0, 120, 50, 60, 15),
                ("body",  90.0, 180, 50, 60, 15),
                ("text",  92.0, 250, 50, 60, 15),
            ]),
            (2, [
                ("noise1", 20.0, 500, 800, 50, 15),
                ("noise2", 15.0, 560, 800, 50, 15),
                ("noise3", 10.0, 620, 800, 50, 15),
                ("noise4", 25.0, 680, 800, 50, 15),
            ]),
        )
        noisy = find_noisy_blocks(tsv, min_confidence=60.0)
        assert len(noisy) == 1
        left, top, _, _ = noisy[0]
        # The returned bbox is block 2's (stamp zone), not block 1's.
        assert left == 500
        assert top == 800

    def test_noise_ratio_threshold_boundary(self) -> None:
        """Exactly at the ratio threshold counts as "noisy"."""
        # 2/4 = 50% exactly — at the threshold, included.
        tsv = _tsv_blocks(
            (1, [
                ("a", 95.0, 0,  0, 10, 10),
                ("b", 90.0, 20, 0, 10, 10),
                ("c", 20.0, 40, 0, 10, 10),
                ("d", 15.0, 60, 0, 10, 10),
            ]),
        )
        assert len(find_noisy_blocks(
            tsv, min_confidence=60.0, noise_ratio=0.5,
        )) == 1

    def test_custom_noise_ratio(self) -> None:
        """Stricter ratio (e.g. 0.8) lets more blocks through."""
        tsv = _tsv_blocks(
            (1, [
                ("a", 95.0, 0,  0, 10, 10),
                ("b", 20.0, 20, 0, 10, 10),
                ("c", 15.0, 40, 0, 10, 10),
                ("d", 10.0, 60, 0, 10, 10),
            ]),
        )
        # 3/4 = 75% — noisy at default 0.5, NOT noisy at 0.8.
        assert len(find_noisy_blocks(
            tsv, min_confidence=60.0, noise_ratio=0.5,
        )) == 1
        assert len(find_noisy_blocks(
            tsv, min_confidence=60.0, noise_ratio=0.8,
        )) == 0


# ---------------------------------------------------------------------------
# filter_pdf_text_layer — block-level redaction end-to-end
# ---------------------------------------------------------------------------


class TestBlockRedaction:
    """``redact_noisy_blocks=True`` wipes the whole block, not just
    the individually-low-conf words inside it."""

    def test_borderline_conf_word_in_noisy_block_is_also_redacted(
        self, tmp_path: Path
    ) -> None:
        """A word with conf=70 (above threshold=60) but sitting inside
        a 75 %-noise block gets swept out with the rest of the block.
        Without this, a borderline-confident 'ТРАНСПОРТ' stamped
        inside a stamp rectangle would sneak through the per-word
        filter."""
        pdf = _build_pdf_with_words(
            tmp_path / "stamp.pdf",
            words=[
                ("body",    50.0, 100.0),
                ("stamp1",  100.0, 400.0),
                ("stamp2",  160.0, 400.0),
                ("stamp3",  220.0, 400.0),
                ("STAMP70", 280.0, 400.0),  # inside stamp block, conf=70
            ],
        )
        tsv = _tsv_blocks(
            (1, [("body", 92.0, 50, 90, 40, 15)]),
            # Block 2: 3/4 noise (75 %), above the 50 % ratio threshold.
            (2, [
                ("stamp1",  20.0, 100, 390, 40, 15),
                ("stamp2",  15.0, 160, 390, 40, 15),
                ("stamp3",  25.0, 220, 390, 40, 15),
                ("STAMP70", 70.0, 280, 390, 60, 15),  # borderline
            ]),
        )
        redacted = filter_pdf_text_layer(
            pdf,
            tsv_per_page=[tsv],
            image_sizes_px=[(612, 792)],
            min_confidence=60.0,
            redact_noisy_blocks=True,
        )
        # 3 per-word (stamp1/2/3) + 1 per-block = 4 redactions.
        assert redacted == 4
        text = _pdf_text(pdf)
        assert "body" in text, "clean block was preserved"
        assert "stamp1" not in text
        assert "stamp2" not in text
        assert "stamp3" not in text
        # Critical: the borderline-conf word that survived per-word
        # filtering is nevertheless gone because its block was noisy.
        assert "STAMP70" not in text, (
            "borderline-conf word inside stamp block leaked through"
        )

    def test_flag_off_leaves_borderline_words(self, tmp_path: Path) -> None:
        """Default behaviour (flag off) keeps the borderline word —
        backwards-compat for profiles that haven't opted in."""
        pdf = _build_pdf_with_words(
            tmp_path / "stamp.pdf",
            words=[
                ("body",    50.0, 100.0),
                ("stamp1",  100.0, 400.0),
                ("stamp2",  160.0, 400.0),
                ("STAMP70", 220.0, 400.0),
            ],
        )
        tsv = _tsv_blocks(
            (1, [("body", 92.0, 50, 90, 40, 15)]),
            (2, [
                ("stamp1",  20.0, 100, 390, 40, 15),
                ("stamp2",  15.0, 160, 390, 40, 15),
                ("STAMP70", 70.0, 220, 390, 60, 15),
            ]),
        )
        redacted = filter_pdf_text_layer(
            pdf,
            tsv_per_page=[tsv],
            image_sizes_px=[(612, 792)],
            min_confidence=60.0,
            redact_noisy_blocks=False,
        )
        assert redacted == 2  # only per-word (stamp1, stamp2)
        text = _pdf_text(pdf)
        assert "STAMP70" in text, (
            "flag off → borderline word survives, as expected"
        )
