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

from src.core.pdf_text_filter import filter_pdf_text_layer  # noqa: E402


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
) -> dict[str, list]:
    """Build an ``image_to_data``-shaped dict from ``(text, conf,
    left_px, top_px, width_px, height_px)`` tuples."""
    return {
        "text":   [r[0] for r in rows],
        "conf":   [r[1] for r in rows],
        "left":   [r[2] for r in rows],
        "top":    [r[3] for r in rows],
        "width":  [r[4] for r in rows],
        "height": [r[5] for r in rows],
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

    def test_module_exports_filter_only(self) -> None:
        """Tripwire: if someone adds another public function without
        updating __all__ / the module docstring, this fails loudly."""
        import src.core.pdf_text_filter as m

        assert m.__all__ == ["filter_pdf_text_layer"]


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
