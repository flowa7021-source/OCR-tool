"""Test :meth:`TextPostprocessor._validate_identifiers` wiring.

Unit coverage for the pure validators lives in
``test_doc_validators.py`` and for the catalog loader in
``test_doc_catalog.py``. Here we verify the postprocessor actually
calls through: with ``config.validate_identifiers=True`` AND a
non-empty catalog, corrupt identifiers in the OCR output get
rewritten to their canonical form. With the flag off or the catalog
empty, the text passes through unchanged.
"""

from __future__ import annotations

from src.core.doc_catalog import DocCatalog
from src.core.models import PostprocessConfig
from src.core.text_postprocessor import TextPostprocessor

_CATALOG = DocCatalog(
    inns=frozenset({"7813266190", "7707820890"}),
    ogrns=frozenset({"5137746157490"}),
)


def _cfg(*, validate: bool = True) -> PostprocessConfig:
    """Produce a minimal config that enables identifier validation
    but disables every other step so we can assert only the
    validation effect."""
    return PostprocessConfig(
        autocorrect_russian=False,
        autocorrect_english=False,
        merge_hyphenated=False,
        normalize_whitespace=False,
        normalize_unicode=False,
        remove_artifacts=False,
        fix_cyrillic_latin_confusion=False,
        garbage_filter_strictness="disabled",
        validate_identifiers=validate,
    )


class TestIdentifierFixup:
    def test_corrupt_inn_is_rewritten_to_catalog_value(self) -> None:
        """The critical user-visible win: a 1-digit-off ИНН in the
        OCR output gets replaced with the real company's ИНН from
        the catalog."""
        pp = TextPostprocessor(catalog=_CATALOG)
        text = "ООО ГЕКСАФОРМ ИНН 7813266199, далее"
        # 7813266199 is 1 edit from 7813266190 — unique catalog match.
        out = pp.process(text, _cfg())
        assert "7813266190" in out
        assert "7813266199" not in out

    def test_canonical_inn_passes_through_unchanged(self) -> None:
        pp = TextPostprocessor(catalog=_CATALOG)
        text = "ООО ГЕКСАФОРМ ИНН 7813266190"
        assert pp.process(text, _cfg()) == text

    def test_valid_but_not_in_catalog_passes_through(self) -> None:
        """A valid ИНН we've never seen before is probably a new
        counterparty, not a typo — leave it alone."""
        pp = TextPostprocessor(catalog=_CATALOG)
        # 5003131890 validates but isn't in our test catalog.
        text = "ИНН 5003131890"
        assert pp.process(text, _cfg()) == text

    def test_ambiguous_token_left_alone(self) -> None:
        """When a corrupt token has TWO 1-edit matches in the
        catalog, the fixup refuses to pick."""
        ambiguous_cat = DocCatalog(
            inns=frozenset({"7700000001", "7800000001"}),
        )
        pp = TextPostprocessor(catalog=ambiguous_cat)
        text = "ИНН 7900000001"  # 1 edit from EITHER
        assert pp.process(text, _cfg()) == text

    def test_ogrn_fixup_routes_to_ogrn_catalog(self) -> None:
        """13-digit tokens look for OGRN matches, not INN — the
        catalog-per-length routing prevents cross-type confusion."""
        pp = TextPostprocessor(catalog=_CATALOG)
        # 5137746157491 is 1 edit from real 5137746157490
        text = "ОГРН 5137746157491"
        out = pp.process(text, _cfg())
        assert "5137746157490" in out

    def test_flag_off_leaves_text_untouched(self) -> None:
        pp = TextPostprocessor(catalog=_CATALOG)
        text = "ИНН 7813266199"  # corrupt
        out = pp.process(text, _cfg(validate=False))
        # No rewrite when the profile hasn't opted in.
        assert out == text

    def test_empty_catalog_leaves_text_untouched(self) -> None:
        """No catalog data = no fixup, even with flag on."""
        pp = TextPostprocessor(catalog=DocCatalog())
        text = "ИНН 7813266199"
        assert pp.process(text, _cfg()) == text

    def test_no_catalog_instance_at_all(self) -> None:
        """TextPostprocessor constructed without a catalog — fixup
        is silently a no-op regardless of config."""
        pp = TextPostprocessor()  # default: catalog=None
        text = "ИНН 7813266199"
        assert pp.process(text, _cfg()) == text

    def test_multiple_tokens_on_one_line(self) -> None:
        """A paragraph with two corrupt identifiers — both get
        fixed in a single pass."""
        pp = TextPostprocessor(catalog=_CATALOG)
        text = "ИНН 7813266199 и ОГРН 5137746157491 в одной строке"
        out = pp.process(text, _cfg())
        assert "7813266190" in out and "5137746157490" in out
        assert "7813266199" not in out and "5137746157491" not in out

    def test_non_identifier_digit_runs_untouched(self) -> None:
        """10-digit phone numbers, product codes, etc. that aren't in
        the catalog and don't match any 1-edit entry stay as-is."""
        pp = TextPostprocessor(catalog=_CATALOG)
        text = "телефон 84951234567"  # 11 digits
        assert pp.process(text, _cfg()) == text
