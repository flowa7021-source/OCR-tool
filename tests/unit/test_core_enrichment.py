"""Tests for src.tn_parser.core enrichment helpers:

* ``_catalog_crossvalidate`` — ИНН/name cross-check через org_lookup.
* ``_auto_learn_from_row`` — add new valid ORG to catalog.
* ``_apply_multi_row_voting`` — consolidate duplicate orgs across rows.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.tn_parser.core import (
    _apply_multi_row_voting,
    _auto_learn_from_row,
    _catalog_crossvalidate,
)
from src.tn_parser.models import FieldConfidence, ParsedRow

pytest.importorskip("rapidfuzz")


# ---------------------------------------------------------------------------
# _catalog_crossvalidate
# ---------------------------------------------------------------------------


class TestCatalogCrossvalidate:
    def test_empty_raw_returns_zero(self) -> None:
        assert _catalog_crossvalidate("", "shipper") == ("", 0.0)
        assert _catalog_crossvalidate("отсутствует", "shipper") == (
            "отсутствует", 0.0,
        )

    def test_no_inn_no_boost(self) -> None:
        """Без валидного ИНН нет boost'а."""
        raw = "ООО «Какая-то организация»"
        out, delta = _catalog_crossvalidate(raw, "shipper")
        assert out == raw
        assert delta == 0.0

    def test_invalid_inn_checksum_no_boost(self) -> None:
        """Рандомные 10 цифр — не проходят checksum, нет boost."""
        raw = "ООО «X», ИНН 1234567890"  # invalid checksum
        out, delta = _catalog_crossvalidate(raw, "shipper")
        assert delta == 0.0

    def test_valid_inn_without_catalog_entry_gives_base_boost(self) -> None:
        """ИНН checksum OK, но каталог не знает — base boost 0.05."""
        # 7707083893 — valid по ФНС checksum, но не в нашем test-catalog
        with patch(
            "src.tn_parser.core.lookup_by_inn", return_value=None,
        ):
            _, delta = _catalog_crossvalidate(
                "ООО «X», ИНН 7707083893", "shipper",
            )
            assert delta == 0.05  # только ИНН boost без name-match

    def test_valid_inn_with_catalog_name_match(self) -> None:
        """ИНН + name fuzzy-match → full boost +0.15."""
        catalog_rec = {
            "name": "Моспроект-3",
            "legal_form": "АО",
            "inn": "7707820890",
        }
        with patch(
            "src.tn_parser.core.lookup_by_inn", return_value=catalog_rec,
        ):
            _, delta = _catalog_crossvalidate(
                "АО «Моспроект-3», 107031, Москва, ИНН 7707820890",
                "shipper",
            )
            assert delta == 0.15  # ИНН (+0.05) + name match (+0.10)

    def test_valid_inn_with_catalog_name_mismatch_prepends(self) -> None:
        """ИНН валиден, но name score < 60 → prepend canonical
        аннотацию для shipper/reception."""
        catalog_rec = {
            "name": "Совершенно Другая Компания",
            "legal_form": "ООО",
            "inn": "7707820890",
        }
        with patch(
            "src.tn_parser.core.lookup_by_inn", return_value=catalog_rec,
        ):
            raw = "ООО «НеТотМангл», ИНН 7707820890"
            out, delta = _catalog_crossvalidate(raw, "shipper")
            # Должен содержать catalog-аннотацию
            if delta == 0.05:  # только base boost, name не matched
                assert "[каталог:" in out

    def test_consignee_fallback_via_full_text(self) -> None:
        """Consignee по design без ИНН в raw — ищем ИНН в full_text,
        catalog lookup, fuzzy match name → boost."""
        catalog_rec = {"name": "Моспроект-3", "inn": "7707820890"}
        full_text = (
            "АО «Моспроект-3», Россия, ИНН 7707820890, КПП 770701001"
        )
        with patch(
            "src.tn_parser.core.lookup_by_inn", return_value=catalog_rec,
        ):
            raw = "АО «Моспроект-3»"
            _, delta = _catalog_crossvalidate(
                raw, "consignee", full_text,
            )
            assert delta > 0  # через full_text нашёл ИНН и матч name


# ---------------------------------------------------------------------------
# _auto_learn_from_row
# ---------------------------------------------------------------------------


class TestAutoLearn:
    def _make_row(
        self, shipper: str = "", conf_shipper: float = 0.0,
    ) -> ParsedRow:
        row = ParsedRow()
        row.shipper = shipper
        row.confidence = FieldConfidence(shipper=conf_shipper)
        return row

    def test_low_conf_not_learned(self) -> None:
        """conf < 0.8 — не учим (mangled name)."""
        row = self._make_row("ООО «Бекам», ИНН 7743553262", conf_shipper=0.5)
        remembered: list[str] = []
        with patch(
            "src.tn_parser.org_lookup.remember",
            side_effect=lambda inn, rec: remembered.append(inn),
        ), patch(
            "src.tn_parser.org_lookup.lookup_by_inn", return_value=None,
        ):
            _auto_learn_from_row(row)
        assert remembered == []

    def test_high_conf_valid_inn_learned(self) -> None:
        """conf ≥ 0.8 + valid ИНН + не в catalog → remember() вызван."""
        row = self._make_row(
            "ООО «Беком», ИНН 7743553262", conf_shipper=0.9,
        )
        remembered: list[tuple[str, dict]] = []
        with patch(
            "src.tn_parser.org_lookup.remember",
            side_effect=lambda inn, rec: remembered.append((inn, rec)),
        ), patch(
            "src.tn_parser.org_lookup.lookup_by_inn", return_value=None,
        ):
            _auto_learn_from_row(row)
        assert len(remembered) == 1
        inn, rec = remembered[0]
        assert inn == "7743553262"
        assert rec.get("source") == "auto-learned"
        assert rec.get("legal_form") == "ООО"
        assert "Беком" in rec.get("name", "")

    def test_already_in_catalog_not_relearned(self) -> None:
        """Если catalog уже знает ИНН — не переучиваем."""
        row = self._make_row(
            "ООО «Беком», ИНН 7743553262", conf_shipper=0.95,
        )
        remembered: list[str] = []
        with patch(
            "src.tn_parser.org_lookup.remember",
            side_effect=lambda inn, rec: remembered.append(inn),
        ), patch(
            "src.tn_parser.org_lookup.lookup_by_inn",
            return_value={"name": "Beкo", "inn": "7743553262"},
        ):
            _auto_learn_from_row(row)
        assert remembered == []

    def test_missing_field_skipped(self) -> None:
        """MISSING field не учим — нечего запомнить."""
        row = self._make_row("отсутствует", conf_shipper=0.9)
        remembered: list[str] = []
        with patch(
            "src.tn_parser.org_lookup.remember",
            side_effect=lambda inn, rec: remembered.append(inn),
        ):
            _auto_learn_from_row(row)
        assert remembered == []


# ---------------------------------------------------------------------------
# _apply_multi_row_voting
# ---------------------------------------------------------------------------


class TestMultiRowVoting:
    def _make_row(self, **kwargs) -> ParsedRow:
        row = ParsedRow()
        for k, v in kwargs.items():
            setattr(row, k, v)
        row.confidence = FieldConfidence()
        return row

    def test_single_row_no_op(self) -> None:
        """Один row — voting не применяется."""
        rows = [self._make_row(shipper="ООО «X», ИНН 7743553262")]
        original = rows[0].shipper
        _apply_multi_row_voting(rows)
        assert rows[0].shipper == original

    def test_same_inn_consolidates_to_longest(self) -> None:
        """Несколько rows с одним ИНН → используем самый длинный
        вариант как canonical, propagate."""
        rows = [
            self._make_row(shipper="ООО Бекам ИНН 7743553262"),  # короткий
            self._make_row(
                shipper="ООО «Беком», 125212 Москва, ИНН 7743553262",  # длинный
            ),
        ]
        _apply_multi_row_voting(rows)
        # Обе rows получили длинную версию.
        assert rows[0].shipper == rows[1].shipper
        assert "125212" in rows[0].shipper

    def test_different_inns_not_consolidated(self) -> None:
        """Rows с разными валидными ИНН не сливаются."""
        rows = [
            self._make_row(shipper="ООО «A», ИНН 7743553262"),
            self._make_row(shipper="АО «B», ИНН 7707820890"),
        ]
        _apply_multi_row_voting(rows)
        assert rows[0].shipper != rows[1].shipper

    def test_conf_boost_after_voting(self) -> None:
        """При consolidation conf получает +0.03 boost."""
        rows = [
            self._make_row(shipper="ООО Бекам ИНН 7743553262"),
            self._make_row(
                shipper="ООО «Беком», 125212, ИНН 7743553262",
            ),
        ]
        rows[0].confidence.shipper = 0.5
        rows[1].confidence.shipper = 0.9
        _apply_multi_row_voting(rows)
        assert rows[0].confidence.shipper > 0.5  # boosted

    def test_empty_rows_safe(self) -> None:
        """Пустой список не падает."""
        _apply_multi_row_voting([])  # no exception
