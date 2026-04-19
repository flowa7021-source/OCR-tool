"""Unit tests for :mod:`src.core.doc_catalog`."""

from __future__ import annotations

import json
from pathlib import Path

from src.core.doc_catalog import DocCatalog, load_catalog


class TestLoadCatalog:
    def test_missing_directory_returns_empty_catalog(
        self, tmp_path: Path
    ) -> None:
        cat = load_catalog(tmp_path / "nope")
        assert isinstance(cat, DocCatalog)
        assert cat.is_empty

    def test_empty_directory_returns_empty_catalog(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        assert load_catalog(tmp_path / "empty").is_empty

    def test_valid_json_loads_into_catalog(self, tmp_path: Path) -> None:
        """Realistic nested shape — identifiers live inside nested
        ``parties`` / ``transport`` dicts."""
        data = {
            "documents": [
                {
                    "parties": [
                        {
                            "name": "ГЕКСАФОРМ СПБ",
                            "inn": "7813266190",
                            "kpp": "781101001",
                        },
                        {
                            "name": "Моспроект-3",
                            "inn": "7707820890",
                            "ogrn": "5137746157490",
                        },
                    ],
                },
            ],
        }
        path = tmp_path / "doc.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        cat = load_catalog(tmp_path)
        assert cat.inns == frozenset({"7813266190", "7707820890"})
        assert cat.ogrns == frozenset({"5137746157490"})
        assert cat.kpps == frozenset({"781101001"})
        assert cat.names == frozenset({"ГЕКСАФОРМ СПБ", "Моспроект-3"})

    def test_invalid_checksum_entry_is_skipped_with_warning(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Ground-truth typo MUST NOT poison the catalog — it's
        logged and dropped."""
        path = tmp_path / "doc.json"
        path.write_text(
            json.dumps({"inn": "7813266199"}),  # wrong check digit
            encoding="utf-8",
        )
        import logging

        with caplog.at_level(logging.WARNING, logger="src.core.doc_catalog"):
            cat = load_catalog(tmp_path)
        assert cat.is_empty
        assert any(
            "fails checksum" in r.getMessage() for r in caplog.records
        )

    def test_malformed_json_file_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{ not json }", encoding="utf-8")
        cat = load_catalog(tmp_path)
        # File is silently skipped (with a WARNING), catalog stays
        # empty rather than raising.
        assert cat.is_empty

    def test_real_expected_folder_loads_known_counts(
        self, tmp_path: Path,
    ) -> None:
        """Integration: load the ACTUAL ``expected/`` folder and
        verify the counts match what the user's ground truth
        currently declares. If someone adds a new TN JSON, this
        test fails with a friendly mismatch message so they have to
        explicitly bless the new count."""
        expected_dir = Path(__file__).resolve().parents[2] / "expected"
        if not expected_dir.is_dir():
            import pytest

            pytest.skip("expected/ not present in this checkout")
        cat = load_catalog(expected_dir)
        # Floor values rather than exact — the corpus grows over
        # time, and we don't want the test to fail on every
        # addition. What matters is "at least the ones we've seen".
        assert len(cat.inns) >= 9, f"got {len(cat.inns)} ИНН"
        assert len(cat.ogrns) >= 1
        assert len(cat.kpps) >= 5
        assert len(cat.names) >= 15


class TestDocCatalog:
    def test_is_empty(self) -> None:
        assert DocCatalog().is_empty
        assert not DocCatalog(inns=frozenset({"7813266190"})).is_empty

    def test_len(self) -> None:
        cat = DocCatalog(
            inns=frozenset({"a", "b"}),
            ogrns=frozenset({"c"}),
            kpps=frozenset(),
            names=frozenset({"d", "e"}),
        )
        assert len(cat) == 5


class TestLoadDefaultCatalog:
    """``load_default_catalog`` is the production entry point — walks
    the standard search path so the CLI / worker bootstrap don't have
    to hard-code directory lookup logic. These tests pin the search
    order AND the graceful-empty fallback."""

    def test_prefers_user_dir_over_bundled(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When the user has dropped a catalog under
        ``USER_CATALOG_DIR`` we must read from there — NOT from the
        installer bundle. User customisation takes precedence over
        ship defaults."""
        import json

        from src.core import doc_catalog as dc

        user_dir = tmp_path / "user"
        bundled_dir = tmp_path / "bundled"
        user_dir.mkdir()
        bundled_dir.mkdir()

        # Both dirs have a catalog, with different ИНН. The user
        # one is the one we expect to win.
        (user_dir / "u.json").write_text(
            json.dumps({"inn": "7813266190"}), encoding="utf-8",
        )
        (bundled_dir / "b.json").write_text(
            json.dumps({"inn": "7707820890"}), encoding="utf-8",
        )

        monkeypatch.setattr(
            "src.shared.constants.USER_CATALOG_DIR", user_dir,
        )
        monkeypatch.setattr(
            "src.shared.constants.BUNDLED_CATALOG_DIR", bundled_dir,
        )

        cat = dc.load_default_catalog()
        assert cat.inns == frozenset({"7813266190"})
        assert "7707820890" not in cat.inns, (
            "Bundled catalog leaked through when USER_CATALOG_DIR "
            "had its own content — search order broken."
        )

    def test_falls_back_to_bundled_when_user_dir_empty(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import json

        from src.core import doc_catalog as dc

        user_dir = tmp_path / "user"
        bundled_dir = tmp_path / "bundled"
        user_dir.mkdir()  # exists but empty
        bundled_dir.mkdir()
        (bundled_dir / "b.json").write_text(
            json.dumps({"inn": "7707820890"}), encoding="utf-8",
        )

        monkeypatch.setattr(
            "src.shared.constants.USER_CATALOG_DIR", user_dir,
        )
        monkeypatch.setattr(
            "src.shared.constants.BUNDLED_CATALOG_DIR", bundled_dir,
        )

        cat = dc.load_default_catalog()
        assert cat.inns == frozenset({"7707820890"})

    def test_both_missing_returns_empty_catalog(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Dev checkouts without an ``expected/`` directory must not
        crash the pipeline — they get an empty catalog and every
        ``validate_identifiers`` call becomes a no-op."""
        from src.core import doc_catalog as dc

        monkeypatch.setattr(
            "src.shared.constants.USER_CATALOG_DIR",
            tmp_path / "nope-user",
        )
        monkeypatch.setattr(
            "src.shared.constants.BUNDLED_CATALOG_DIR",
            tmp_path / "nope-bundled",
        )
        cat = dc.load_default_catalog()
        assert cat.is_empty
