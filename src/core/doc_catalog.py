"""Load ground-truth business identifiers from ``expected/*.json``.

The user maintains a growing corpus of hand-validated document
JSONs under ``expected/`` — transport invoices, UPDs, powers-of-
attorney. Every such JSON already contains the authoritative ИНН /
ОГРН / КПП values for the parties involved. This module walks
those JSONs and returns flat sets of known-good identifiers that
the post-OCR fixup can consult when Tesseract emits a corrupt
version of one of them.

The loader is defensive: unknown JSON shapes, missing keys, invalid
entries are silently skipped with a debug log. A missing ``expected``
directory returns empty sets rather than raising — the feature is
optional, not a hard dependency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from src.core.doc_validators import (
    validate_inn,
    validate_kpp,
    validate_ogrn,
)

logger = logging.getLogger(__name__)

__all__ = ["DocCatalog", "load_catalog", "load_default_catalog"]


@dataclass(frozen=True)
class DocCatalog:
    """Frozen sets of known-good identifiers for look-up-style fixup.

    All four fields are guaranteed to contain ONLY checksum-valid
    entries — the loader discards anything that fails its validator
    (with a WARNING so the user learns about typos in their own
    ground-truth data). Organisation names are not checksum-checked,
    just stripped / deduped.
    """

    inns: frozenset[str] = field(default_factory=frozenset)
    ogrns: frozenset[str] = field(default_factory=frozenset)
    kpps: frozenset[str] = field(default_factory=frozenset)
    names: frozenset[str] = field(default_factory=frozenset)

    def __len__(self) -> int:
        return len(self.inns) + len(self.ogrns) + len(self.kpps) + len(self.names)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0


def _walk_values(node: object):
    """Yield every (key, value) scalar pair in a nested JSON tree.

    Works on arbitrary dict / list combinations. Non-dict, non-list
    leaves are skipped (we only care about string scalars nested
    under named keys).
    """
    if isinstance(node, dict):
        for key, val in node.items():
            if isinstance(val, (dict, list)):
                yield from _walk_values(val)
            else:
                yield key, val
    elif isinstance(node, list):
        for item in node:
            yield from _walk_values(item)


def load_catalog(expected_dir: Path) -> DocCatalog:
    """Scan every ``*.json`` under ``expected_dir`` and collect IDs.

    Returns an empty :class:`DocCatalog` (not an error) when the
    directory doesn't exist, contains no JSONs, or every JSON
    parses empty — the feature is opt-in and a missing corpus
    should be a silent no-op.

    Args:
        expected_dir: Path to the repository's ``expected/``
            directory (or a test-provided equivalent).

    Returns:
        Frozen sets of every valid ``inn`` / ``ogrn`` / ``kpp`` /
        ``name`` value found. Invalid entries (checksum failure)
        are logged at WARNING and omitted from the result.
    """
    if not expected_dir.is_dir():
        logger.debug(
            "DocCatalog: %s is not a directory — returning empty catalog",
            expected_dir,
        )
        return DocCatalog()

    inns: set[str] = set()
    ogrns: set[str] = set()
    kpps: set[str] = set()
    names: set[str] = set()

    for json_path in sorted(expected_dir.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "DocCatalog: skipping %s (parse error: %s)",
                json_path.name, exc,
            )
            continue

        for key, val in _walk_values(data):
            if not isinstance(val, str):
                continue
            v = val.strip()
            if not v:
                continue
            k = key.lower()
            if k == "inn":
                if validate_inn(v):
                    inns.add(v)
                else:
                    logger.warning(
                        "DocCatalog: %s contains inn=%r that fails "
                        "checksum — skipping",
                        json_path.name, v,
                    )
            elif k == "ogrn":
                if validate_ogrn(v):
                    ogrns.add(v)
                else:
                    logger.warning(
                        "DocCatalog: %s contains ogrn=%r that fails "
                        "checksum — skipping",
                        json_path.name, v,
                    )
            elif k == "kpp":
                if validate_kpp(v):
                    kpps.add(v)
                else:
                    logger.warning(
                        "DocCatalog: %s contains kpp=%r that fails "
                        "format check — skipping",
                        json_path.name, v,
                    )
            elif k == "name":
                names.add(v)

    catalog = DocCatalog(
        inns=frozenset(inns),
        ogrns=frozenset(ogrns),
        kpps=frozenset(kpps),
        names=frozenset(names),
    )
    logger.info(
        "DocCatalog loaded from %s: %d inn(s), %d ogrn(s), "
        "%d kpp(s), %d name(s)",
        expected_dir, len(catalog.inns), len(catalog.ogrns),
        len(catalog.kpps), len(catalog.names),
    )
    return catalog


def load_default_catalog() -> DocCatalog:
    """Load the catalog from the OS-standard search path.

    Tries in order:

      1. ``USER_CATALOG_DIR`` (``%LOCALAPPDATA%/OCRStudio/expected`` on
         Windows) — lets the user drop per-install JSON fixtures
         without a rebuild. Takes precedence so updates take effect
         on the next run.
      2. ``BUNDLED_CATALOG_DIR`` (``<app>/expected``) — shipped with
         the installer so a fresh checkout has a usable catalog.

    Returns an empty :class:`DocCatalog` when neither directory has
    any loadable JSONs — the feature degrades gracefully.

    Callers (pipeline worker bootstrap, CLI entry) invoke this
    ONCE and pass the result to every :class:`~src.core.text_postprocessor
    .TextPostprocessor` they construct. The catalog is read-only and
    immutable, so it's safe to share across threads / processes.
    """
    from src.shared.constants import BUNDLED_CATALOG_DIR, USER_CATALOG_DIR

    for candidate in (USER_CATALOG_DIR, BUNDLED_CATALOG_DIR):
        if candidate.is_dir():
            cat = load_catalog(candidate)
            if not cat.is_empty:
                return cat
    logger.debug(
        "load_default_catalog: no non-empty catalog found in %s or %s",
        USER_CATALOG_DIR, BUNDLED_CATALOG_DIR,
    )
    return DocCatalog()
