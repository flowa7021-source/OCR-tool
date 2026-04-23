"""Reference registries for snap-to-nearest correction of extracted entities.

When OCR returns a legal-form or bank name that's close-but-wrong,
fuzzy-match against a whitelist of known valid values pulls it back
to canonical form. Covers:

- Legal entity forms (ООО, АО, ИП, ЗАО, ПАО, КФХ, …)
- Bank BIK → name mapping (seed data for common Russian banks;
  full list ~400 banks is available from ЦБ РФ but not shipped)
- Common organization name tokens (ТНК, РЖД, …)

All whitelists are small (≤ 500 entries) so fuzzy lookup is O(n) with
Levenshtein cutoff — plenty fast. For a full-scale production system,
back with a fuzzy-index library (rapidfuzz.process) and larger
curated lists.

The registries are Python constants in this module so they ship
in the installer without relying on network access or external data
files at runtime. Update by editing the constants; regenerate tests.
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein

# ── Legal entity forms ────────────────────────────────────────────────────
# Ordered by commonness. Snap-to-nearest will pick the shortest one on
# ties since "ИП" is more common than many lookalikes.
_LEGAL_FORMS_RU: tuple[str, ...] = (
    "ООО",       # Общество с ограниченной ответственностью
    "АО",        # Акционерное общество
    "ПАО",       # Публичное АО
    "ЗАО",       # Закрытое АО (legacy но ещё встречается)
    "ОАО",       # Открытое АО (legacy)
    "ИП",        # Индивидуальный предприниматель
    "КФХ",       # Крестьянское (фермерское) хозяйство
    "НКО",       # Некоммерческая организация
    "УФПС",      # Управление федеральной почтовой связи
    "ГУП",       # Государственное унитарное предприятие
    "МУП",       # Муниципальное унитарное предприятие
    "ФГУП",
    "ФГУ",
    "ФГБУ",
    "ФГБОУ",
    "АНО",
    "ТСЖ",
)

# ── Bank BIK seed data ────────────────────────────────────────────────────
# Subset of the most frequently seen banks in our corpus. Full registry
# lives at https://cbr.ru/scripts/XML_bik.asp (XML), fetchable offline.
# Structure: BIK (9 digits) → canonical name fragment.
_BANK_BIK_SEEDS: dict[str, str] = {
    "044525225": "ПАО Сбербанк",
    "044525745": "Банк ВТБ",
    "044525593": "АО Альфа-Банк",
    "044525974": "АО Тинькофф Банк",
    "044525411": "Банк Открытие",
    "044525201": "АО Газпромбанк",
    "044525823": "ПАО Промсвязьбанк",
    "044525999": "ПАО РНКБ Банк",
    "044525187": "ПАО Банк Санкт-Петербург",
    "044525161": "ПАО Росбанк",
    # Regional
    "040813608": "Отделение по ЛО Северо-Западного ГУ Банка России",
    "044030653": "Северо-Западное ГУ Банка России",
}

# ── Common organization names / abbreviations ─────────────────────────────
# Targets typical OCR-ugly outputs like "ТhК-ВР" → "ТНК-ВР".
_KNOWN_ORG_TOKENS: tuple[str, ...] = (
    "ТНК-ВР", "ТНК", "ЛУКОЙЛ", "ГАЗПРОМ", "РОСНЕФТЬ", "РЖД",
    "АЭРОФЛОТ", "СБЕРБАНК", "РОСАТОМ", "РОССЕТИ",
    "МОСЭНЕРГО", "ГЕКСАФОРМ", "АВТОПРОФИТ", "ТЕНСАР",
)


def snap_legal_form(token: str, max_edits: int = 2) -> str | None:
    """Return the canonical legal form within ``max_edits`` Levenshtein
    distance of ``token``, or ``None`` if nothing close.

    Case-insensitive. Ties broken by preferring shorter canonical form
    (closer match to the user's intent for ambiguous short tokens).
    """
    if not token:
        return None
    lw = token.strip().upper().replace('"', "")
    if lw in _LEGAL_FORMS_RU:
        return lw
    best: tuple[int, str] | None = None
    for form in _LEGAL_FORMS_RU:
        if abs(len(form) - len(lw)) > max_edits:
            continue
        d = Levenshtein.distance(lw, form, score_cutoff=max_edits)
        if d <= max_edits and (
            best is None
            or d < best[0]
            or (d == best[0] and len(form) < len(best[1]))
        ):
            best = (d, form)
    return best[1] if best else None


def snap_bank_by_bik(bik: str) -> str | None:
    """Return the canonical bank name for a valid BIK, or None.

    Does NOT fuzzy-match the BIK itself — BIK is numeric-exact.
    Use ``src.shared.requisite_validators.correct_inn``-style OCR
    correction upstream to rescue letter-confusable BIK digits first.
    """
    return _BANK_BIK_SEEDS.get(bik.strip())


def snap_known_org(token: str, max_edits: int = 2) -> str | None:
    """Best-effort match against a small list of well-known org tokens.
    Case-insensitive.
    """
    if not token:
        return None
    lw = token.strip().upper().replace('"', "")
    if lw in _KNOWN_ORG_TOKENS:
        return lw
    best: tuple[int, str] | None = None
    for org in _KNOWN_ORG_TOKENS:
        if abs(len(org) - len(lw)) > max_edits:
            continue
        d = Levenshtein.distance(lw, org, score_cutoff=max_edits)
        if d <= max_edits and (best is None or d < best[0]):
            best = (d, org)
    return best[1] if best else None


__all__ = [
    "snap_legal_form",
    "snap_bank_by_bik",
    "snap_known_org",
]
