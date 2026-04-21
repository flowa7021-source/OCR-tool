"""Cross-field requisites consistency (idea #2 top-10).

Поля ``shipper_inn``, ``shipper_kpp``, ``shipper_ogrn`` (и
аналогичные для consignee) извлекаются из OCR-текста независимо.
Если OCR сбил блоки (shipper-section поглотил часть consignee-
section), парсер может смешать реквизиты двух организаций и
молчаливо отдать wrong mapping как «зелёный» high-confidence
record.

Этот модуль валидирует связность между ИНН, КПП и ОГРН:

  * Структурные инварианты (работают без catalog):
      ИНН 10-digit  ↔ ЮЛ     ↔ ОГРН 13-digit
      ИНН 12-digit  ↔ ИП     ↔ ОГРН 15-digit
    Смешанные комбинации (13-digit ОГРН + 12-digit ИНН) —
    структурно невозможны и flag'ируются как mismatch.

  * Catalog-проверка (опциональная):
      Caller передаёт catalog = dict[inn, dict] где inner-dict
      имеет ключи 'kpp', 'ogrn'. Если наш inn в каталоге, сверяем
      извлечённые KPP/OGRN с каталожными. Mismatch → flag +
      confidence drop.

Главный entry point — :func:`cross_check_requisites`. Возвращает
dataclass :class:`CrossCheckResult` с полями:

  * ``ok`` — все проверки прошли.
  * ``note`` — human-readable описание mismatch'а (для
    Excel-колонки «Примечание»).
  * ``confidence_delta`` — на сколько поднять/опустить общую
    confidence этой группы полей. При mismatch — отрицательное.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrossCheckResult:
    """Результат cross-validation'а ИНН/КПП/ОГРН-группы."""

    ok: bool
    note: str = ""
    confidence_delta: float = 0.0


def check_inn_ogrn_compatibility(inn: str, ogrn: str) -> bool:
    """Структурная проверка связки ИНН↔ОГРН.

    Формат ИНН/ОГРН регламентируется ФНС:
      * ИНН 10 цифр = юридическое лицо (ЮЛ)
      * ИНН 12 цифр = индивидуальный предприниматель (ИП) /
        физлицо
      * ОГРН 13 цифр = ЮЛ
      * ОГРН 15 цифр (ОГРНИП) = ИП

    ЮЛ-ОГРН не может принадлежать ИП-ИНН'у и наоборот. Если
    один из параметров пуст или имеет не-стандартную длину —
    возвращаем True (нечего проверять).
    """
    if not inn or not ogrn:
        return True
    inn_len = len(inn)
    ogrn_len = len(ogrn)
    # Отсекаем невалидные длины — это уже забота find_inn/find_ogrn.
    if inn_len not in (10, 12) or ogrn_len not in (13, 15):
        return True
    legal_entity = (inn_len == 10 and ogrn_len == 13)
    individual = (inn_len == 12 and ogrn_len == 15)
    return legal_entity or individual


def check_catalog_consistency(
    inn: str,
    kpp: str,
    ogrn: str,
    catalog: Mapping[str, Any] | None,
) -> CrossCheckResult:
    """Сверка с каталогом (опциональная).

    Catalog — mapping ``{inn: {kpp, ogrn, ...}}``. Если ИНН в
    каталоге, сравниваем извлечённые KPP/OGRN с каталожными.
    ИНН не в каталоге → pass-through (не наше дело validate
    unknown org).
    """
    if not catalog or not inn:
        return CrossCheckResult(ok=True)
    record = catalog.get(inn) if isinstance(catalog, Mapping) else None
    if not record or not isinstance(record, Mapping):
        return CrossCheckResult(ok=True)

    notes: list[str] = []
    cat_kpp = str(record.get("kpp", "") or "").strip()
    cat_ogrn = str(record.get("ogrn", "") or "").strip()

    if kpp and cat_kpp and kpp != cat_kpp:
        notes.append(
            f"catalog_conflict: КПП документа {kpp} ≠ каталожного "
            f"{cat_kpp} для ИНН {inn}"
        )
    if ogrn and cat_ogrn and ogrn != cat_ogrn:
        notes.append(
            f"catalog_conflict: ОГРН документа {ogrn} ≠ каталожного "
            f"{cat_ogrn} для ИНН {inn}"
        )

    if notes:
        return CrossCheckResult(
            ok=False, note="; ".join(notes), confidence_delta=-0.3,
        )
    return CrossCheckResult(ok=True)


def cross_check_requisites(
    inn: str,
    kpp: str,
    ogrn: str,
    catalog: Mapping[str, Any] | None = None,
) -> CrossCheckResult:
    """Главный entry point: aggregate structural + catalog checks.

    Args:
        inn, kpp, ogrn: Извлечённые значения (могут быть пустыми).
        catalog: Optional ``{inn: {kpp, ogrn}}`` mapping. Обычно
            собирается из ``expected/*.json`` / auto-learn log.

    Returns:
        :class:`CrossCheckResult` с aggregated verdict. Note-строка
        pipeline кладёт в row.note; confidence_delta применяется к
        confidence.shipper_inn/_kpp/_ogrn (или consignee_).
    """
    if not any((inn, kpp, ogrn)):
        return CrossCheckResult(ok=True)

    # 1. Структурная проверка ИНН↔ОГРН.
    if not check_inn_ogrn_compatibility(inn, ogrn):
        return CrossCheckResult(
            ok=False,
            note=(
                f"structural_mismatch: ИНН длиной {len(inn)} несовместим "
                f"с ОГРН длиной {len(ogrn)} (ЮЛ=ИНН10+ОГРН13, "
                f"ИП=ИНН12+ОГРН15)"
            ),
            confidence_delta=-0.5,
        )

    # 2. Catalog-сверка.
    catalog_result = check_catalog_consistency(inn, kpp, ogrn, catalog)
    if not catalog_result.ok:
        return catalog_result

    return CrossCheckResult(ok=True, confidence_delta=0.0)


__all__ = [
    "CrossCheckResult",
    "check_inn_ogrn_compatibility",
    "check_catalog_consistency",
    "cross_check_requisites",
]
