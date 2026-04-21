"""Batch document-profile learning (idea #3 top-10).

Сценарий: пакет из N ТН с повторяющимися контрагентами. Первый
документ OCR'ится хорошо (ИНН/КПП/ОГРН extract'нуты с conf ≥ 0.9);
следующие — с низкой conf (рукопись / размытая печать / плохой
scan). Независимая обработка каждого docomento теряет легитимно-
известные реквизиты.

**BatchContext** — накопитель «уверенно извлечённых» организаций
в рамках одного batch-run'а. Второй+ документ проверяет: если
его shipper-имя fuzzy-match'ит одного из seen_shippers, и
недостающие реквизиты у current-row пусты — подтягиваем из seen.

Usage::

    ctx = BatchContext()
    for doc in batch:
        row = parser.parse(doc)
        ctx.apply_learning(row)  # до add_row — чтобы не само-learn'ить
        ctx.add_row(row)         # после — если row high-conf

Ограничения:

  * Работает в пределах **одного run'а** — state не persist'ится
    на диск между запусками (это задача auto-learn каталога,
    отдельная штука).
  * Применяется только к shipper / consignee (именно они дают
    критичные реквизиты для бухучёта). Driver / vehicle — per-doc
    unique, batch-learning неуместен.
  * High-conf gate = ``overall_confidence ≥ 0.7`` — защищает от
    self-corruption когда первый doc был неуверенно распознан.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Минимальная overall confidence чтобы row попал в seen_* .
_HIGH_CONF_THRESHOLD = 0.7

#: Fuzzy ratio для считать что имя — тот же контрагент.
_NAME_MATCH_THRESHOLD = 80


@dataclass
class BatchContext:
    """Per-batch state: накапливает уверенно-extract'нутые
    организации, применяет их к последующим row'ам."""

    seen_shippers: list[dict[str, str]] = field(default_factory=list)
    seen_consignees: list[dict[str, str]] = field(default_factory=list)

    def add_row(self, row: dict[str, Any]) -> None:
        """Запомнить row-реквизиты если row high-conf.

        Проверяется ``overall_confidence`` (default 0.0 если ключа
        нет). Добавляет в seen_shippers / seen_consignees entry
        с ``name``, ``inn``, ``kpp``, ``ogrn``.
        """
        try:
            overall = float(row.get("overall_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            overall = 0.0
        if overall < _HIGH_CONF_THRESHOLD:
            return

        sh_name = (row.get("shipper") or "").strip()
        sh_inn = (row.get("shipper_inn") or "").strip()
        if sh_name and sh_inn:
            self.seen_shippers.append({
                "name": sh_name,
                "inn": sh_inn,
                "kpp": (row.get("shipper_kpp") or "").strip(),
                "ogrn": (row.get("shipper_ogrn") or "").strip(),
            })

        cn_name = (row.get("consignee") or "").strip()
        cn_inn = (row.get("consignee_inn") or "").strip()
        if cn_name and cn_inn:
            self.seen_consignees.append({
                "name": cn_name,
                "inn": cn_inn,
                "kpp": (row.get("consignee_kpp") or "").strip(),
                "ogrn": (row.get("consignee_ogrn") or "").strip(),
            })

    def apply_learning(self, row: dict[str, Any]) -> None:
        """In-place: заполнить пустые реквизиты из seen.

        * Только для shipper + consignee полей.
        * Не перезаписывает уже-заполненные поля (primary-extraction
          priority).
        * Trigger — fuzzy name-match ≥ :data:`_NAME_MATCH_THRESHOLD`
          между current-row name и seen.name.

        При success-learning'е добавляет маркер
        ``learned_from_batch`` в ``row.note``.
        """
        changed = False
        if self._fill_missing_from_seen(row, "shipper", self.seen_shippers):
            changed = True
        if self._fill_missing_from_seen(row, "consignee", self.seen_consignees):
            changed = True
        if changed:
            note = (row.get("note") or "").strip()
            marker = "learned_from_batch"
            if marker not in note:
                row["note"] = f"{note};{marker}".strip(";")

    @staticmethod
    def _fuzzy_match_name(a: str, b: str) -> bool:
        """rapidfuzz token_sort_ratio ≥ threshold; fallback на
        token-overlap если rapidfuzz недоступен."""
        if not a or not b:
            return False
        try:
            from rapidfuzz.fuzz import token_sort_ratio
            return token_sort_ratio(a, b) >= _NAME_MATCH_THRESHOLD
        except ImportError:
            at = set(a.lower().split())
            bt = set(b.lower().split())
            if not at or not bt:
                return False
            overlap = len(at & bt) / max(len(at), len(bt))
            return overlap * 100 >= _NAME_MATCH_THRESHOLD

    def _fill_missing_from_seen(
        self,
        row: dict[str, Any],
        prefix: str,
        seen: list[dict[str, str]],
    ) -> bool:
        """Если ``row[prefix]`` fuzzy-match'ит один из seen.name,
        заполняем пустые inn/kpp/ogrn из того seen. Returns True
        если что-то изменилось."""
        current_name = (row.get(prefix) or "").strip()
        if not current_name or not seen:
            return False

        # Ищем первую near-matching запись.
        matched = None
        for rec in seen:
            if self._fuzzy_match_name(current_name, rec.get("name", "")):
                matched = rec
                break
        if not matched:
            return False

        changed = False
        for sub in ("inn", "kpp", "ogrn"):
            key = f"{prefix}_{sub}"
            current_val = (row.get(key) or "").strip()
            learned_val = matched.get(sub, "").strip()
            if not current_val and learned_val:
                row[key] = learned_val
                changed = True
        return changed


__all__ = ["BatchContext"]
