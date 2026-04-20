# -*- coding: utf-8 -*-
"""Feedback loop: собираем правки оператора как обучающий корпус.

Цикл без GUI-компонента (через Excel):

    1. Парсер пишет Excel `out.xlsx` + снапшот `out.xlsx.snapshot.json`
       (оригинальные значения всех 9 полей по source-файлам).
    2. Оператор открывает Excel, правит неверные поля вручную, сохраняет.
    3. `python tools/collect_feedback.py out.xlsx` сравнивает текущий
       Excel со снапшотом и записывает правки в
       `feedback/corrections.jsonl` — append-only лог.
    4. Накопленные правки становятся:
       - источником новых регресс-тестов (особенно если правки повторяются);
       - обучающим корпусом для улучшения эвристик (мы видим, какие
         поля оператор правит чаще всего → где парсер слабее);
       - материалом для few-shot промпта LLM-fallback.

Преимущества этого подхода:
    - Нулевая нагрузка на оператора (он и так правит Excel).
    - Не требует GUI-разработки.
    - Простой append-only формат (JSONL) легко версионировать
      и анализировать.

Модуль содержит только бизнес-логику; чтение/запись Excel — в
скриптах `tools/collect_feedback.py`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import MISSING, GARBAGE


# Поля, которые может корректировать оператор.
TRACKABLE_FIELDS = (
    "number", "date", "shipper", "consignee", "cargo",
    "volume", "driver", "vehicle", "reception",
)


@dataclass
class Correction:
    """Одна правка оператора: поле X изменено со значения A на B.

    Поля:
        source: имя исходного PDF (или виртуальный #N для multi-doc).
        field: название поля (см. TRACKABLE_FIELDS).
        original: что выдал парсер.
        corrected: что написал оператор.
        original_confidence: уверенность парсера в original (0..1).
        kind: 'fix' (оба заполнены, исправление) | 'fill' (original был
              MISSING, оператор заполнил) | 'clear' (оператор очистил).
        timestamp: ISO 8601 время записи правки.
        excel_row: номер строки в Excel (для отладки).
        context: опциональная подсказка — что за документ / короткий
                 хвост текста или сырой OCR для few-shot промпта.
    """

    source: str
    field: str
    original: str
    corrected: str
    original_confidence: float
    kind: str  # fix | fill | clear
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    excel_row: Optional[int] = None
    context: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _kind_of(original: str, corrected: str) -> str:
    orig_empty = original in (MISSING, GARBAGE, "")
    corr_empty = corrected in (MISSING, GARBAGE, "")
    if orig_empty and not corr_empty:
        return "fill"
    if not orig_empty and corr_empty:
        return "clear"
    return "fix"


def diff_rows(
    snapshot: List[dict],
    corrected: List[dict],
    *,
    source_key: str = "source",
) -> List[Correction]:
    """Сравнивает снапшот парсера с правленной версией и возвращает
    список обнаруженных правок.

    Оба списка — dict с ключами TRACKABLE_FIELDS + source + confidence.
    Матчим строки по source (с fallback по row_index, если ключа нет).
    """
    # Строим индекс снапшота по source.
    by_source: Dict[str, dict] = {}
    for i, row in enumerate(snapshot):
        key = str(row.get(source_key) or f"__row_{i}")
        # Дубликаты source: добавим индекс, чтобы не перезаписывать.
        if key in by_source:
            key = f"{key}__dup{i}"
        by_source[key] = row

    corrections: List[Correction] = []
    for i, corr_row in enumerate(corrected):
        key = str(corr_row.get(source_key) or f"__row_{i}")
        orig_row = by_source.get(key)
        if orig_row is None:
            # Нет в снапшоте — не учитываем (возможно оператор добавил
            # новую строку вручную, это уже не автоматическая правка).
            continue
        for fld in TRACKABLE_FIELDS:
            orig_val = (orig_row.get(fld) or "").strip()
            corr_val = (corr_row.get(fld) or "").strip()
            if orig_val == corr_val:
                continue
            conf = float(orig_row.get(f"{fld}_conf", 0.0) or 0.0)
            corrections.append(
                Correction(
                    source=str(orig_row.get(source_key) or ""),
                    field=fld,
                    original=orig_val,
                    corrected=corr_val,
                    original_confidence=conf,
                    kind=_kind_of(orig_val, corr_val),
                    excel_row=i + 2,  # +2: header + 1-based
                )
            )
    return corrections


def snapshot_from_rows(rows: Iterable["object"]) -> List[dict]:
    """Превращает итерируемое ParsedRow в список dict'ов для снапшота.

    Сохраняем value + confidence по каждому полю (нужно при
    построении diff — низкая confidence подсказывает, что правка
    достовернее исходного значения).
    """
    out: List[dict] = []
    for row in rows:
        d: Dict[str, object] = {"source": row.source}
        for fld in TRACKABLE_FIELDS:
            d[fld] = getattr(row, fld, "")
            d[f"{fld}_conf"] = getattr(row.confidence, fld, 0.0)
        d["overall_confidence"] = row.confidence.overall()
        d["waybill"] = row.waybill
        d["note"] = row.note
        out.append(d)
    return out


def save_snapshot(rows: Iterable["object"], snapshot_path: Path) -> None:
    """Сохраняет снапшот рядом с Excel. Называется `<excel>.snapshot.json`."""
    data = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "rows": snapshot_from_rows(rows),
    }
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_snapshot(snapshot_path: Path) -> Optional[dict]:
    if not snapshot_path.exists():
        return None
    try:
        return json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def append_corrections(
    corrections: List[Correction],
    log_path: Path,
) -> None:
    """Append-only лог. Каждая правка — одна строка JSON."""
    if not corrections:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        for c in corrections:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_corrections(log_path: Path) -> List[Correction]:
    """Читает накопленный корпус правок."""
    if not log_path.exists():
        return []
    out: List[Correction] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        # Совместимость: игнорим лишние ключи.
        allowed = {
            k: d[k] for k in (
                "source", "field", "original", "corrected",
                "original_confidence", "kind", "timestamp",
                "excel_row", "context",
            ) if k in d
        }
        try:
            out.append(Correction(**allowed))
        except TypeError:
            continue
    return out
