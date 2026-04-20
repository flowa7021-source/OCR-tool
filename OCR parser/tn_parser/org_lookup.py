# -*- coding: utf-8 -*-
"""Локальный справочник «ИНН → организация».

Назначение: автоматическая коррекция OCR-искажений реквизитов через
валидный ИНН. Если парсер извлёк валидный ИНН (по контр-сумме) — можно
подтянуть каноничные название/адрес/КПП и не зависеть от OCR-качества
конкретного скана.

Источники справочника (по приоритету):
    1) JSON-кеш `data/org_cache.json` — наш собственный append-only
       словарь, заполняемый вручную и накопительно (каждый раз, когда
       видим новую организацию с валидным ИНН).
    2) Экспертная разметка `expected/*.json` — для golden-датасета и
       интеграционных тестов.

Оба источника сливаются при `load_lookup()`. Формат записи:

    "7707820890": {
        "name": "Моспроект-3",
        "legal_form": "АО",
        "inn": "7707820890",
        "kpp": "770701001",
        "ogrn": "5137746157490",
        "address": "107031, Москва, ул. Кузнецкий мост, д. 3, стр. 1",
        "source": "expected/TN_k_UPD_36_ot_02.09.2022.json"
    }

Справочник НЕ ходит в интернет. Для онлайн-обогащения (ФНС API,
DaData, ЕГРЮЛ) сделан отдельный hook — см. `tn_parser.org_online`
(не реализован, задел на будущее).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .validators import is_valid_inn


ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CACHE = ROOT / "data" / "org_cache.json"
_DEFAULT_EXPECTED_DIR = ROOT / "expected"


_cache: Optional[Dict[str, Dict]] = None


def _ensure_data_dir() -> None:
    _DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _harvest_from_expected(expected_dir: Path) -> Dict[str, Dict]:
    """Вычитывает из expected/*.json все упоминания организаций
    в разделах «parties» (любой role) и «loading.infrastructure_owner».
    """
    out: Dict[str, Dict] = {}
    if not expected_dir.is_dir():
        return out
    for fp in sorted(expected_dir.glob("*.json")):
        data = _read_json(fp)
        if not isinstance(data, dict):
            continue
        for doc in data.get("documents", []) or []:
            if not isinstance(doc, dict):
                continue
            for p in doc.get("parties", []) or []:
                _absorb(out, p, source=fp.name)
            loading = doc.get("loading") or {}
            _absorb(out, loading.get("infrastructure_owner"), source=fp.name)
            unloading = doc.get("unloading") or {}
            _absorb(out, unloading.get("infrastructure_owner"), source=fp.name)
    return out


def _absorb(store: Dict[str, Dict], party: Optional[dict], source: str) -> None:
    if not isinstance(party, dict):
        return
    inn = (party.get("inn") or "").strip()
    if not inn or not is_valid_inn(inn):
        return
    # Не перезаписываем существующую запись (первый встретившийся
    # источник считается каноничным — обычно expected).
    if inn in store:
        return
    store[inn] = {
        "inn": inn,
        "name": party.get("name") or "",
        "legal_form": party.get("legal_form") or "",
        "kpp": party.get("kpp") or "",
        "ogrn": party.get("ogrn") or "",
        "address": party.get("address") or "",
        "phone": party.get("phone") or "",
        "source": source,
    }


def load_lookup(
    *,
    cache_path: Optional[Path] = None,
    expected_dir: Optional[Path] = None,
    refresh: bool = False,
) -> Dict[str, Dict]:
    """Загружает и сливает кеш и expected. Результат кешируется в модуле."""
    global _cache
    if _cache is not None and not refresh:
        return _cache
    cache_path = cache_path or _DEFAULT_CACHE
    expected_dir = expected_dir or _DEFAULT_EXPECTED_DIR
    merged: Dict[str, Dict] = {}
    cached = _read_json(cache_path)
    if isinstance(cached, dict):
        for inn, rec in cached.items():
            if is_valid_inn(inn) and isinstance(rec, dict):
                merged[inn] = rec
    for inn, rec in _harvest_from_expected(expected_dir).items():
        merged.setdefault(inn, rec)
    _cache = merged
    return merged


def lookup_by_inn(inn: str) -> Optional[Dict]:
    """Вернёт запись организации по валидному ИНН или None."""
    if not inn or not is_valid_inn(inn):
        return None
    return load_lookup().get(inn)


def remember(inn: str, record: Dict) -> None:
    """Добавить запись в кеш (append-only) и сохранить на диск.

    Если запись по этому ИНН уже есть — ничего не делаем
    (чтобы не перезаписать экспертную разметку «грязным» OCR-текстом).
    """
    if not is_valid_inn(inn):
        return
    lookup = load_lookup()
    if inn in lookup:
        return
    rec = dict(record)
    rec["inn"] = inn
    lookup[inn] = rec
    _ensure_data_dir()
    try:
        existing = _read_json(_DEFAULT_CACHE) or {}
        existing[inn] = rec
        _DEFAULT_CACHE.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def known_inns() -> List[str]:
    return sorted(load_lookup().keys())
