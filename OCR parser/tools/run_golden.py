# -*- coding: utf-8 -*-
"""Прогон парсера на golden-датасете (inputs/ + expected/).

Сравнивает 9 извлекаемых полей с экспертной разметкой и печатает
per-field accuracy. Запуск:

    python tools/run_golden.py
    python tools/run_golden.py --verbose   # построчно показать misses

В expected/*.json лежит развёрнутая схема (parties/cargo_header/transport/
loading/...). Здесь — лёгкий маппинг к нашим 9 полям.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tn_parser.core import extract_raw_text, parse_text  # noqa: E402
from tn_parser.normalize import normalize_for_sections  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs"
EXPECTED = ROOT / "expected"


_OCR_SUFFIXES = (" ocred", "-ocred", "_ocred", "-выход", "_выход")


def _strip_ocr_suffix(stem: str) -> Tuple[str, Optional[str]]:
    """Возвращает (base, suffix_or_None). Для 'foo ocred' → ('foo', ' ocred')."""
    for sfx in _OCR_SUFFIXES:
        if stem.endswith(sfx):
            return stem[: -len(sfx)], sfx
    return stem, None


def _match_expected(pdf: Path) -> Optional[Path]:
    """Ищем expected/<base>.json, где base — имя PDF без OCR-суффиксов."""
    base, _ = _strip_ocr_suffix(pdf.stem)
    p = EXPECTED / (base + ".json")
    return p if p.exists() else None


def _load_rows(pdf: Path):
    """Возвращает список ParsedRow.

    Стратегия:
        1) sidecar «<stem>.txt» рядом с PDF — если есть, берём его.
        2) PyMuPDF extract_raw_text — для PDF с OCR text-layer.
    """
    sidecar = pdf.with_suffix(".txt")
    if sidecar.exists():
        raw = sidecar.read_text(encoding="utf-8")
        return parse_text(normalize_for_sections(raw), pdf.name)
    raw = extract_raw_text(str(pdf))
    if not raw or not raw.strip():
        return None
    return parse_text(raw, pdf.name)


def _select_pdfs() -> list:
    """На каждый base выбираем один PDF по приоритету:
    ocred > выход > оригинал без суффикса.
    """
    priority = {" ocred": 0, "-ocred": 0, "_ocred": 0,
                "-выход": 1, "_выход": 1, None: 2}
    best: Dict[str, Path] = {}
    for pdf in sorted(INPUTS.glob("*.pdf")):
        base, sfx = _strip_ocr_suffix(pdf.stem)
        p = priority.get(sfx, 3)
        if base not in best or priority.get(
                _strip_ocr_suffix(best[base].stem)[1], 3) > p:
            best[base] = pdf
    return sorted(best.values())

FIELDS = ("number", "date", "shipper", "consignee", "cargo",
          "volume", "driver", "vehicle", "reception")


# --------------------------------------------------------------------------
# Маппинг expected JSON → dict с ожиданиями по нашим полям


def _date_iso_to_ru(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    return f"{m.group(3)}.{m.group(2)}.{m.group(1)}" if m else s


def _first_tn(doc: dict) -> Optional[dict]:
    """Выбираем первую ТН. Поддерживаем два формата expected:

    A) documents: [{type: "TN", ...}]  — наш старый формат.
    B) documents: ["UPD", "TTN_LIST", ...] + pages: [{document_type: "TTN",
       structured_fields: {...}}]  — УПД-пакеты (upd_549).
    """
    for d in doc.get("documents", []) or []:
        if isinstance(d, dict) and d.get("type") == "TN":
            return d
    # Fallback: новая pages-схема.
    for p in doc.get("pages", []) or []:
        if p.get("document_type") in ("TTN", "TN"):
            sf = p.get("structured_fields") or {}
            if sf:
                return _ttn_page_to_legacy(sf)
    return None


def _ttn_page_to_legacy(sf: dict) -> dict:
    """Маппим per-page structured_fields к legacy схеме TN."""
    consignor = sf.get("consignor") or {}
    consignee = sf.get("consignee") or {}
    vehicle = sf.get("vehicle") or {}
    cargo = sf.get("cargo") or {}
    loading = sf.get("loading_point") or {}
    # ttn_date: «20.10.22» → «20.10.2022»
    tdate = (sf.get("ttn_date") or "").strip()
    if re.fullmatch(r"\d{2}\.\d{2}\.\d{2}", tdate):
        tdate = tdate[:6] + "20" + tdate[6:]
    return {
        "type": "TN",
        "number": sf.get("ttn_internal_number") or sf.get("order_number"),
        "date": _ru_date_to_iso(tdate),
        "parties": [
            {"role": "shipper", **{k: consignor.get(k) for k in
                ("name", "legal_form", "inn", "kpp", "address") if consignor.get(k)}},
            {"role": "consignee", **{k: consignee.get(k) for k in
                ("name", "legal_form", "inn", "kpp", "address") if consignee.get(k)}},
            {"role": "carrier", "driver": {"short_name": sf.get("carrier_driver")}},
        ],
        "transport": {
            "vehicle_make": vehicle.get("brand"),
            "vehicle_reg_plate": vehicle.get("plate"),
        },
        "cargo_header": {
            "description": cargo.get("name"),
            "places_count": cargo.get("places_count"),
            "net_weight_t": cargo.get("net_weight_t"),
            "volume_m3": cargo.get("volume_m3"),
        },
        "loading": {
            "infrastructure_owner": {
                "name": loading.get("loader"),
                "legal_form": "ООО" if loading.get("loader", "").startswith("ООО") else "",
                "inn": loading.get("loader_inn"),
            } if loading.get("loader") else None,
        },
    }


def _ru_date_to_iso(s: str) -> Optional[str]:
    if not s:
        return None
    m = re.fullmatch(r"(\d{2})\.(\d{2})\.(\d{4})", s)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else s


def _party(tn: dict, role: str) -> Optional[dict]:
    for p in tn.get("parties", []):
        if p.get("role") == role:
            return p
    return None


def expected_fields(expected: dict) -> Dict[str, Any]:
    tn = _first_tn(expected)
    if not tn:
        return {}
    out: Dict[str, Any] = {}
    out["number"] = tn.get("number")
    out["date"] = _date_iso_to_ru(tn.get("date"))
    out["shipper"] = _party(tn, "shipper")
    out["consignee"] = _party(tn, "consignee")
    car = _party(tn, "carrier")
    if car and isinstance(car.get("driver"), dict):
        d = car["driver"]
        out["driver"] = d.get("short_name") or d.get("full_name")
    else:
        out["driver"] = None
    t = tn.get("transport", {}) or {}
    out["vehicle"] = (t.get("vehicle_make"), t.get("vehicle_reg_plate"))
    ch = tn.get("cargo_header", {}) or {}
    out["cargo"] = ch.get("description")
    out["volume"] = (
        ch.get("places_count"),
        ch.get("net_weight_t"),
        ch.get("volume_m3"),
    )
    l = tn.get("loading", {}) or {}
    out["reception"] = l.get("infrastructure_owner")
    return out


# --------------------------------------------------------------------------
# Сравнение


def _norm_az(s: str) -> str:
    return s.replace("А", "A").replace("Е", "E").replace("О", "O").upper()


def _norm_name(s: str) -> str:
    """Толерантная нормализация названий: lowercase + типовые OCR-подмены
    кириллица ↔ латиница ↔ цифры («Моспроект-З» ≡ «Моспроект-3»).
    """
    s = s.lower()
    for a, b in (("ё", "е"), ("з", "3"), ("о", "0"), ("е", "e"),
                 ("а", "a"), ("р", "p"), ("с", "c"), ("х", "x"),
                 ("у", "y"), ("к", "k"), ("м", "m"), ("т", "t"),
                 ("в", "b"), ("н", "h")):
        s = s.replace(a, b)
    return re.sub(r"[\s\-_.,«»\"\'()]+", "", s)


def _fuzzy_name_match(expected: str, got: str, threshold: int = 75) -> bool:
    """«Бекам» ≈ «Беком» (OCR-дрейф одной буквы)."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        e = _norm_name(expected)
        g = _norm_name(got)
        return e in g or g in e
    e = _norm_name(expected)
    g = _norm_name(got)
    if e in g:
        return True
    return fuzz.partial_ratio(e, g) >= threshold


def check_field(name: str, expected: Any, got: str
                ) -> Tuple[Optional[bool], str]:
    """Возвращает (ok|None, комментарий). None — нет ожидания."""
    if expected in (None, "", [], (None, None), (None, None, None)):
        return None, "no expectation"
    g = (got or "").strip()
    if not g or g == "отсутствует":
        return False, f"MISSING (expected: {expected!r})"

    if name == "number":
        return _norm_az(str(expected)) == _norm_az(g), \
               f"got {g!r}, expected {expected!r}"
    if name == "date":
        return g == expected, f"got {g!r}, expected {expected!r}"

    if name in ("shipper", "consignee"):
        ok = True
        miss = []
        if expected.get("inn") and expected["inn"] not in g:
            ok = False
            miss.append(f"inn {expected['inn']}")
        if expected.get("name"):
            short_name = expected["name"].split(",")[0].strip()
            if not _fuzzy_name_match(short_name, g):
                ok = False
                miss.append(f"name «{short_name}»")
        return ok, (f"got {g!r}" if ok
                    else f"got {g!r}, missing: {', '.join(miss)}")

    if name == "driver":
        # Сравнение по фамилии (первое слово) — толерантно к формату
        # «Иванов И.И.» vs «И. И. Иванов».
        e_tokens = re.findall(r"[А-ЯЁ][а-яё]{2,}", str(expected))
        g_tokens = re.findall(r"[А-ЯЁ][а-яё]{2,}", g)
        if not e_tokens:
            return None, f"no surname in expected {expected!r}"
        if not g_tokens:
            return False, f"got {g!r}, no surname found"
        ok = e_tokens[0].lower() == g_tokens[0].lower()
        return ok, f"got {g!r}, expected {expected!r}"

    if name == "vehicle":
        make, plate = expected
        plate_compact = re.sub(r"\s+", "", plate or "").upper()
        g_compact = re.sub(r"\s+", "", g).upper()
        ok_plate = bool(plate_compact) and _norm_az(plate_compact) in _norm_az(g_compact)
        ok_make = (not make) or make.upper() in g.upper()
        # Считаем ок, если совпал ГРЗ — марка бонус.
        ok = ok_plate
        marker = "full" if (ok_plate and ok_make) else ("plate-only" if ok_plate else "no-plate")
        return ok, f"got {g!r}, expected make={make!r}, plate={plate!r} [{marker}]"

    if name == "cargo":
        words = [w for w in re.findall(r"[А-Яа-яёЁA-Za-z]{4,}", expected.lower())
                 if w not in ("груз", "груза", "масса", "вес", "брутто",
                              "нетто", "объем", "объём")]
        if not words:
            return None, "expected too generic"
        keys = words[:3]
        hits = sum(1 for w in keys if w in g.lower())
        ok = hits >= 1
        return ok, f"got {g!r}; key words {keys}, hits {hits}"

    if name == "volume":
        places, net, vol = expected
        ok = False
        clues = []
        if places is not None and str(places) in g:
            ok, clues = True, clues + [f"places={places}"]
        if vol is not None:
            v_norm = str(vol).replace(".", ",")
            if v_norm in g.replace(".", ","):
                ok, clues = True, clues + [f"vol={vol}"]
        if net is not None:
            n_norm = str(net).replace(".", ",")
            if n_norm in g.replace(".", ","):
                ok, clues = True, clues + [f"net={net}"]
        return ok, f"got {g!r}; matched [{', '.join(clues) or 'none'}]"

    if name == "reception":
        if not isinstance(expected, dict):
            return None, f"unexpected type {type(expected).__name__}"
        miss = []
        if expected.get("inn") and expected["inn"] not in g:
            miss.append(f"inn {expected['inn']}")
        if expected.get("name"):
            short = expected["name"].split(",")[0].strip()
            if not _fuzzy_name_match(short, g):
                miss.append(f"name «{short}»")
        ok = not miss
        return ok, (f"got {g!r}" if ok
                    else f"got {g!r}, missing: {', '.join(miss)}")

    return None, "no logic"


# --------------------------------------------------------------------------
# main


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not INPUTS.exists() or not EXPECTED.exists():
        print(f"Папок {INPUTS}/ и {EXPECTED}/ не существует.", file=sys.stderr)
        return 1

    pdfs = _select_pdfs()
    if not pdfs:
        print(f"В {INPUTS}/ нет PDF.", file=sys.stderr)
        return 1

    totals: Dict[str, Dict[str, int]] = {
        f: {"ok": 0, "fail": 0, "skip": 0} for f in FIELDS
    }

    for pdf in pdfs:
        exp_path = _match_expected(pdf)
        if exp_path is None:
            print(f"\n=== {pdf.name} === (нет expected, пропуск)")
            continue
        expected = json.loads(exp_path.read_text(encoding="utf-8"))
        rows = _load_rows(pdf)
        if not rows:
            print(f"\n=== {pdf.name} === нет текстового слоя в PDF "
                  f"и нет sidecar-файла «{pdf.stem}.txt»")
            continue
        row = rows[0]
        exp = expected_fields(expected)
        print(f"\n=== {pdf.name} ===")
        for fld in FIELDS:
            ok, comment = check_field(fld, exp.get(fld), getattr(row, fld))
            mark = "✓" if ok else ("✗" if ok is False else "·")
            if ok is True:
                totals[fld]["ok"] += 1
            elif ok is False:
                totals[fld]["fail"] += 1
            else:
                totals[fld]["skip"] += 1
            if args.verbose or ok is False:
                print(f"  {mark} {fld:10s}  {comment}")
            else:
                print(f"  {mark} {fld:10s}")

    print("\n" + "=" * 60)
    print("ИТОГ:")
    overall_ok = overall_fail = 0
    for fld in FIELDS:
        t = totals[fld]
        ok, fail, skip = t["ok"], t["fail"], t["skip"]
        overall_ok += ok
        overall_fail += fail
        denom = ok + fail
        acc = f"{int(ok / denom * 100)}%" if denom else "—"
        print(f"  {fld:10s}  ok={ok}  fail={fail}  skip={skip}  acc={acc}")
    denom = overall_ok + overall_fail
    print("-" * 60)
    print(f"  {'TOTAL':10s}  ok={overall_ok}  fail={overall_fail}  "
          f"acc={int(overall_ok / denom * 100) if denom else '—'}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
