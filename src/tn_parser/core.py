"""Связывающий модуль: PDF → список ParsedRow + кэш.

Публичные функции:
    extract_raw_text(pdf_path)    — сырой (но нормализованный) текст
    parse_text(text, source)      — список ParsedRow из текста
    process_one_pdf(pdf_path)     — основной вход для GUI
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile

from .fields import extract_all
from .layout import extract_best_text
from .models import GARBAGE, MISSING, FieldConfidence, ParsedRow
from .normalize import normalize_for_sections
from .org_lookup import lookup_by_inn
from .sections import split_sections
from .splitter import split_documents
from .validators import is_valid_inn

LOW_TEXT_THRESHOLD = 200  # символов
CACHE_VERSION = 10  # ↑ при изменении логики парсинга


def _enrich_with_inn(raw: str, full_text: str) -> str:
    """Консервативное обогащение: если в `raw` ИНН отсутствует, но в
    `full_text` найден валидный ИНН организации с именем, кусок
    которого присутствует в `raw` — добавляем «, ИНН XXX» в конец.

    НЕ переписываем уже найденное имя (OCR мог распознать «Бекам», а
    в справочнике «Беком» — нам не надо спорить с экспертом на лету).
    """
    if not raw or raw in (MISSING, GARBAGE) or not full_text:
        return raw
    import re
    # Если ИНН уже есть в строке (валидный или нет) — не трогаем,
    # чтобы не дублировать.
    if re.search(r"\bИНН\s*\d{10,12}", raw, re.IGNORECASE):
        return raw
    if re.search(r"\b(\d{10}|\d{12})\b", raw):
        return raw
    raw_low = raw.lower()
    for m in re.finditer(r"\b(\d{10}|\d{12})\b", full_text):
        inn = m.group(1)
        if not is_valid_inn(inn):
            continue
        rec = lookup_by_inn(inn)
        if not rec:
            continue
        name = (rec.get("name") or "").lower()
        # Достаточно 4 первых букв имени в raw, чтобы поверить, что
        # этот ИНН относится к этой же организации.
        if name and len(name) >= 4 and name[:4] in raw_low:
            return f"{raw.rstrip(' ,;')}, ИНН {inn}"
    return raw


# ---------------------------------------------------------------------------
# Кэш
# ---------------------------------------------------------------------------


def _cache_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), "transport_parser_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _file_signature(pdf_path: str) -> str:
    try:
        st = os.stat(pdf_path)
        raw = f"{os.path.abspath(pdf_path)}|{st.st_size}|{int(st.st_mtime)}|v{CACHE_VERSION}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()
    except OSError:
        return ""


def _cache_get(pdf_path: str) -> list[ParsedRow] | None:
    sig = _file_signature(pdf_path)
    if not sig:
        return None
    cache_path = os.path.join(_cache_dir(), sig + ".json")
    try:
        with open(cache_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # Обратная совместимость с предыдущей версией кэша (одна строка).
            data = [data]
        return [ParsedRow.from_json_dict(d) for d in data]
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _cache_put(pdf_path: str, rows: list[ParsedRow]) -> None:
    sig = _file_signature(pdf_path)
    if not sig:
        return
    cache_path = os.path.join(_cache_dir(), sig + ".json")
    try:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump([r.to_json_dict() for r in rows], fh, ensure_ascii=False)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Текст
# ---------------------------------------------------------------------------


def extract_raw_text(pdf_path: str) -> str:
    """PDF → нормализованный текст с сохранением структуры строк."""
    raw = extract_best_text(pdf_path)
    return normalize_for_sections(raw)


# ---------------------------------------------------------------------------
# Парсинг
# ---------------------------------------------------------------------------


def _build_row(text: str, source: str, global_fallback: str = "") -> ParsedRow:
    sections = split_sections(text)
    fields = extract_all(sections, text)

    # Для сводных PDF (несколько ТН в одном файле) одна TN иногда
    # располагается на двух страницах, и секция «Груз» попадает в
    # соседний splitter-документ. Если cargo/volume не нашлись в
    # своём doc, делаем последнюю попытку в полном тексте PDF.
    if global_fallback and global_fallback != text:
        from .fields import extract_cargo, extract_volume  # локальный импорт
        if fields["cargo"][0] in (MISSING, GARBAGE):
            val, _ = extract_cargo("", global_fallback)
            if val not in (MISSING, GARBAGE):
                fields["cargo"] = (val, 0.3)
        if fields["volume"][0] in (MISSING, GARBAGE):
            val, _ = extract_volume("", global_fallback)
            if val not in (MISSING, GARBAGE):
                fields["volume"] = (val, 0.3)

    row = ParsedRow(source=source)
    row.number = fields["number"][0]
    row.date = fields["date"][0]
    row.shipper = fields["shipper"][0]
    row.consignee = fields["consignee"][0]
    row.cargo = fields["cargo"][0]
    row.volume = fields["volume"][0]
    row.driver = fields["driver"][0]
    row.vehicle = fields["vehicle"][0]
    row.reception = fields["reception"][0]

    row.confidence = FieldConfidence(
        date=fields["date"][1],
        number=fields["number"][1],
        shipper=fields["shipper"][1],
        consignee=fields["consignee"][1],
        cargo=fields["cargo"][1],
        volume=fields["volume"][1],
        driver=fields["driver"][1],
        vehicle=fields["vehicle"][1],
        reception=fields["reception"][1],
    )

    if row.number not in (MISSING, GARBAGE):
        row.waybill = f"Транспортная накладная № {row.number}"
    else:
        row.waybill = "Транспортная накладная"

    notes = []
    if len(text) < LOW_TEXT_THRESHOLD:
        notes.append("LOW_TEXT")
    if row.confidence.overall() < 0.4:
        notes.append("LOW_CONF")
    row.note = ";".join(notes)

    # LLM-fallback при низкой уверенности. No-op без ANTHROPIC_API_KEY
    # и без пакета `anthropic` — парсер работает как раньше.
    if row.confidence.overall() < 0.4:
        try:
            from .llm_fallback import improve_row
            row, _ = improve_row(row, global_fallback or text)
        except Exception:  # pragma: no cover — никогда не ломаем pipeline
            pass

    return row


def _is_noise_row(row: ParsedRow) -> bool:
    """Мусорный row: ни одного «опорного» поля не заполнено.

    Применяется только к multi-doc случаю, чтобы не создавать пустые
    строки из UPD / REGISTRY / BLANK страниц сводных PDF. Для
    одиночных ТН (len(documents) == 1) фильтр не активируется — row
    с пустыми полями всё равно выводится (чтобы пользователь видел,
    что документ был обработан).

    Опорные поля: number, date, vehicle, driver, shipper — если хотя
    бы одно не MISSING, row считается осмысленным.
    """
    signals = (
        (row.number not in (MISSING, GARBAGE, ""))
        + (row.date not in (MISSING, GARBAGE, ""))
        + (row.vehicle not in (MISSING, GARBAGE, ""))
        + (row.driver not in (MISSING, GARBAGE, ""))
        + (row.shipper not in (MISSING, GARBAGE, ""))
    )
    return signals == 0


def parse_text(text: str, source: str) -> list[ParsedRow]:
    """Парсит нормализованный текст, возвращая одну или несколько строк."""
    if not text or not text.strip():
        return [ParsedRow.empty_missing(source, note="LOW_TEXT")]

    documents = split_documents(text)
    # Для сводных PDF (несколько ТН) передаём полный текст как fallback
    # для cargo/volume — иначе груз, попавший в чужой doc, теряется.
    global_fallback = text if len(documents) > 1 else ""
    rows: list[ParsedRow] = []
    for i, doc in enumerate(documents):
        row_source = source if len(documents) == 1 else f"{source}#{i + 1}"
        row = _build_row(doc, row_source, global_fallback)
        # Фильтр мусора для сводных PDF: UPD / REGISTRY / BLANK страницы
        # часто создают пустые row'ы с конф 0. Не загрязняем Excel.
        if len(documents) > 1 and _is_noise_row(row):
            continue
        rows.append(row)
    # Если все отфильтрованы — вернём хотя бы первый (fallback-страховка).
    if not rows and documents:
        rows.append(_build_row(documents[0], source, global_fallback))
    return rows


def process_one_pdf(pdf_path: str, use_cache: bool = True) -> list[ParsedRow]:
    """Полный цикл обработки одного PDF. Возвращает список ParsedRow."""
    fname = os.path.basename(pdf_path)

    if use_cache:
        cached = _cache_get(pdf_path)
        if cached is not None:
            return cached

    try:
        text = extract_raw_text(pdf_path)
        rows = parse_text(text, fname)
    except Exception as exc:  # noqa: BLE001
        return [ParsedRow.empty_missing(fname, note=f"ERROR: {exc}")]

    if use_cache:
        _cache_put(pdf_path, rows)
    return rows


# ---------------------------------------------------------------------------
# Пакетная обработка (общая для GUI и CLI)
# ---------------------------------------------------------------------------


def iter_pdfs(input_path: str) -> list[str]:
    """Принимает папку или один PDF, возвращает отсортированный список путей."""
    if os.path.isfile(input_path):
        return [input_path] if input_path.lower().endswith(".pdf") else []
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(".pdf")
            and os.path.isfile(os.path.join(input_path, f))
        )
    return []


def process_batch(
    pdfs: list[str],
    use_cache: bool = True,
    max_workers: int | None = None,
    progress=None,
) -> dict[str, list[ParsedRow]]:
    """Пакетная обработка: возвращает dict {pdf_path: [ParsedRow, ...]}.

    `progress` — опциональный callable(done, total) для обновления UI.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers is None:
        max_workers = min(8, max(2, (os.cpu_count() or 2)))

    results: dict[str, list[ParsedRow]] = {}
    total = len(pdfs)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_to_path = {
            pool.submit(process_one_pdf, p, use_cache): p for p in pdfs
        }
        for fut in as_completed(fut_to_path):
            p = fut_to_path[fut]
            try:
                results[p] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[p] = [
                    ParsedRow.empty_missing(os.path.basename(p), note=f"ERROR: {exc}")
                ]
            done += 1
            if progress is not None:
                with contextlib.suppress(Exception):
                    progress(done, total, p, results[p])

    # Сохраняем исходный порядок.
    return {p: results[p] for p in pdfs if p in results}
