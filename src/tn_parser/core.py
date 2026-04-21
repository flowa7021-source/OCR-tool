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


def _catalog_crossvalidate(
    raw: str, field_name: str, full_text: str = "",
) -> tuple[str, float]:
    """Cross-validate ORG-поле через ИНН-каталог.

    Если в ``raw`` найден валидный ИНН (by checksum) и в каталоге
    есть запись по нему, производим три проверки:

    1. **ИНН валиден** — базовый boost confidence +0.05.

    2. **Name fuzzy-match** — сравниваем OCR-имя (первое слово после
       ORG-префикса) с canonical-именем из каталога через rapidfuzz.
       Если score ≥ 85 — это ТА ЖЕ организация; boost +0.10.

    3. **Prepend canonical name** (только для сильно-mangled raw) —
       если OCR-имя fuzzy-мимо canonical (score < 60), но ИНН
       валиден → OCR катастрофически исказил имя. Подклеиваем
       canonical как «(каталог: ООО XYZ)» для пользователя.

    Возвращает ``(enriched_raw, conf_delta)``. Если ИНН не найден
    или каталог пуст, возвращает (raw, 0.0) без изменений.

    Применяется к полям где ИНН legitim:
      * shipper — включает ИНН, boost полный
      * reception — включает ИНН (из «Владелец инфраструктуры»)
      * consignee — БЕЗ prepend (parser design: без ИНН), но boost
        по match'у имени
    """
    if not raw or raw in (MISSING, GARBAGE):
        return raw, 0.0
    import re
    conf_delta = 0.0
    enriched = raw

    # Находим первый валидный ИНН в строке. Для consignee (по
    # design без ИНН) пробуем искать в full_text, fuzzy-match'а
    # имя из raw против каталога по канд. ИНН из full_text.
    found_inn: str | None = None
    for m in re.finditer(r"\b(\d{10}|\d{12})\b", raw):
        candidate = m.group(1)
        if is_valid_inn(candidate):
            found_inn = candidate
            break

    if not found_inn and full_text:
        # Fallback: для consignee raw без ИНН → сканируем full_text
        # и fuzzy-сравниваем имя из raw с каталогом-именем каждого
        # найденного ИНН. Если совпало — это тот же контрагент.
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return raw, 0.0
        # Нормализуем обе строки одинаково: только буквы + пробелы,
        # lowercase. Дефисы / цифры / пунктуация убираются — иначе
        # «Моспроект-3» vs «Моспроект 3» даёт partial_ratio 82 %
        # (ниже нашего 85 % threshold), хотя это очевидно та же
        # организация.
        def _norm(s: str) -> str:
            return re.sub(
                r"\s+", " ",
                re.sub(r"[^А-Яа-яЁёA-Za-z]", " ", s),
            ).strip().lower()

        raw_letters = _norm(raw)
        if not raw_letters or len(raw_letters) < 3:
            return raw, 0.0
        for m in re.finditer(r"\b(\d{10}|\d{12})\b", full_text):
            candidate = m.group(1)
            if not is_valid_inn(candidate):
                continue
            rec = lookup_by_inn(candidate)
            if not rec:
                continue
            canon_name = _norm(rec.get("name") or "")
            if not canon_name:
                continue
            if fuzz.partial_ratio(canon_name, raw_letters) >= 85:
                found_inn = candidate
                break

    if not found_inn:
        return raw, 0.0

    # ИНН прошёл контрольную сумму — это объективный сигнал
    # что OCR правильно распознал цифры (12 чисел не могут
    # случайно сложиться в валидный контрольный разряд).
    conf_delta += 0.05

    rec = lookup_by_inn(found_inn)
    if not rec:
        # Catalog не знает этот ИНН — но checksum всё равно пройден,
        # boost + держим.
        return raw, conf_delta

    canonical_name = (rec.get("name") or "").strip()
    if not canonical_name:
        return raw, conf_delta

    # Fuzzy-сравнение canonical-name с raw (OCR-строкой). Сравниваем
    # только буквенную часть — игнорируем цифры ИНН/КПП/адрес.
    raw_letters = re.sub(r"[^А-Яа-яЁёA-Za-z ]", " ", raw).strip()
    canon_letters = re.sub(r"[^А-Яа-яЁёA-Za-z ]", " ", canonical_name).strip()

    try:
        from rapidfuzz import fuzz
        score = fuzz.partial_ratio(
            canon_letters.lower(), raw_letters.lower(),
        )
    except ImportError:
        # Без rapidfuzz обойдёмся без fuzzy-boost, но ИНН-boost
        # остаётся — он не требует библиотеки.
        return enriched, conf_delta

    if score >= 85:
        # Name совпадает — сильный сигнал что OCR верно распознал
        # организацию. Boost дополнительно.
        conf_delta += 0.10
    elif score < 60 and field_name in ("shipper", "reception"):
        # OCR катастрофически исказил имя, но ИНН валиден. Добавляем
        # canonical-имя из каталога как аннотацию, чтобы пользователь
        # видел что recovery произошло.
        legal_form = (rec.get("legal_form") or "").strip()
        display = f"{legal_form} «{canonical_name}»" if legal_form else canonical_name
        enriched = f"{raw.rstrip(' ,;')} [каталог: {display}]"

    return enriched, min(conf_delta, 0.15)  # cap at +0.15 suma


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

    # Cross-validate ORG-поля через ИНН-каталог: если извлечённая
    # строка содержит валидный ИНН и он есть в каталоге, добавляем
    # canonical-name как аннотацию (для shipper/reception) и
    # повышаем confidence поля (ИНН с checksum — объективный
    # сигнал). На consignee confidence тоже boost'им, но без
    # prepend (parser design держит consignee без ИНН).
    sh_enriched, sh_delta = _catalog_crossvalidate(
        row.shipper, "shipper", text,
    )
    row.shipper = sh_enriched
    rc_enriched, rc_delta = _catalog_crossvalidate(
        row.reception, "reception", text,
    )
    row.reception = rc_enriched
    cn_enriched, cn_delta = _catalog_crossvalidate(
        row.consignee, "consignee", text,
    )
    row.consignee = cn_enriched  # consignee prepend не делается внутри

    row.confidence = FieldConfidence(
        date=fields["date"][1],
        number=fields["number"][1],
        # min(1.0, ...) — confidence не может превышать 1.0 после boost
        shipper=min(1.0, fields["shipper"][1] + sh_delta),
        consignee=min(1.0, fields["consignee"][1] + cn_delta),
        cargo=fields["cargo"][1],
        volume=fields["volume"][1],
        driver=fields["driver"][1],
        vehicle=fields["vehicle"][1],
        reception=min(1.0, fields["reception"][1] + rc_delta),
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


def _apply_multi_row_voting(rows: list[ParsedRow]) -> None:
    """Voting/consolidation across rows of one multi-TN PDF.

    В сводных PDF типа UPD_41/UPD_47/UPD_48 одна и та же
    организация (shipper / consignee) фигурирует в каждой ТН —
    и OCR часто распознаёт её с разными degrees of mangling
    на разных страницах. Voting-стратегия:

    1. Для каждого ORG-поля (shipper / consignee / reception)
       собираем non-missing значения.
    2. Группируем по ИНН (если есть) — если разные значения
       ссылаются на один ИНН, значит это одна организация и
       OCR просто разошёлся в написании имени.
    3. В каждой группе выбираем «канонический» вариант — самый
       длинный (обычно = меньше OCR-обрезки) И имеющий
       catalog-enrichment (если есть).
    4. Проставляем его всем rows группы + boost confidence на
       +0.05 (cross-row confirmation).

    Без этого: row[0].shipper = «ООО Бекам», row[1].shipper =
    «ООО Беком», row[2].shipper = «ООО Беком, ИНН 7743553262».
    Пользователь в Excel видит три разных написания одной
    организации. С voting'ом: все три становятся одним и тем
    же каноничным вариантом (с ИНН и catalog-аннотацией).

    Применяется в-place к ``rows``.
    """
    import re
    if len(rows) < 2:
        return
    for field in ("shipper", "consignee", "reception"):
        # Группировка по ИНН из raw.
        groups: dict[str, list[int]] = {}
        unkeyed: list[int] = []
        for i, row in enumerate(rows):
            val = getattr(row, field, "") or ""
            if val in (MISSING, GARBAGE, ""):
                continue
            inn_m = re.search(r"\b(\d{10}|\d{12})\b", val)
            if inn_m and is_valid_inn(inn_m.group(1)):
                groups.setdefault(inn_m.group(1), []).append(i)
            else:
                unkeyed.append(i)
        # Для каждой группы выбираем canonical.
        for _inn, indices in groups.items():
            if len(indices) < 2:
                continue
            # Canonical = самая длинная строка (длинная = меньше
            # обрезана OCR-шумом, больше контекста).
            canonical_idx = max(indices, key=lambda i: len(
                getattr(rows[i], field, "") or "",
            ))
            canonical_val = getattr(rows[canonical_idx], field)
            # Пропагируем во все rows группы.
            for i in indices:
                if getattr(rows[i], field) != canonical_val:
                    setattr(rows[i], field, canonical_val)
                # Cross-row confirmation — небольшой boost на
                # consensus. Учитываем, что максимум 1.0.
                cur = getattr(rows[i].confidence, field, 0.0)
                setattr(
                    rows[i].confidence, field, min(1.0, cur + 0.03),
                )


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
    # Cross-row voting по ORG-полям (shipper/consignee/reception).
    # На sparse multi-TN PDF даёт консистентность + conf boost.
    _apply_multi_row_voting(rows)
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
