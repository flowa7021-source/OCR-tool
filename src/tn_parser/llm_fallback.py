"""LLM-fallback через Claude API для полей с низкой уверенностью.

Когда regex/эвристический парсер не справляется (overall confidence < 0.4),
отправляем полный OCR-текст документа в Claude и просим извлечь поля
структурированно. Применяем только к полям, которые у нас пустые /
MISSING — чтобы не затирать уже найденное.

Модуль спроектирован как консервативный и безопасный:

* Без пакета `anthropic` или без `ANTHROPIC_API_KEY` → no-op, парсер
  работает как раньше.
* Вызов обёрнут в try/except — любая ошибка API тихо возвращает row
  без изменений (не ломаем pipeline пользователя).
* Используется `prompt caching` на system-prompt (он стабильный)
  и `adaptive thinking` — Claude сам решает, сколько думать.
* `messages.parse()` + Pydantic гарантирует валидную структуру ответа.

Интеграция в core._build_row после всех regex-извлечений:

    if row.confidence.overall() < 0.4:
        row, _ = improve_row(row, text)
"""

from __future__ import annotations

import os

try:
    from pydantic import BaseModel
    _HAS_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAS_PYDANTIC = False
    BaseModel = object  # type: ignore

from .models import GARBAGE, MISSING, ParsedRow

_LOW_CONF_THRESHOLD = 0.4
_LLM_CONF = 0.6  # уверенность, которую присваиваем полю, выставленному LLM


if _HAS_PYDANTIC:

    class LLMExtract(BaseModel):
        """Поля, извлечённые LLM-fallback'ом. Все опциональны — если
        Claude не нашёл поле в тексте, возвращает null."""

        number: str | None = None
        date: str | None = None  # DD.MM.YYYY
        shipper: str | None = None
        consignee: str | None = None
        cargo: str | None = None
        volume: str | None = None
        driver: str | None = None
        vehicle: str | None = None
        reception: str | None = None


_SYSTEM_PROMPT = """Ты — экстрактор полей транспортных накладных (ТН) из сырого OCR-текста. OCR часто ломает кириллицу (Грузоотправитель → Грузовтиравитель, Транспортная накладная → Tpaucnopfnan Haxnagnan), смешивает кириллицу с латиницей (000 вместо ООО, Tpys вместо Груз) и теряет буквы. Твоя задача — узнавать содержание несмотря на искажения.

Контракт полей:

- number: значение после «№» / «No» (пример: «7145/Б», «2908-23А»). Не путать с «Приложение № 4» из шапки формы.
- date: DD.MM.YYYY (пример: «29.08.2022»).
- shipper: «ORG_FORM «NAME», ИНН XXX» — от ORG-формы (ООО/АО/ИП/ПАО/ЗАО/…) до «ИНН \\d{10,12}» ВКЛЮЧИТЕЛЬНО. КПП/ОГРН/ОКПО/БИК обрезать. OCR-искажения «ИНИ»/«HHH»/«И Н Н» относятся к ИНН.
- consignee: «ORG_FORM «NAME», адрес» — без ИНН, без КПП, без ОГРН.
- cargo: наименование товара БЕЗ колонки «Кол-во мест», без хвоста «N шт», без «Класс опасности», без «Упаковка», без «Тара».
- volume: склейка через «, »: «N мест» (если есть в «Кол-во мест — N») + «M шт» (если есть «M шт» в наименовании) + «Нетто — X т., Брутто — Y т., Объём — Z м³» (если есть). Если ничего не нашёл — null.
- driver: ТОЛЬКО ФИО водителя (обычно правая колонка графы 6 «Перевозчик»). Формат: «Фамилия И.О.» или «Фамилия Имя Отчество». Название компании-перевозчика («ООО ДЕЛОВЫЕ ПЕРЕВОЗКИ», «ИП Иванов») в driver НЕ попадает.
- vehicle: «МАРКА\\nГРЗ_слитно» — марка, перенос строки, ГРЗ без пробелов (пример: «Scania\\nС201ВХ152», не «Scania С 201 ВХ 152»).
- reception: первая содержательная строка графы 8 «Приём груза»: «ORG_FORM «NAME», ИНН XXX». От ORG до ИНН включительно, КПП обрезается.

Правила:
1. Если поля нет в тексте — верни null. НЕ выдумывай данные.
2. Восстанавливай OCR-искажения: «Бекам» и «Беком» — одна и та же компания; латинская C часто означает кириллическую С; «З» в цифрах чаще всего «3».
3. Если в тексте несколько ТН — бери данные ПЕРВОЙ.
4. ГРЗ выдавай слитно, БЕЗ пробелов: «С201ВХ152».
"""


_USER_TEMPLATE = """Сырой OCR-текст транспортной накладной:

<ocr_text>
{text}
</ocr_text>

Извлеки поля согласно контракту. Поля, которых нет в тексте, оставь null."""


def _is_low_conf(row: ParsedRow) -> bool:
    return row.confidence.overall() < _LOW_CONF_THRESHOLD


def improve_row(
    row: ParsedRow,
    full_text: str,
    *,
    model: str = "claude-opus-4-7",
    force: bool = False,
    max_text_chars: int = 20000,
) -> tuple[ParsedRow, bool]:
    """Дополнить row через Claude API, если доступен.

    Args:
        row: уже заполненный регекс-парсером ParsedRow.
        full_text: полный нормализованный OCR-текст документа.
        model: model ID (по умолчанию Opus 4.7).
        force: вызвать API даже если confidence выше порога.
        max_text_chars: лимит длины OCR-текста (защита от слишком
            больших сканов).

    Returns:
        (row, used_llm) — row тот же объект (изменяется на месте,
        только MISSING/пустые поля), used_llm=True если вызов прошёл.
    """
    if not force and not _is_low_conf(row):
        return row, False
    if not os.getenv("ANTHROPIC_API_KEY"):
        return row, False
    if not _HAS_PYDANTIC:
        return row, False

    # Ленивый импорт: пакет `anthropic` может быть не установлен в
    # minimal-deployment. Тогда fallback тихо отключён.
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return row, False

    client = anthropic.Anthropic()
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(
                        text=full_text[:max_text_chars]
                    ),
                }
            ],
            output_format=LLMExtract,
        )
        extracted = response.parsed_output
    except Exception:
        # Любая ошибка API/сети/квоты — продолжаем без LLM.
        return row, False

    applied = False
    for field in ("number", "date", "shipper", "consignee", "cargo",
                  "volume", "driver", "vehicle", "reception"):
        current = getattr(row, field, "") or ""
        new_val = getattr(extracted, field, None)
        if not new_val:
            continue
        # Применяем только если своё значение отсутствует/GARBAGE.
        if current in (MISSING, GARBAGE, ""):
            setattr(row, field, new_val.strip())
            setattr(row.confidence, field, _LLM_CONF)
            applied = True

    if applied and row.note:
        row.note = (row.note + ";LLM_FALLBACK") if "LLM_FALLBACK" not in row.note else row.note
    elif applied:
        row.note = "LLM_FALLBACK"

    return row, applied
