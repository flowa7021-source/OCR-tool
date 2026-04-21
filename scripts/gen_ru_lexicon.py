"""Build-time генерация Russian-словаря для fuzzy OCR-коррекции.

Стратегия (апрель 2026): берём ВСЕ формы 5-10 chars из pymorphy3
OpenCorpora-словаря. Это ~1.3M уникальных форм, покрывает
подавляющее большинство словарного запаса бизнес-текстов на
русском. Файл ~23 MB, загружается за ~400ms на первый call
fuzzy_corrector'а (rapidfuzz процесс-native C-ext быстро
indexирует).

Почему 5-10 chars:
  * < 5 chars — noise для fuzzy (случайные 1-edit match'и дают
    false-positives, "не" → "на" и т.п.);
  * > 10 chars — редко встречаются в OCR-ящихся полях ТН
    (названия организаций типа «Моспроект-3» больше но там другое
    исправление через catalog);
  * в этом диапазоне лежит ~1.3M форм из 3M всех в OpenCorpora.

Runtime pymorphy3 НЕ требуется — fuzzy_corrector читает готовый
.txt. Pymorphy3 используется только этим скриптом + morphology_
validator.py (для post-match validation).

Запуск:
    pip install pymorphy3 pymorphy3-dicts-ru  # build-time
    python scripts/gen_ru_lexicon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Минимальная / максимальная длина формы для inclusion в словарь.
# 5-12 — sweet spot: покрывает «организация»/«организации» (11 chars),
# «предприятие» (11), «документация» (12). Ограничение до 10 оставляло
# за бортом ключевые business-слова, и rapidfuzz возвращал близкие
# но неправильные substring'и («организма» вместо «организация»).
_MIN_LEN = 5
_MAX_LEN = 12


def generate() -> set[str]:
    try:
        import pymorphy3
    except ImportError:
        sys.exit(
            "Установите pymorphy3: pip install pymorphy3 pymorphy3-dicts-ru\n"
            "Это build-time зависимость; runtime её не требует."
        )
    morph = pymorphy3.MorphAnalyzer()
    out: set[str] = set()
    for item in morph.dictionary.iter_known_words():
        w = item[0]
        if _MIN_LEN <= len(w) <= _MAX_LEN:
            out.add(w.lower())
    print(f"Generated forms ({_MIN_LEN}-{_MAX_LEN} chars): {len(out)}")
    return out


def main() -> int:
    forms = generate()
    out_path = _REPO_ROOT / "resources" / "ru_lexicon.txt"
    out_path.write_text("\n".join(sorted(forms)) + "\n", encoding="utf-8")
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {len(forms)} words to {out_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
