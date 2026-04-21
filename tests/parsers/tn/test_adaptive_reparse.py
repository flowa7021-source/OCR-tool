"""Тесты adaptive re-parse (idea #10 top-10).

Проблема: если overall_conf < threshold после первого pass'а,
парсер молча выдаёт low-conf result. Можно попытаться
переинтерпретировать text с альтернативной стратегией splitting'а
и выбрать лучший вариант.

**Lightweight вариант** (без re-running OCR):

  * pass 1: page-aware splitter (default)
  * pass 2 (если pass1 conf < 0.5): чистый anchor-based splitter
    на полном тексте, игнорируя page-разбивку
  * pass 3 (если оба < 0.5): extraction без splitter'а — каждое
    поле ищется регексами по ВСЕМУ тексту

Returns best-of-3 по overall_confidence. Бесплатно для high-conf
случаев (pass 1 уже > 0.5 — short-circuit).

**TDD-план:**

  * ``reparse_if_low_conf(text, source, catalog, threshold=0.5)``
    — entry point.
  * ``best_of_passes(passes)`` — pick tie-breaker по overall_conf.
  * high-conf pass → short-circuit без fallback'ов (для perf).
"""

from __future__ import annotations


class TestShortCircuitOnHighConf:
    """Если первый pass уже дал overall ≥ threshold, альтернативные
    strategy НЕ вызываются."""

    def test_high_conf_skips_fallback(self):
        """mock: primary = row с conf=0.9. Ожидаем что
        reparse_if_low_conf вернёт его сразу."""
        from src.tn_parser.adaptive_reparse import reparse_if_low_conf

        def primary_pass(text):
            return {"number": "1", "overall_confidence": 0.9}

        def fallback_pass(text):
            raise AssertionError("fallback must NOT be called on high-conf primary")

        result = reparse_if_low_conf(
            text="some text",
            primary_fn=primary_pass,
            fallback_fns=[fallback_pass],
            threshold=0.5,
        )
        assert result["number"] == "1"
        assert result["overall_confidence"] == 0.9


class TestFallbackPicksBest:
    """Если primary < threshold, fallback'и вызываются. Результат
    — best-of по overall_confidence."""

    def test_fallback_conf_higher_than_primary_wins(self):
        from src.tn_parser.adaptive_reparse import reparse_if_low_conf

        def primary(text):
            return {"marker": "primary", "overall_confidence": 0.3}

        def fallback1(text):
            return {"marker": "fallback1", "overall_confidence": 0.7}

        def fallback2(text):
            return {"marker": "fallback2", "overall_confidence": 0.5}

        result = reparse_if_low_conf(
            text="text",
            primary_fn=primary,
            fallback_fns=[fallback1, fallback2],
            threshold=0.5,
        )
        assert result["marker"] == "fallback1"
        assert result["overall_confidence"] == 0.7

    def test_all_low_conf_picks_highest(self):
        """Все passes < threshold — всё равно возвращаем лучший
        по conf (best available, а не fail)."""
        from src.tn_parser.adaptive_reparse import reparse_if_low_conf

        def primary(text):
            return {"marker": "primary", "overall_confidence": 0.2}

        def fallback(text):
            return {"marker": "fallback", "overall_confidence": 0.35}

        result = reparse_if_low_conf(
            text="text", primary_fn=primary,
            fallback_fns=[fallback], threshold=0.5,
        )
        assert result["marker"] == "fallback"

    def test_primary_better_than_fallback_primary_wins(self):
        from src.tn_parser.adaptive_reparse import reparse_if_low_conf

        def primary(text):
            return {"marker": "primary", "overall_confidence": 0.4}

        def fallback(text):
            return {"marker": "fallback", "overall_confidence": 0.35}

        result = reparse_if_low_conf(
            text="text", primary_fn=primary,
            fallback_fns=[fallback], threshold=0.5,
        )
        # Primary 0.4 > fallback 0.35, primary wins.
        assert result["marker"] == "primary"


class TestFallbackExceptionSafety:
    """Fallback-функции не должны ронять pipeline своими
    exception'ами — логируем и идём к следующему."""

    def test_fallback_raises_is_skipped(self):
        from src.tn_parser.adaptive_reparse import reparse_if_low_conf

        def primary(text):
            return {"marker": "primary", "overall_confidence": 0.3}

        def broken_fallback(text):
            raise RuntimeError("boom")

        def good_fallback(text):
            return {"marker": "good", "overall_confidence": 0.6}

        result = reparse_if_low_conf(
            text="text",
            primary_fn=primary,
            fallback_fns=[broken_fallback, good_fallback],
            threshold=0.5,
        )
        assert result["marker"] == "good"

    def test_all_fallbacks_fail_returns_primary(self):
        from src.tn_parser.adaptive_reparse import reparse_if_low_conf

        def primary(text):
            return {"marker": "primary", "overall_confidence": 0.2}

        def broken1(text):
            raise RuntimeError("boom1")

        def broken2(text):
            raise ValueError("boom2")

        result = reparse_if_low_conf(
            text="text",
            primary_fn=primary,
            fallback_fns=[broken1, broken2],
            threshold=0.5,
        )
        # Всё упало — primary всё равно возвращается.
        assert result["marker"] == "primary"
