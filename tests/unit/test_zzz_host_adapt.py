"""Tests for adaptive defaults on low-spec hardware.

See ``src.infrastructure.host_resources`` — we lower the parallel
worker count and OCR-cache budget on machines that would otherwise
swap themselves to death, without overriding choices the user
already made above the auto-defaults.

Named with a ``zzz_`` prefix so the file sorts after any test that
asserts a specific un-patched settings value (avoids cross-file
fixture leakage the way test_followup_audit.py already does for the
no-QApplication flake).
"""

from __future__ import annotations

from pathlib import Path


def _make_settings(
    parallel_workers: int = 2, ocr_cache_max_mb: int = 2048
):
    """Build an ``AppSettings`` with just the fields we care about."""
    from src.infrastructure.config_storage import AppSettings

    s = AppSettings()
    s.parallel_workers = parallel_workers
    s.ocr_cache_max_mb = ocr_cache_max_mb
    return s


class TestAdjustSettingsForHost:
    def test_low_memory_drops_workers_to_one(self) -> None:
        from src.infrastructure.host_resources import (
            HostResources,
            adjust_settings_for_host,
        )

        s = _make_settings(parallel_workers=2)
        host = HostResources(
            total_ram_gb=4.0, available_ram_gb=2.0,
            cpu_count=4, free_disk_gb=50.0,
        )
        assert adjust_settings_for_host(s, host) is True
        assert s.parallel_workers == 1

    def test_plenty_of_ram_leaves_workers_alone(self) -> None:
        from src.infrastructure.host_resources import (
            HostResources,
            adjust_settings_for_host,
        )

        s = _make_settings(parallel_workers=4)
        host = HostResources(
            total_ram_gb=32.0, available_ram_gb=20.0,
            cpu_count=8, free_disk_gb=500.0,
        )
        assert adjust_settings_for_host(s, host) is False
        assert s.parallel_workers == 4

    def test_user_explicit_single_worker_stays_single(self) -> None:
        """Don't bump a user who already lowered the worker count."""
        from src.infrastructure.host_resources import (
            HostResources,
            adjust_settings_for_host,
        )

        s = _make_settings(parallel_workers=1)
        host = HostResources(
            total_ram_gb=4.0, available_ram_gb=2.0,
            cpu_count=4, free_disk_gb=50.0,
        )
        # No change: downgrade is from the user-default 2 down to 1,
        # not from 1 to anything.
        adjust_settings_for_host(s, host)
        assert s.parallel_workers == 1

    def test_tight_disk_shrinks_cache_budget(self) -> None:
        from src.infrastructure.host_resources import (
            HostResources,
            adjust_settings_for_host,
        )

        s = _make_settings(parallel_workers=1, ocr_cache_max_mb=2048)
        host = HostResources(
            total_ram_gb=16.0, available_ram_gb=8.0,
            cpu_count=8, free_disk_gb=2.0,  # SSD full
        )
        adjust_settings_for_host(s, host)
        assert s.ocr_cache_max_mb == 512

    def test_custom_cache_respected(self) -> None:
        """If the user already picked a non-default cache, don't touch it."""
        from src.infrastructure.host_resources import (
            HostResources,
            adjust_settings_for_host,
        )

        s = _make_settings(parallel_workers=1, ocr_cache_max_mb=5120)
        host = HostResources(
            total_ram_gb=3.0, available_ram_gb=1.0,
            cpu_count=4, free_disk_gb=1.0,
        )
        adjust_settings_for_host(s, host)
        assert s.ocr_cache_max_mb == 5120  # user's explicit choice wins

    def test_undetected_host_is_noop(self) -> None:
        """``HostResources(detected=False)`` never triggers any change."""
        from src.infrastructure.host_resources import (
            HostResources,
            adjust_settings_for_host,
        )

        s = _make_settings(parallel_workers=4, ocr_cache_max_mb=2048)
        host = HostResources(0.0, 0.0, 1, 0.0, detected=False)
        assert adjust_settings_for_host(s, host) is False
        assert s.parallel_workers == 4


class TestDetectReturnsSensibleNumbers:
    def test_detect_does_not_raise(self) -> None:
        """psutil is a hard dependency — detect() must always return a dataclass."""
        from src.infrastructure.host_resources import detect

        host = detect()
        assert host.total_ram_gb >= 0.0
        assert host.cpu_count >= 1


class TestAppSettingsCacheField:
    def test_default_is_2_gb(self, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import SettingsStorage

        s = SettingsStorage(config_dir=tmp_path).load()
        assert s.ocr_cache_max_mb == 2048

    def test_round_trip(self, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import SettingsStorage

        storage = SettingsStorage(config_dir=tmp_path)
        cur = storage.load()
        cur.ocr_cache_max_mb = 500
        storage.save(cur)
        reloaded = SettingsStorage(config_dir=tmp_path).load()
        assert reloaded.ocr_cache_max_mb == 500

    def test_zero_disables_cache(self, tmp_path: Path) -> None:
        from src.infrastructure.config_storage import SettingsStorage

        storage = SettingsStorage(config_dir=tmp_path)
        cur = storage.load()
        cur.ocr_cache_max_mb = 0
        storage.save(cur)
        assert SettingsStorage(config_dir=tmp_path).load().ocr_cache_max_mb == 0

    def test_negative_is_clamped_to_zero(self, tmp_path: Path) -> None:
        """from_dict() must defensively clamp garbage to a safe value."""
        import json

        from src.infrastructure.config_storage import SettingsStorage

        storage = SettingsStorage(config_dir=tmp_path)
        # Seed an on-disk file so we can mutate it.
        storage.save(storage.load())
        path = storage.path
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["ocr_cache_max_mb"] = -100
        path.write_text(json.dumps(payload), encoding="utf-8")
        reloaded = SettingsStorage(config_dir=tmp_path).load()
        assert reloaded.ocr_cache_max_mb == 0
