"""Runtime host-resource detection for adaptive defaults.

Picks sensible parallelism / cache defaults on low-spec hardware.
Called from ``src.app.create_application`` after settings are loaded,
before :class:`ParallelProcessor` is instantiated. Never raises —
any detection failure yields the original settings untouched.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HostResources:
    """Snapshot of host-level resource readings."""

    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    free_disk_gb: float
    # Whether the detection actually worked. ``psutil`` is a hard dep
    # of the project so this should always be True in production;
    # only falls to False in stripped-down test environments.
    detected: bool = True

    @property
    def low_memory(self) -> bool:
        """True on < 6 GB installed RAM."""
        return self.detected and 0 < self.total_ram_gb < 6.0

    @property
    def very_low_memory(self) -> bool:
        """True on < 4 GB installed RAM."""
        return self.detected and 0 < self.total_ram_gb < 4.0


def detect(root: Path | None = None) -> HostResources:
    """Read current system resources.

    ``root`` is the filesystem path used for the disk-free query;
    defaults to the user-data dir / current drive. Never raises.
    """
    try:
        import psutil  # type: ignore[import-not-found]

        vm = psutil.virtual_memory()
        cpu = psutil.cpu_count(logical=True) or 1
    except ImportError:  # pragma: no cover
        logger.debug("psutil unavailable — skipping resource detection")
        return HostResources(0.0, 0.0, 1, 0.0, detected=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("psutil read failed: %s", exc)
        return HostResources(0.0, 0.0, 1, 0.0, detected=False)

    disk_root = root or Path.home()
    try:
        du = shutil.disk_usage(str(disk_root))
        free_gb = du.free / (1024**3)
    except OSError as exc:
        logger.debug("disk_usage failed on %s: %s", disk_root, exc)
        free_gb = 0.0

    return HostResources(
        total_ram_gb=vm.total / (1024**3),
        available_ram_gb=vm.available / (1024**3),
        cpu_count=int(cpu),
        free_disk_gb=free_gb,
        detected=True,
    )


def adjust_settings_for_host(settings, host: HostResources) -> bool:
    """Modify ``settings`` in place to fit ``host`` constraints.

    Returns True if anything changed. Only downgrades — never upgrades
    the user's choices, and never touches a value that was explicitly
    saved above the auto-default (assume user knew what they picked).

    Concretely:

      * ``< 6 GB`` RAM → ``parallel_workers = 1``. Two workers at
        300 DPI peak around 1.4 GB; the OS + browser easily eat the
        remainder and trigger swap on a 4 GB laptop.
      * ``< 4 GB`` RAM AND ``ocr_cache_max_mb`` at default (2048)
        → drop cache budget to 512 MB so we don't fight the OS page
        cache for the little RAM that's left.
      * ``free_disk < 5 GB`` → cap cache at 500 MB too (cache should
        never be the thing that fills the disk).
    """
    if not host.detected:
        return False
    changed = False

    if host.low_memory and settings.parallel_workers > 1:
        logger.info(
            "Low-memory host detected (%.1f GB RAM) — dropping "
            "parallel_workers from %d to 1",
            host.total_ram_gb, settings.parallel_workers,
        )
        settings.parallel_workers = 1
        changed = True

    tight_disk = 0 < host.free_disk_gb < 5.0
    cache_on_default = settings.ocr_cache_max_mb == 2048
    if (host.very_low_memory or tight_disk) and cache_on_default:
        logger.info(
            "Constrained host (RAM=%.1f GB, free disk=%.1f GB) — "
            "lowering ocr_cache_max_mb to 512",
            host.total_ram_gb, host.free_disk_gb,
        )
        settings.ocr_cache_max_mb = 512
        changed = True

    return changed
