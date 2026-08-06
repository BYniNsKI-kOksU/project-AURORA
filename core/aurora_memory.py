"""Shared adaptive RAM-pressure controller for AURORA programs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass

try:
    import psutil
except ImportError:  # Keep every renderer usable in a minimal environment.
    psutil = None


GIB = 2**30
DEFAULT_RESERVE_FRACTION = 0.15
MINIMUM_RESERVE_GIB = 2.0
MAXIMUM_RESERVE_GIB = 8.0
DEFAULT_MEMORY_WAIT_SECONDS = 0.5
PROCESS_PRESSURE_FRACTION = 0.85
BLOCK_GROWTH_PROCESS_FRACTION = 0.60
BLOCK_GROWTH_AVAILABLE_RESERVE_MULTIPLIER = 1.5
BLOCK_GROWTH_FACTOR = 1.25
FALLBACK_AVAILABLE_MEMORY_FRACTION = 0.5
DARWIN_FALLBACK_PAGE_SIZE_BYTES = 4096


def _fallback_memory_state() -> tuple[int, int]:
    """Return conservative RSS/available-memory estimates without psutil."""
    process_bytes = 0
    if sys.platform.startswith("linux"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            with open("/proc/self/statm", encoding="ascii") as statm:
                resident_pages = int(statm.read().split()[1])
            process_bytes = resident_pages * page_size
        except (OSError, IndexError, ValueError):
            pass
    else:
        try:
            import resource

            maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            process_bytes = (
                maximum_rss if sys.platform == "darwin" else maximum_rss * 1024
            )
        except (ImportError, OSError, ValueError):
            pass

    available_bytes = 0
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        available_bytes = page_size * available_pages
    except (AttributeError, OSError, ValueError):
        pass
    if available_bytes <= 0 and sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["vm_stat"],
                check=True,
                capture_output=True,
                text=True,
            )
            page_size = DARWIN_FALLBACK_PAGE_SIZE_BYTES
            pages = {}
            for line in result.stdout.splitlines():
                if "page size of" in line:
                    page_size = int(line.split("page size of", 1)[1].split()[0])
                elif ":" in line:
                    name, raw_value = line.split(":", 1)
                    digits = "".join(
                        character
                        for character in raw_value
                        if character.isdigit()
                    )
                    if digits:
                        pages[name.strip()] = int(digits)
            available_pages = sum(
                pages.get(name, 0)
                for name in (
                    "Pages free",
                    "Pages inactive",
                    "Pages speculative",
                    "Pages purgeable",
                )
            )
            available_bytes = available_pages * page_size
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if available_bytes <= 0:
        total_bytes = _fallback_total_memory()
        available_bytes = (
            int(total_bytes * FALLBACK_AVAILABLE_MEMORY_FRACTION)
            if total_bytes > 0
            else 0
        )
    return process_bytes, available_bytes


def _fallback_total_memory() -> int:
    """Read total physical RAM from POSIX sysconf when psutil is absent."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * page_count
    except (AttributeError, OSError, ValueError):
        return 0


def _system_memory_snapshot() -> tuple[int, int]:
    """Return total and currently available physical RAM in bytes."""
    if psutil is not None:
        state = psutil.virtual_memory()
        return int(state.total), int(state.available)
    process_bytes, available_bytes = _fallback_memory_state()
    del process_bytes
    return _fallback_total_memory(), available_bytes


def _default_ram_reserve(total_bytes: int) -> int:
    """Keep 15% for the OS, with a 2–8 GiB safety band."""
    if total_bytes <= 0:
        return int(MINIMUM_RESERVE_GIB * GIB)
    return int(
        max(
            MINIMUM_RESERVE_GIB * GIB,
            min(
                MAXIMUM_RESERVE_GIB * GIB,
                DEFAULT_RESERVE_FRACTION * total_bytes,
            ),
        )
    )


def _environment_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from error
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _optional_environment_float(name: str) -> float | None:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw_value!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass
class MemoryController:
    """Pause work and optionally shrink chunks under RAM pressure."""

    max_process_bytes: int
    min_available_bytes: int
    wait_seconds: float
    maximum_block_size: int | None = None
    minimum_block_size: int | None = None

    def __post_init__(self) -> None:
        self.block_size = self.maximum_block_size
        self.was_throttled = False

    @classmethod
    def from_environment(
        cls,
        *,
        maximum_block_size: int | None = None,
        minimum_block_size: int | None = None,
        max_ram_gb: float | None = None,
        min_free_ram_gb: float | None = None,
        wait_seconds: float | None = None,
    ) -> "MemoryController":
        """Create a controller from RAM available at program start.

        With no explicit limit, AURORA reserves 15% of physical RAM for the
        operating system and other applications (clamped to 2–8 GiB), then
        uses the rest of the currently available RAM as its process budget.
        ``AURORA_MAX_RAM_GB`` or ``max_ram_gb`` remains an optional hard cap.
        """
        total_bytes, available_bytes = _system_memory_snapshot()
        configured_max = (
            max_ram_gb
            if max_ram_gb is not None
            else _optional_environment_float("AURORA_MAX_RAM_GB")
        )
        configured_reserve = (
            min_free_ram_gb
            if min_free_ram_gb is not None
            else (
                _optional_environment_float("AURORA_MIN_FREE_RAM_GB")
                or _optional_environment_float("AURORA_RAM_RESERVE_GB")
            )
        )
        reserve_bytes = int(
            (configured_reserve * GIB)
            if configured_reserve is not None
            else _default_ram_reserve(total_bytes)
        )
        dynamic_budget = max(GIB, available_bytes - reserve_bytes)
        return cls(
            max_process_bytes=(
                int(configured_max * GIB)
                if configured_max is not None
                else dynamic_budget
            ),
            min_available_bytes=reserve_bytes,
            wait_seconds=(
                _environment_float(
                    "AURORA_MEMORY_WAIT",
                    DEFAULT_MEMORY_WAIT_SECONDS,
                )
                if wait_seconds is None
                else wait_seconds
            ),
            maximum_block_size=maximum_block_size,
            minimum_block_size=minimum_block_size,
        )

    def _memory_state(self) -> tuple[int, int, bool]:
        if psutil is None:
            process_bytes, available_bytes = _fallback_memory_state()
        else:
            available_bytes = psutil.virtual_memory().available
            process_bytes = psutil.Process().memory_info().rss
        process_near_limit = (
            process_bytes
            >= PROCESS_PRESSURE_FRACTION * self.max_process_bytes
        )
        system_near_limit = available_bytes <= self.min_available_bytes
        return (
            process_bytes,
            available_bytes,
            process_near_limit or system_near_limit,
        )

    def throttle(self) -> bool:
        """Wait once when RAM is pressured; return whether throttling occurred."""
        process_bytes, available_bytes, throttled = self._memory_state()
        if throttled:
            if not self.was_throttled:
                print(
                    "RAM pressure: slowing down "
                    f"(process {process_bytes / GIB:.2f} GB, "
                    f"available {available_bytes / GIB:.2f} GB)"
                )
            time.sleep(self.wait_seconds)
        elif self.was_throttled:
            print("RAM pressure ended: restoring normal processing speed")
        self.was_throttled = throttled
        return throttled

    def next_block_size(self) -> int:
        """Throttle and return an adaptively reduced or restored block size."""
        if self.block_size is None or self.minimum_block_size is None:
            raise RuntimeError("Adaptive block sizing is not configured")

        process_bytes, available_bytes, throttled = self._memory_state()
        if throttled:
            self.block_size = max(
                self.minimum_block_size,
                self.block_size // 2,
            )
            if not self.was_throttled:
                print(
                    "RAM pressure: slowing down "
                    f"(process {process_bytes / GIB:.2f} GB, "
                    f"available {available_bytes / GIB:.2f} GB, "
                    f"block {self.block_size:,} rows)"
                )
            time.sleep(self.wait_seconds)
        elif (
            self.block_size < self.maximum_block_size
            and process_bytes
            < BLOCK_GROWTH_PROCESS_FRACTION * self.max_process_bytes
            and available_bytes
            > BLOCK_GROWTH_AVAILABLE_RESERVE_MULTIPLIER
            * self.min_available_bytes
        ):
            self.block_size = min(
                self.maximum_block_size,
                max(
                    self.block_size + 1,
                    int(self.block_size * BLOCK_GROWTH_FACTOR),
                ),
            )
            if self.was_throttled:
                print(
                    "RAM pressure ended: increasing block to "
                    f"{self.block_size:,} rows"
                )

        self.was_throttled = throttled
        return self.block_size
