"""CPU device discovery using psutil and platform."""

from __future__ import annotations

import logging
import platform

import psutil

from uniinfer.hal.interface import DeviceAdapter, DeviceInfo, DeviceType

logger = logging.getLogger(__name__)


class CpuAdapter(DeviceAdapter):
    """CPU device adapter. Always available.

    Uses psutil for memory info and platform for CPU identification.
    """

    def __init__(self) -> None:
        self._cpu_name = self._detect_cpu_name()
        logger.debug("CPU adapter initialized: %s", self._cpu_name)

    @staticmethod
    def _detect_cpu_name() -> str:
        """Detect the CPU model name."""
        # platform.processor() often returns useful info
        proc = platform.processor()
        if proc and proc.strip() and proc.strip() not in ("", "x86_64", "AMD64", "aarch64"):
            return proc.strip()

        # Fallback: try reading from /proc/cpuinfo on Linux
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except (FileNotFoundError, PermissionError, OSError):
            pass

        # Fallback: use platform info
        machine = platform.machine()
        system = platform.system()
        return f"{system} {machine} CPU"

    def get_device_count(self) -> int:
        return 1  # CPU is always a single logical device

    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        if device_id != 0:
            raise RuntimeError(f"CPU device {device_id} does not exist (only device 0)")

        mem = psutil.virtual_memory()
        core_count = psutil.cpu_count(logical=False) or 1
        thread_count = psutil.cpu_count(logical=True) or 1

        return DeviceInfo(
            name=self._cpu_name,
            device_type=DeviceType.CPU,
            device_id=0,
            total_memory=mem.total,
            free_memory=mem.available,
            extra={
                "cores": str(core_count),
                "threads": str(thread_count),
                "architecture": platform.machine(),
            },
        )

    def get_free_memory(self, device_id: int = 0) -> int:
        return psutil.virtual_memory().available  # type: ignore[no-any-return]
