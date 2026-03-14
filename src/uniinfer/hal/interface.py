"""Hardware Abstraction Layer interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DeviceType(str, Enum):
    """Supported device types."""

    CUDA = "cuda"
    ROCM = "rocm"
    VULKAN = "vulkan"
    CPU = "cpu"


@dataclass(frozen=True)
class DeviceInfo:
    """Information about a discovered hardware device."""

    name: str
    device_type: DeviceType
    device_id: int
    total_memory: int  # bytes
    free_memory: int  # bytes
    compute_capability: Optional[tuple[int, int]] = None  # (major, minor), CUDA only
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory / (1024**3)

    @property
    def free_memory_gb(self) -> float:
        return self.free_memory / (1024**3)

    @property
    def device_string(self) -> str:
        """Return a device string like 'cuda:0' or 'cpu:0'."""
        return f"{self.device_type.value}:{self.device_id}"


class DeviceAdapter(ABC):
    """Abstract base class for hardware device adapters.

    Each adapter is responsible for discovering and querying devices
    of a specific type (CUDA, ROCm, Vulkan, CPU).
    """

    @abstractmethod
    def get_device_count(self) -> int:
        """Return the number of available devices of this type."""
        ...

    @abstractmethod
    def get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """Return detailed information about a specific device."""
        ...

    @abstractmethod
    def get_free_memory(self, device_id: int = 0) -> int:
        """Return free memory in bytes for the specified device."""
        ...

    def get_all_devices(self) -> list[DeviceInfo]:
        """Return info for all devices of this type."""
        devices: list[DeviceInfo] = []
        count = self.get_device_count()
        for i in range(count):
            try:
                devices.append(self.get_device_info(i))
            except Exception:
                continue
        return devices
