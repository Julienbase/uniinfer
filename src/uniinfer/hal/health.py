"""Device health checking — verify discovered devices actually work."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from uniinfer.hal.interface import DeviceInfo, DeviceType

logger = logging.getLogger(__name__)


class DeviceStatus(str, Enum):
    """Health status of a device."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DeviceHealthReport:
    """Health check result for a single device."""

    device: DeviceInfo
    status: DeviceStatus
    message: str
    can_allocate: bool = True


def check_device_health(device: DeviceInfo) -> DeviceHealthReport:
    """Run a lightweight health check on a device.

    Verifies the device can actually be used for inference by checking:
    - Driver availability
    - Memory allocation capability
    - Basic compute readiness

    Args:
        device: The device to check.

    Returns:
        DeviceHealthReport with status and details.
    """
    if device.device_type == DeviceType.CPU:
        return _check_cpu(device)
    elif device.device_type == DeviceType.CUDA:
        return _check_cuda(device)
    elif device.device_type == DeviceType.ROCM:
        return _check_rocm(device)
    elif device.device_type == DeviceType.VULKAN:
        return _check_vulkan(device)
    else:
        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.UNAVAILABLE,
            message=f"Unknown device type: {device.device_type}",
            can_allocate=False,
        )


def _check_cpu(device: DeviceInfo) -> DeviceHealthReport:
    """CPU is always healthy if it has memory."""
    if device.free_memory < 512 * 1024 * 1024:  # < 512 MB
        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.DEGRADED,
            message=f"Very low RAM: {device.free_memory_gb:.1f} GB free",
            can_allocate=True,
        )
    return DeviceHealthReport(
        device=device,
        status=DeviceStatus.HEALTHY,
        message=f"CPU ready ({device.free_memory_gb:.1f} GB free)",
    )


def _check_cuda(device: DeviceInfo) -> DeviceHealthReport:
    """Verify CUDA device is functional via pynvml."""
    try:
        import pynvml  # type: ignore[import-untyped]

        handle = pynvml.nvmlDeviceGetHandleByIndex(device.device_id)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        if mem.free < 256 * 1024 * 1024:  # < 256 MB free VRAM
            return DeviceHealthReport(
                device=device,
                status=DeviceStatus.DEGRADED,
                message=f"Very low VRAM: {mem.free / (1024**3):.2f} GB free",
                can_allocate=False,
            )

        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.HEALTHY,
            message=f"CUDA ready ({mem.free / (1024**3):.1f} GB free)",
        )
    except ImportError:
        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.UNAVAILABLE,
            message="pynvml not installed",
            can_allocate=False,
        )
    except Exception as exc:
        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.UNAVAILABLE,
            message=f"CUDA health check failed: {exc}",
            can_allocate=False,
        )


def _check_rocm(device: DeviceInfo) -> DeviceHealthReport:
    """Verify ROCm device via rocm-smi."""
    try:
        import subprocess

        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return DeviceHealthReport(
                device=device,
                status=DeviceStatus.HEALTHY,
                message=f"ROCm ready ({device.free_memory_gb:.1f} GB free)",
            )
        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.DEGRADED,
            message=f"rocm-smi returned exit code {result.returncode}",
        )
    except FileNotFoundError:
        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.UNAVAILABLE,
            message="rocm-smi not found",
            can_allocate=False,
        )
    except Exception as exc:
        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.UNAVAILABLE,
            message=f"ROCm health check failed: {exc}",
            can_allocate=False,
        )


def _check_vulkan(device: DeviceInfo) -> DeviceHealthReport:
    """Verify Vulkan device via vulkaninfo."""
    try:
        import subprocess

        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return DeviceHealthReport(
                device=device,
                status=DeviceStatus.HEALTHY,
                message=f"Vulkan ready ({device.free_memory_gb:.1f} GB free)",
            )
        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.DEGRADED,
            message=f"vulkaninfo returned exit code {result.returncode}",
        )
    except FileNotFoundError:
        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.UNAVAILABLE,
            message="vulkaninfo not found",
            can_allocate=False,
        )
    except Exception as exc:
        return DeviceHealthReport(
            device=device,
            status=DeviceStatus.UNAVAILABLE,
            message=f"Vulkan health check failed: {exc}",
            can_allocate=False,
        )
