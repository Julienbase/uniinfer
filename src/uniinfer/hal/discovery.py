"""Hardware discovery — probes all available device adapters."""

from __future__ import annotations

import logging
from typing import Optional

from uniinfer.hal.cpu_adapter import CpuAdapter
from uniinfer.hal.cuda_adapter import CudaAdapter
from uniinfer.hal.interface import DeviceInfo, DeviceType
from uniinfer.hal.rocm_adapter import RocmAdapter
from uniinfer.hal.vulkan_adapter import VulkanAdapter

logger = logging.getLogger(__name__)

# Probe order: CUDA -> ROCm -> Vulkan -> CPU
_ADAPTER_CLASSES = [
    (DeviceType.CUDA, CudaAdapter),
    (DeviceType.ROCM, RocmAdapter),
    (DeviceType.VULKAN, VulkanAdapter),
    (DeviceType.CPU, CpuAdapter),
]


def devices() -> list[DeviceInfo]:
    """Discover all available hardware devices.

    Probes in order: CUDA -> ROCm -> Vulkan -> CPU.
    CPU is always included as a fallback.

    Returns:
        List of DeviceInfo for all discovered devices.
    """
    all_devices: list[DeviceInfo] = []

    for device_type, adapter_cls in _ADAPTER_CLASSES:
        try:
            adapter = adapter_cls()
            found = adapter.get_all_devices()
            if found:
                logger.info("Found %d %s device(s)", len(found), device_type.value)
                all_devices.extend(found)
        except Exception as exc:
            logger.debug("Error probing %s: %s", device_type.value, exc)

    return all_devices


def select_best_device(
    preferred: str = "auto",
    available: Optional[list[DeviceInfo]] = None,
) -> DeviceInfo:
    """Select the best available device for inference.

    Args:
        preferred: Device preference. "auto" picks the best GPU, falling back to CPU.
                   Can also be "cuda:0", "rocm:0", "vulkan:0", "cpu", etc.
        available: Pre-discovered device list. If None, runs discovery.

    Returns:
        The selected DeviceInfo.

    Raises:
        RuntimeError: If no devices are found or the requested device is unavailable.
    """
    if available is None:
        available = devices()

    if not available:
        raise RuntimeError("No hardware devices discovered. This should never happen (CPU always available).")

    # Parse device preference
    if preferred == "auto":
        return _auto_select(available)

    # Parse "type:id" format
    parts = preferred.split(":")
    device_type_str = parts[0].lower()
    device_id = int(parts[1]) if len(parts) > 1 else 0

    try:
        target_type = DeviceType(device_type_str)
    except ValueError:
        raise RuntimeError(
            f"Unknown device type '{device_type_str}'. "
            f"Supported types: {', '.join(t.value for t in DeviceType)}"
        )

    # Find matching device
    for dev in available:
        if dev.device_type == target_type and dev.device_id == device_id:
            return dev

    raise RuntimeError(
        f"Requested device '{preferred}' not found. "
        f"Available devices: {[d.device_string for d in available]}"
    )


def _auto_select(available: list[DeviceInfo]) -> DeviceInfo:
    """Automatically select the best device.

    Priority: CUDA > ROCm > Vulkan > CPU.
    Within same type, prefer the device with the most free memory.
    """
    priority = {
        DeviceType.CUDA: 0,
        DeviceType.ROCM: 1,
        DeviceType.VULKAN: 2,
        DeviceType.CPU: 3,
    }

    sorted_devices = sorted(
        available,
        key=lambda d: (priority.get(d.device_type, 99), -d.free_memory),
    )

    selected = sorted_devices[0]
    logger.info("Auto-selected device: %s (%s)", selected.name, selected.device_string)
    return selected
