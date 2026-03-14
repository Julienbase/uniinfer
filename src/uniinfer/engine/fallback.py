"""Hardware fallback chain — retry model loading on alternative devices."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from uniinfer.hal.health import DeviceHealthReport, DeviceStatus, check_device_health
from uniinfer.hal.interface import DeviceInfo, DeviceType

logger = logging.getLogger(__name__)

# Fallback priority: CUDA → ROCm → Vulkan → CPU
_FALLBACK_ORDER = [DeviceType.CUDA, DeviceType.ROCM, DeviceType.VULKAN, DeviceType.CPU]


@dataclass
class FallbackEvent:
    """Record of a single fallback attempt."""

    from_device: str
    to_device: str
    reason: str
    success: bool


@dataclass
class FallbackResult:
    """Result of the fallback chain execution."""

    final_device: DeviceInfo
    fell_back: bool
    events: list[FallbackEvent] = field(default_factory=list)
    health_reports: dict[str, DeviceHealthReport] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if not self.fell_back:
            return f"Loaded on {self.final_device.device_string} (no fallback needed)"
        chain = " → ".join(e.from_device for e in self.events)
        chain += f" → {self.final_device.device_string}"
        return f"Fallback chain: {chain}"


def build_fallback_chain(
    preferred: DeviceInfo,
    available: list[DeviceInfo],
    exclude: Optional[set[str]] = None,
) -> list[DeviceInfo]:
    """Build an ordered fallback chain starting from the preferred device.

    Args:
        preferred: The initially selected device.
        available: All discovered devices.
        exclude: Set of device_string values to skip.

    Returns:
        Ordered list of devices to try, starting with preferred.
    """
    if exclude is None:
        exclude = set()

    chain = [preferred]
    seen = {preferred.device_string}

    # Add remaining devices in fallback priority order
    for device_type in _FALLBACK_ORDER:
        for dev in available:
            if dev.device_string in seen or dev.device_string in exclude:
                continue
            if dev.device_type == device_type:
                chain.append(dev)
                seen.add(dev.device_string)

    return chain


def try_with_fallback(
    preferred: DeviceInfo,
    available: list[DeviceInfo],
    load_fn: Any,
    check_health: bool = True,
) -> tuple[Any, DeviceInfo, FallbackResult]:
    """Try loading a model with fallback to alternative devices.

    Attempts to load using `load_fn(device)` on the preferred device first.
    If it fails, tries each device in the fallback chain.

    Args:
        preferred: The initially selected device.
        available: All discovered devices.
        load_fn: Callable that takes a DeviceInfo and returns a loaded model.
                 Should raise on failure.
        check_health: Whether to run health checks before attempting load.

    Returns:
        Tuple of (loaded_result, final_device, fallback_result).

    Raises:
        RuntimeError: If all devices in the fallback chain fail.
    """
    chain = build_fallback_chain(preferred, available)
    result = FallbackResult(final_device=preferred, fell_back=False)
    last_error: Optional[Exception] = None

    for i, device in enumerate(chain):
        # Health check (skip for CPU — always healthy enough to try)
        if check_health and device.device_type != DeviceType.CPU:
            health = check_device_health(device)
            result.health_reports[device.device_string] = health

            if health.status == DeviceStatus.UNAVAILABLE:
                logger.warning(
                    "Skipping %s: %s", device.device_string, health.message
                )
                if i > 0:
                    result.events.append(FallbackEvent(
                        from_device=chain[i - 1].device_string if i > 0 else "none",
                        to_device=device.device_string,
                        reason=health.message,
                        success=False,
                    ))
                continue

        # Attempt load
        try:
            logger.info("Attempting to load model on %s...", device.device_string)
            loaded = load_fn(device)

            result.final_device = device
            result.fell_back = i > 0

            if i > 0:
                result.events.append(FallbackEvent(
                    from_device=chain[i - 1].device_string,
                    to_device=device.device_string,
                    reason="previous device failed",
                    success=True,
                ))
                logger.info(
                    "Fallback successful: loaded on %s", device.device_string
                )

            return loaded, device, result

        except Exception as exc:
            last_error = exc
            logger.warning(
                "Failed to load on %s: %s", device.device_string, exc
            )

            if i < len(chain) - 1:
                next_dev = chain[i + 1]
                result.events.append(FallbackEvent(
                    from_device=device.device_string,
                    to_device=next_dev.device_string,
                    reason=str(exc),
                    success=False,
                ))
                logger.info(
                    "Falling back from %s → %s",
                    device.device_string,
                    next_dev.device_string,
                )

    # All devices failed
    device_list = ", ".join(d.device_string for d in chain)
    raise RuntimeError(
        f"Failed to load model on any device. Tried: {device_list}\n"
        f"Last error: {last_error}"
    )
