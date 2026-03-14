"""Tests for device health checking."""

from __future__ import annotations

import pytest

from uniinfer.hal.health import (
    DeviceHealthReport,
    DeviceStatus,
    check_device_health,
)
from uniinfer.hal.interface import DeviceInfo, DeviceType


def _make_device(
    device_type: DeviceType = DeviceType.CPU,
    free_gb: float = 16.0,
    device_id: int = 0,
) -> DeviceInfo:
    return DeviceInfo(
        name="Test Device",
        device_type=device_type,
        device_id=device_id,
        total_memory=int(free_gb * 1.2 * 1024**3),
        free_memory=int(free_gb * 1024**3),
    )


class TestCPUHealthCheck:
    def test_healthy_cpu(self) -> None:
        device = _make_device(DeviceType.CPU, free_gb=16.0)
        report = check_device_health(device)
        assert report.status == DeviceStatus.HEALTHY
        assert report.can_allocate is True

    def test_low_ram_cpu(self) -> None:
        device = _make_device(DeviceType.CPU, free_gb=0.3)
        report = check_device_health(device)
        assert report.status == DeviceStatus.DEGRADED
        assert report.can_allocate is True  # Still usable, just degraded


class TestCUDAHealthCheck:
    def test_cuda_without_pynvml(self) -> None:
        """CUDA health check when pynvml is not available or device doesn't exist."""
        device = _make_device(DeviceType.CUDA, free_gb=8.0, device_id=99)
        report = check_device_health(device)
        # Either HEALTHY (if pynvml works and device exists) or UNAVAILABLE
        assert report.status in (DeviceStatus.HEALTHY, DeviceStatus.UNAVAILABLE)


class TestROCmHealthCheck:
    def test_rocm_without_tool(self) -> None:
        """ROCm health check when rocm-smi is not installed."""
        device = _make_device(DeviceType.ROCM, free_gb=8.0)
        report = check_device_health(device)
        # On systems without ROCm, should be UNAVAILABLE
        assert report.status in (DeviceStatus.HEALTHY, DeviceStatus.UNAVAILABLE)


class TestVulkanHealthCheck:
    def test_vulkan_without_tool(self) -> None:
        """Vulkan health check when vulkaninfo is not installed."""
        device = _make_device(DeviceType.VULKAN, free_gb=8.0)
        report = check_device_health(device)
        assert report.status in (DeviceStatus.HEALTHY, DeviceStatus.UNAVAILABLE)


class TestDeviceHealthReport:
    def test_report_fields(self) -> None:
        device = _make_device()
        report = DeviceHealthReport(
            device=device,
            status=DeviceStatus.HEALTHY,
            message="All good",
        )
        assert report.device is device
        assert report.status == DeviceStatus.HEALTHY
        assert report.message == "All good"
        assert report.can_allocate is True

    def test_unhealthy_report(self) -> None:
        device = _make_device()
        report = DeviceHealthReport(
            device=device,
            status=DeviceStatus.UNAVAILABLE,
            message="Driver crashed",
            can_allocate=False,
        )
        assert report.can_allocate is False
