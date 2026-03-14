"""Tests for hardware discovery with mocked adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from uniinfer.hal.discovery import devices, select_best_device
from uniinfer.hal.interface import DeviceInfo, DeviceType


class TestDevices:
    """Test hardware discovery function."""

    def test_cpu_always_found(self) -> None:
        """CPU should always be discovered."""
        found = devices()
        assert len(found) >= 1
        cpu_devices = [d for d in found if d.device_type == DeviceType.CPU]
        assert len(cpu_devices) == 1

    def test_cpu_has_memory(self) -> None:
        """CPU device should report non-zero memory."""
        found = devices()
        cpu = [d for d in found if d.device_type == DeviceType.CPU][0]
        assert cpu.total_memory > 0
        assert cpu.free_memory > 0

    @patch("uniinfer.hal.discovery.CudaAdapter")
    def test_cuda_discovery_failure_graceful(self, mock_cuda_cls: MagicMock) -> None:
        """CUDA failure should not prevent other devices from being discovered."""
        mock_cuda_cls.side_effect = RuntimeError("No NVIDIA driver")
        found = devices()
        # Should still find CPU
        assert any(d.device_type == DeviceType.CPU for d in found)


class TestSelectBestDevice:
    """Test device selection logic."""

    def test_auto_prefers_cuda(
        self,
        cuda_device_24gb: DeviceInfo,
        cpu_device: DeviceInfo,
    ) -> None:
        """Auto selection should prefer CUDA over CPU."""
        available = [cpu_device, cuda_device_24gb]
        selected = select_best_device("auto", available)
        assert selected.device_type == DeviceType.CUDA

    def test_auto_prefers_rocm_over_vulkan(
        self,
        rocm_device: DeviceInfo,
        vulkan_device: DeviceInfo,
        cpu_device: DeviceInfo,
    ) -> None:
        available = [cpu_device, vulkan_device, rocm_device]
        selected = select_best_device("auto", available)
        assert selected.device_type == DeviceType.ROCM

    def test_auto_falls_back_to_cpu(self, cpu_device: DeviceInfo) -> None:
        selected = select_best_device("auto", [cpu_device])
        assert selected.device_type == DeviceType.CPU

    def test_explicit_device_selection(
        self,
        cuda_device_24gb: DeviceInfo,
        cpu_device: DeviceInfo,
    ) -> None:
        available = [cpu_device, cuda_device_24gb]
        selected = select_best_device("cpu", available)
        assert selected.device_type == DeviceType.CPU

    def test_explicit_device_with_id(
        self,
        cuda_device_24gb: DeviceInfo,
        cpu_device: DeviceInfo,
    ) -> None:
        available = [cpu_device, cuda_device_24gb]
        selected = select_best_device("cuda:0", available)
        assert selected.device_type == DeviceType.CUDA
        assert selected.device_id == 0

    def test_unknown_device_type(self, cpu_device: DeviceInfo) -> None:
        with pytest.raises(RuntimeError, match="Unknown device type"):
            select_best_device("tpu:0", [cpu_device])

    def test_device_not_found(self, cpu_device: DeviceInfo) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            select_best_device("cuda:0", [cpu_device])

    def test_empty_device_list(self) -> None:
        with pytest.raises(RuntimeError, match="No hardware devices"):
            select_best_device("auto", [])

    def test_auto_prefers_more_memory(self) -> None:
        """When two CUDA devices exist, prefer the one with more free memory."""
        gpu_small = DeviceInfo(
            name="Small GPU",
            device_type=DeviceType.CUDA,
            device_id=0,
            total_memory=8 * 1024**3,
            free_memory=4 * 1024**3,
        )
        gpu_big = DeviceInfo(
            name="Big GPU",
            device_type=DeviceType.CUDA,
            device_id=1,
            total_memory=24 * 1024**3,
            free_memory=20 * 1024**3,
        )
        selected = select_best_device("auto", [gpu_small, gpu_big])
        assert selected.name == "Big GPU"
