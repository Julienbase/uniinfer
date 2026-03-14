"""Tests for hardware fallback chain."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from uniinfer.engine.fallback import (
    FallbackEvent,
    FallbackResult,
    build_fallback_chain,
    try_with_fallback,
)
from uniinfer.hal.health import DeviceHealthReport, DeviceStatus
from uniinfer.hal.interface import DeviceInfo, DeviceType


def _make_device(
    device_type: DeviceType,
    device_id: int = 0,
    free_gb: float = 8.0,
    name: str = "",
) -> DeviceInfo:
    if not name:
        name = f"{device_type.value}:{device_id}"
    return DeviceInfo(
        name=name,
        device_type=device_type,
        device_id=device_id,
        total_memory=int(free_gb * 1.2 * 1024**3),
        free_memory=int(free_gb * 1024**3),
    )


class TestBuildFallbackChain:
    def test_preferred_is_first(self) -> None:
        cuda = _make_device(DeviceType.CUDA)
        cpu = _make_device(DeviceType.CPU)
        chain = build_fallback_chain(cuda, [cuda, cpu])
        assert chain[0] is cuda

    def test_includes_all_devices(self) -> None:
        cuda = _make_device(DeviceType.CUDA)
        vulkan = _make_device(DeviceType.VULKAN)
        cpu = _make_device(DeviceType.CPU)
        chain = build_fallback_chain(cuda, [cuda, vulkan, cpu])
        assert len(chain) == 3
        assert chain[0] is cuda
        assert cpu in chain

    def test_respects_priority_order(self) -> None:
        cpu = _make_device(DeviceType.CPU)
        cuda = _make_device(DeviceType.CUDA)
        vulkan = _make_device(DeviceType.VULKAN)
        # Preferred is CPU, but chain should include GPU devices after it
        chain = build_fallback_chain(cpu, [cpu, cuda, vulkan])
        assert chain[0] is cpu
        # CUDA comes before Vulkan in fallback order
        cuda_idx = next(i for i, d in enumerate(chain) if d.device_type == DeviceType.CUDA)
        vulkan_idx = next(i for i, d in enumerate(chain) if d.device_type == DeviceType.VULKAN)
        assert cuda_idx < vulkan_idx

    def test_excludes_devices(self) -> None:
        cuda = _make_device(DeviceType.CUDA)
        vulkan = _make_device(DeviceType.VULKAN)
        cpu = _make_device(DeviceType.CPU)
        chain = build_fallback_chain(cuda, [cuda, vulkan, cpu], exclude={"vulkan:0"})
        assert vulkan not in chain
        assert len(chain) == 2

    def test_no_duplicates(self) -> None:
        cuda = _make_device(DeviceType.CUDA)
        cpu = _make_device(DeviceType.CPU)
        chain = build_fallback_chain(cuda, [cuda, cuda, cpu])
        device_strings = [d.device_string for d in chain]
        assert len(device_strings) == len(set(device_strings))


class TestTryWithFallback:
    def test_success_on_first_device(self) -> None:
        cuda = _make_device(DeviceType.CUDA)
        cpu = _make_device(DeviceType.CPU)

        def load_fn(device: DeviceInfo) -> str:
            return f"loaded_on_{device.device_string}"

        with patch("uniinfer.engine.fallback.check_device_health") as mock_health:
            mock_health.return_value = DeviceHealthReport(
                device=cuda, status=DeviceStatus.HEALTHY, message="OK"
            )
            loaded, final_device, result = try_with_fallback(
                cuda, [cuda, cpu], load_fn
            )

        assert loaded == "loaded_on_cuda:0"
        assert final_device is cuda
        assert result.fell_back is False
        assert len(result.events) == 0

    def test_fallback_to_cpu(self) -> None:
        cuda = _make_device(DeviceType.CUDA)
        cpu = _make_device(DeviceType.CPU)

        call_count = 0

        def load_fn(device: DeviceInfo) -> str:
            nonlocal call_count
            call_count += 1
            if device.device_type == DeviceType.CUDA:
                raise RuntimeError("CUDA driver failed")
            return f"loaded_on_{device.device_string}"

        with patch("uniinfer.engine.fallback.check_device_health") as mock_health:
            mock_health.return_value = DeviceHealthReport(
                device=cuda, status=DeviceStatus.HEALTHY, message="OK"
            )
            loaded, final_device, result = try_with_fallback(
                cuda, [cuda, cpu], load_fn
            )

        assert loaded == "loaded_on_cpu:0"
        assert final_device is cpu
        assert result.fell_back is True
        assert len(result.events) >= 1
        assert call_count == 2

    def test_skips_unhealthy_device(self) -> None:
        cuda = _make_device(DeviceType.CUDA)
        vulkan = _make_device(DeviceType.VULKAN)
        cpu = _make_device(DeviceType.CPU)

        load_calls: list[str] = []

        def load_fn(device: DeviceInfo) -> str:
            load_calls.append(device.device_string)
            return f"loaded_on_{device.device_string}"

        def mock_health(device: DeviceInfo) -> DeviceHealthReport:
            if device.device_type == DeviceType.CUDA:
                return DeviceHealthReport(
                    device=device, status=DeviceStatus.UNAVAILABLE,
                    message="Driver missing", can_allocate=False,
                )
            if device.device_type == DeviceType.VULKAN:
                return DeviceHealthReport(
                    device=device, status=DeviceStatus.HEALTHY, message="OK",
                )
            return DeviceHealthReport(
                device=device, status=DeviceStatus.HEALTHY, message="OK",
            )

        with patch("uniinfer.engine.fallback.check_device_health", side_effect=mock_health):
            loaded, final_device, result = try_with_fallback(
                cuda, [cuda, vulkan, cpu], load_fn
            )

        # CUDA should be skipped (unhealthy), Vulkan tried first
        assert "cuda:0" not in load_calls
        assert loaded == "loaded_on_vulkan:0"
        assert final_device is vulkan

    def test_all_devices_fail(self) -> None:
        cuda = _make_device(DeviceType.CUDA)
        cpu = _make_device(DeviceType.CPU)

        def load_fn(device: DeviceInfo) -> str:
            raise RuntimeError(f"Failed on {device.device_string}")

        with patch("uniinfer.engine.fallback.check_device_health") as mock_health:
            mock_health.return_value = DeviceHealthReport(
                device=cuda, status=DeviceStatus.HEALTHY, message="OK"
            )
            with pytest.raises(RuntimeError, match="Failed to load model on any device"):
                try_with_fallback(cuda, [cuda, cpu], load_fn)

    def test_no_health_check_mode(self) -> None:
        cuda = _make_device(DeviceType.CUDA)
        cpu = _make_device(DeviceType.CPU)

        def load_fn(device: DeviceInfo) -> str:
            if device.device_type == DeviceType.CUDA:
                raise RuntimeError("CUDA failed")
            return "ok"

        # With check_health=False, should not call check_device_health
        with patch("uniinfer.engine.fallback.check_device_health") as mock_health:
            loaded, final_device, result = try_with_fallback(
                cuda, [cuda, cpu], load_fn, check_health=False
            )

        mock_health.assert_not_called()
        assert final_device is cpu
        assert result.fell_back is True


class TestFallbackResult:
    def test_summary_no_fallback(self) -> None:
        device = _make_device(DeviceType.CUDA)
        result = FallbackResult(final_device=device, fell_back=False)
        assert "no fallback" in result.summary.lower()

    def test_summary_with_fallback(self) -> None:
        device = _make_device(DeviceType.CPU)
        result = FallbackResult(
            final_device=device,
            fell_back=True,
            events=[
                FallbackEvent(
                    from_device="cuda:0",
                    to_device="cpu:0",
                    reason="driver failed",
                    success=True,
                )
            ],
        )
        assert "cuda:0" in result.summary
        assert "cpu:0" in result.summary
