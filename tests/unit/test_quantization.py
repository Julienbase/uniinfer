"""Tests for quantization selection logic."""

from __future__ import annotations

import pytest

from uniinfer.hal.interface import DeviceInfo, DeviceType
from uniinfer.models.quantization import (
    get_gguf_search_patterns,
    select_quantization,
)


class TestSelectQuantization:
    """Test automatic quantization selection based on device memory."""

    def test_large_gpu_gets_f16(self, cuda_device_24gb: DeviceInfo) -> None:
        """24GB GPU should get FP16."""
        rec = select_quantization(cuda_device_24gb)
        assert rec.quantization == "f16"
        assert rec.reduce_context is False

    def test_medium_gpu_gets_q8(self) -> None:
        """16GB GPU should get Q8_0."""
        device = DeviceInfo(
            name="RTX 4060 Ti",
            device_type=DeviceType.CUDA,
            device_id=0,
            total_memory=16 * 1024**3,
            free_memory=15 * 1024**3,
        )
        rec = select_quantization(device)
        assert rec.quantization == "q8_0"

    def test_small_gpu_gets_q4(self, cuda_device_8gb: DeviceInfo) -> None:
        """8GB GPU should get Q4_K_M."""
        rec = select_quantization(cuda_device_8gb)
        assert rec.quantization == "q4_k_m"
        assert rec.reduce_context is False

    def test_very_small_gpu_reduces_context(self) -> None:
        """4GB GPU with little free memory should reduce context."""
        device = DeviceInfo(
            name="GTX 1050",
            device_type=DeviceType.CUDA,
            device_id=0,
            total_memory=4 * 1024**3,
            free_memory=3 * 1024**3,
        )
        rec = select_quantization(device)
        assert rec.quantization == "q4_k_m"
        assert rec.reduce_context is True
        assert rec.suggested_context_length == 2048

    def test_cpu_conservative_memory(self, cpu_device: DeviceInfo) -> None:
        """CPU should use conservative memory thresholds (70%)."""
        # 12 GB free * 0.7 = 8.4 GB effective -> q4_k_m
        rec = select_quantization(cpu_device)
        assert rec.quantization == "q4_k_m"

    def test_high_ram_cpu_gets_q8(self) -> None:
        """CPU with lots of RAM should still get decent quantization."""
        device = DeviceInfo(
            name="Server CPU",
            device_type=DeviceType.CPU,
            device_id=0,
            total_memory=128 * 1024**3,
            free_memory=100 * 1024**3,
        )
        rec = select_quantization(device)
        assert rec.quantization == "f16"

    def test_rocm_same_as_cuda(self) -> None:
        """ROCm devices should follow the same GPU thresholds as CUDA."""
        device = DeviceInfo(
            name="RX 7900 XTX",
            device_type=DeviceType.ROCM,
            device_id=0,
            total_memory=24 * 1024**3,
            free_memory=24 * 1024**3,
        )
        rec = select_quantization(device)
        assert rec.quantization == "f16"


class TestGgufSearchPatterns:
    """Test GGUF filename pattern generation."""

    def test_f16_patterns(self) -> None:
        patterns = get_gguf_search_patterns("f16")
        assert any("f16" in p for p in patterns)
        assert any("fp16" in p.lower() for p in patterns)

    def test_q4_k_m_patterns(self) -> None:
        patterns = get_gguf_search_patterns("q4_k_m")
        assert any("Q4_K_M" in p for p in patterns)

    def test_unknown_quant_uses_raw(self) -> None:
        patterns = get_gguf_search_patterns("custom_quant")
        assert any("custom_quant" in p for p in patterns)
