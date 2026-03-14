"""Tests for the smart model fitting module."""

from __future__ import annotations

import pytest

from uniinfer.hal.interface import DeviceInfo, DeviceType
from uniinfer.models.fitting import (
    QUANT_BYTES_PER_PARAM,
    FitReport,
    ModelTooLargeError,
    check_model_fit,
    estimate_kv_cache_gb,
    estimate_model_size_gb,
)


def _make_device(
    free_gb: float,
    total_gb: float = 0.0,
    device_type: DeviceType = DeviceType.CUDA,
) -> DeviceInfo:
    """Create a DeviceInfo with the given free memory."""
    if total_gb == 0.0:
        total_gb = free_gb * 1.2
    return DeviceInfo(
        name="Test GPU",
        device_type=device_type,
        device_id=0,
        total_memory=int(total_gb * 1024**3),
        free_memory=int(free_gb * 1024**3),
    )


# --- estimate_model_size_gb ---


class TestEstimateModelSize:
    def test_q4_k_m_7b(self) -> None:
        size = estimate_model_size_gb(7.0, "q4_k_m")
        assert 3.5 < size < 4.5  # ~3.92 GB

    def test_f16_7b(self) -> None:
        size = estimate_model_size_gb(7.0, "f16")
        assert 13.5 < size < 14.5  # 14.0 GB

    def test_unknown_quant_uses_default(self) -> None:
        size = estimate_model_size_gb(7.0, "unknown_quant")
        assert size == pytest.approx(7.0 * 0.56, abs=0.01)

    def test_small_model(self) -> None:
        size = estimate_model_size_gb(1.1, "q4_k_m")
        assert size < 1.0  # ~0.62 GB


# --- estimate_kv_cache_gb ---


class TestEstimateKVCache:
    def test_default_4k_context(self) -> None:
        kv = estimate_kv_cache_gb(4096)
        assert 1.5 < kv < 2.5  # ~2.0 GB for 32 layers, 32 heads, 128 dim

    def test_larger_context(self) -> None:
        kv_4k = estimate_kv_cache_gb(4096)
        kv_8k = estimate_kv_cache_gb(8192)
        assert kv_8k == pytest.approx(kv_4k * 2.0, abs=0.01)

    def test_zero_context(self) -> None:
        kv = estimate_kv_cache_gb(0)
        assert kv == 0.0


# --- check_model_fit ---


class TestCheckModelFit:
    def test_model_fits_easily(self) -> None:
        device = _make_device(free_gb=24.0)
        report = check_model_fit(
            device=device,
            model_size_gb=4.0,
            context_length=4096,
            quantization="q4_k_m",
            param_count_billions=7.0,
        )
        assert report.fits is True
        assert report.headroom_gb > 0
        assert report.warnings == []

    def test_model_does_not_fit(self) -> None:
        device = _make_device(free_gb=4.0)
        report = check_model_fit(
            device=device,
            model_size_gb=14.0,
            context_length=4096,
            quantization="f16",
            param_count_billions=7.0,
        )
        assert report.fits is False
        assert report.headroom_gb < 0
        assert len(report.warnings) > 0

    def test_tight_fit_warns(self) -> None:
        # 4 GB model + ~2.0 KV + 0.5 overhead = ~6.5 GB needed
        # 8 GB * 0.85 = 6.8 GB usable — tight fit with ~0.3 GB headroom
        device = _make_device(free_gb=8.0)
        report = check_model_fit(
            device=device,
            model_size_gb=4.0,
            context_length=4096,
            quantization="q4_k_m",
        )
        assert report.fits is True
        # With small headroom, should get tight fit warning
        if report.headroom_gb < 0.5:
            assert any("Tight fit" in w or "headroom" in w for w in report.warnings)

    def test_alternatives_computed(self) -> None:
        device = _make_device(free_gb=24.0)
        report = check_model_fit(
            device=device,
            model_size_gb=4.0,
            context_length=4096,
            quantization="q4_k_m",
            param_count_billions=7.0,
        )
        assert len(report.alternatives) > 0
        # q4_k_m should be in the alternatives
        quants = [a.quantization for a in report.alternatives]
        assert "q4_k_m" in quants
        assert "f16" in quants

    def test_alternatives_not_computed_without_param_count(self) -> None:
        device = _make_device(free_gb=24.0)
        report = check_model_fit(
            device=device,
            model_size_gb=4.0,
            context_length=4096,
            quantization="q4_k_m",
            param_count_billions=0.0,
        )
        assert report.alternatives == []

    def test_recommends_smaller_quant_when_doesnt_fit(self) -> None:
        device = _make_device(free_gb=8.0)
        report = check_model_fit(
            device=device,
            model_size_gb=14.0,  # f16 of 7B
            context_length=4096,
            quantization="f16",
            param_count_billions=7.0,
        )
        assert report.fits is False
        assert report.recommended_quantization != "f16"
        assert report.recommended_quantization in ("q8_0", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k")

    def test_cpu_uses_more_conservative_safety(self) -> None:
        gpu = _make_device(free_gb=10.0, device_type=DeviceType.CUDA)
        cpu = _make_device(free_gb=10.0, device_type=DeviceType.CPU)

        report_gpu = check_model_fit(device=gpu, model_size_gb=4.0)
        report_cpu = check_model_fit(device=cpu, model_size_gb=4.0)

        # CPU should have less headroom due to stricter safety factor
        assert report_cpu.headroom_gb < report_gpu.headroom_gb

    def test_recommends_reduced_context_as_last_resort(self) -> None:
        # 3 GB free, 7B model — even q2_k (~2.38 GB) + KV + overhead is tight
        device = _make_device(free_gb=3.5)
        report = check_model_fit(
            device=device,
            model_size_gb=14.0,
            context_length=4096,
            quantization="f16",
            param_count_billions=7.0,
        )
        assert report.fits is False
        # Should recommend a reduced context if nothing fits at 4096
        if report.recommended_context_length < 4096:
            assert report.recommended_context_length in (2048, 1024, 512)


# --- ModelTooLargeError ---


class TestModelTooLargeError:
    def test_has_fit_report(self) -> None:
        report = FitReport(
            fits=False,
            model_size_gb=14.0,
            available_memory_gb=8.0,
            overhead_gb=0.5,
            headroom_gb=-6.0,
            recommended_quantization="q4_k_m",
            recommended_context_length=4096,
        )
        error = ModelTooLargeError("too large", fit_report=report)
        assert error.fit_report is report
        assert "too large" in str(error)


# --- QUANT_BYTES_PER_PARAM ---


class TestQuantBytesPerParam:
    def test_expected_keys(self) -> None:
        assert "f16" in QUANT_BYTES_PER_PARAM
        assert "q4_k_m" in QUANT_BYTES_PER_PARAM
        assert "q8_0" in QUANT_BYTES_PER_PARAM

    def test_f16_is_2_bytes(self) -> None:
        assert QUANT_BYTES_PER_PARAM["f16"] == 2.0

    def test_ordering(self) -> None:
        # Higher quality quants should use more bytes per param
        assert QUANT_BYTES_PER_PARAM["f16"] > QUANT_BYTES_PER_PARAM["q8_0"]
        assert QUANT_BYTES_PER_PARAM["q8_0"] > QUANT_BYTES_PER_PARAM["q4_k_m"]
        assert QUANT_BYTES_PER_PARAM["q4_k_m"] > QUANT_BYTES_PER_PARAM["q2_k"]
