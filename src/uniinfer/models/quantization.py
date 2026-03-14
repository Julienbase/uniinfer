"""Quantization selection logic based on available hardware resources."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from uniinfer.hal.interface import DeviceInfo, DeviceType

logger = logging.getLogger(__name__)

# Thresholds in bytes
_GB = 1024**3
_VRAM_F16_THRESHOLD = 20 * _GB
_VRAM_Q8_0_THRESHOLD = 10 * _GB
_VRAM_Q4_K_M_THRESHOLD = 5 * _GB


@dataclass(frozen=True)
class QuantizationRecommendation:
    """Recommendation for which quantization to use."""

    quantization: str
    reason: str
    reduce_context: bool  # whether to reduce context length for memory savings
    suggested_context_length: int  # 0 means no change


def select_quantization(
    device: DeviceInfo,
    model_size_estimate_gb: float = 0.0,
) -> QuantizationRecommendation:
    """Select the best quantization level based on device memory.

    For GPU devices, uses VRAM thresholds.
    For CPU, uses available system RAM with more conservative thresholds
    (since the OS and other processes need memory too).

    Args:
        device: The target device info.
        model_size_estimate_gb: Estimated model size in GB (0 = unknown, use thresholds only).

    Returns:
        QuantizationRecommendation with the selected quantization and reasoning.
    """
    free_memory = device.free_memory

    if device.device_type == DeviceType.CPU:
        # For CPU, be more conservative — leave headroom for OS
        effective_memory = int(free_memory * 0.7)
    else:
        # For GPU, we can use most of the VRAM
        effective_memory = int(free_memory * 0.85)

    if effective_memory >= _VRAM_F16_THRESHOLD:
        return QuantizationRecommendation(
            quantization="f16",
            reason=f"Sufficient memory ({device.free_memory_gb:.1f} GB free) for full-precision FP16",
            reduce_context=False,
            suggested_context_length=0,
        )

    if effective_memory >= _VRAM_Q8_0_THRESHOLD:
        return QuantizationRecommendation(
            quantization="q8_0",
            reason=f"Good memory ({device.free_memory_gb:.1f} GB free) — using INT8 quantization",
            reduce_context=False,
            suggested_context_length=0,
        )

    if effective_memory >= _VRAM_Q4_K_M_THRESHOLD:
        return QuantizationRecommendation(
            quantization="q4_k_m",
            reason=f"Limited memory ({device.free_memory_gb:.1f} GB free) — using Q4_K_M quantization",
            reduce_context=False,
            suggested_context_length=0,
        )

    # Very low memory — use Q4_K_M with reduced context
    return QuantizationRecommendation(
        quantization="q4_k_m",
        reason=f"Very limited memory ({device.free_memory_gb:.1f} GB free) — using Q4_K_M with reduced context",
        reduce_context=True,
        suggested_context_length=2048,
    )


# Map quantization names to GGUF filename suffixes commonly used on HuggingFace
QUANTIZATION_GGUF_SUFFIXES: dict[str, list[str]] = {
    "f16": ["f16", "fp16", "F16", "FP16"],
    "q8_0": ["Q8_0", "q8_0"],
    "q4_k_m": ["Q4_K_M", "q4_k_m", "Q4_K", "q4_k"],
}


def get_gguf_search_patterns(quantization: str) -> list[str]:
    """Return filename patterns to search for a given quantization level.

    Args:
        quantization: One of 'f16', 'q8_0', 'q4_k_m'.

    Returns:
        List of filename substring patterns to match in GGUF repos.
    """
    suffixes = QUANTIZATION_GGUF_SUFFIXES.get(quantization, [quantization])
    return [f".{s}." for s in suffixes] + [f"-{s}." for s in suffixes]
