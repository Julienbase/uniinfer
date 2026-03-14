"""Smart model fitting — VRAM budget calculation and recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from uniinfer.hal.interface import DeviceInfo, DeviceType

logger = logging.getLogger(__name__)

# Bytes-per-parameter for common quantization levels
QUANT_BYTES_PER_PARAM: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "q8_0": 1.0,
    "q5_k_m": 0.69,
    "q5_k_s": 0.69,
    "q4_k_m": 0.56,
    "q4_k_s": 0.53,
    "q4_0": 0.50,
    "q3_k_m": 0.44,
    "q3_k_l": 0.47,
    "q3_k_s": 0.41,
    "q2_k": 0.34,
}

# Overhead constants (GB)
_GPU_OVERHEAD_GB = 0.5
_CPU_OVERHEAD_GB = 1.0

# Safety factor — don't use more than this fraction of available memory
_GPU_SAFETY_FACTOR = 0.85
_CPU_SAFETY_FACTOR = 0.70


@dataclass(frozen=True)
class FitAlternative:
    """An alternative quantization that may fit."""

    quantization: str
    estimated_size_gb: float
    fits: bool
    context_length: int


@dataclass(frozen=True)
class FitReport:
    """Result of a model fitting check."""

    fits: bool
    model_size_gb: float
    available_memory_gb: float
    overhead_gb: float
    headroom_gb: float
    recommended_quantization: str
    recommended_context_length: int
    alternatives: list[FitAlternative] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def estimate_model_size_gb(
    param_count_billions: float,
    quantization: str,
) -> float:
    """Estimate model size in GB for a given param count and quantization.

    Args:
        param_count_billions: Number of parameters in billions.
        quantization: Quantization level (e.g., "q4_k_m", "f16").

    Returns:
        Estimated size in GB.
    """
    bytes_per_param = QUANT_BYTES_PER_PARAM.get(quantization, 0.56)
    return param_count_billions * bytes_per_param


def estimate_kv_cache_gb(
    context_length: int,
    n_layers: int = 32,
    n_heads: int = 32,
    head_dim: int = 128,
) -> float:
    """Estimate KV cache size in GB.

    The KV cache stores key and value tensors for each layer.
    Size = 2 (K+V) * n_layers * context_length * n_heads * head_dim * 2 bytes (fp16)

    Args:
        context_length: Context window size in tokens.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads (for KV, often fewer with GQA).
        head_dim: Dimension per attention head.

    Returns:
        Estimated KV cache size in GB.
    """
    # 2 for K+V, 2 bytes for fp16
    size_bytes = 2 * n_layers * context_length * n_heads * head_dim * 2
    return size_bytes / (1024**3)


def _get_overhead_gb(device_type: DeviceType) -> float:
    """Return runtime overhead estimate based on device type."""
    if device_type == DeviceType.CPU:
        return _CPU_OVERHEAD_GB
    return _GPU_OVERHEAD_GB


def _get_safety_factor(device_type: DeviceType) -> float:
    """Return the safety factor for the given device type."""
    if device_type == DeviceType.CPU:
        return _CPU_SAFETY_FACTOR
    return _GPU_SAFETY_FACTOR


def _usable_memory_gb(device: DeviceInfo) -> float:
    """Return usable memory after applying the safety factor."""
    factor = _get_safety_factor(device.device_type)
    return device.free_memory_gb * factor


def check_model_fit(
    device: DeviceInfo,
    model_size_gb: float,
    context_length: int = 4096,
    quantization: str = "q4_k_m",
    param_count_billions: float = 0.0,
) -> FitReport:
    """Check whether a model fits in the device's available memory.

    Budget formula:
        model_size + kv_cache + overhead <= available_memory * safety_factor

    Args:
        device: Target device info with memory details.
        model_size_gb: Estimated model size in GB (from file size or estimation).
        context_length: Desired context window size.
        quantization: Current quantization level.
        param_count_billions: Parameter count (for computing alternatives). 0 = unknown.

    Returns:
        FitReport with fitting result, headroom, and alternatives.
    """
    overhead_gb = _get_overhead_gb(device.device_type)
    usable_gb = _usable_memory_gb(device)

    # Estimate KV cache (use reasonable defaults for layer/head counts)
    kv_cache_gb = estimate_kv_cache_gb(context_length)

    total_required = model_size_gb + kv_cache_gb + overhead_gb
    headroom_gb = usable_gb - total_required
    fits = headroom_gb >= 0

    warnings: list[str] = []
    if not fits:
        warnings.append(
            f"Model requires ~{total_required:.1f} GB but only "
            f"~{usable_gb:.1f} GB usable ({device.free_memory_gb:.1f} GB free × "
            f"{_get_safety_factor(device.device_type):.0%} safety)."
        )
    elif headroom_gb < 0.5:
        warnings.append(
            f"Tight fit: only {headroom_gb:.2f} GB headroom. "
            "Performance may degrade under load."
        )

    # Compute alternatives if we know param count
    alternatives: list[FitAlternative] = []
    recommended_quant = quantization
    recommended_ctx = context_length

    if param_count_billions > 0:
        # Try all quantization levels from highest to lowest quality
        quant_order = ["f16", "q8_0", "q5_k_m", "q4_k_m", "q3_k_m", "q2_k"]
        for q in quant_order:
            alt_size = estimate_model_size_gb(param_count_billions, q)
            alt_total = alt_size + kv_cache_gb + overhead_gb
            alt_fits = alt_total <= usable_gb
            alternatives.append(FitAlternative(
                quantization=q,
                estimated_size_gb=round(alt_size, 2),
                fits=alt_fits,
                context_length=context_length,
            ))

        # If current doesn't fit, find the best alternative that does
        if not fits:
            for alt in alternatives:
                if alt.fits:
                    recommended_quant = alt.quantization
                    break
            else:
                # Nothing fits at full context — try reducing context
                for q in quant_order:
                    alt_size = estimate_model_size_gb(param_count_billions, q)
                    for ctx in [2048, 1024, 512]:
                        alt_kv = estimate_kv_cache_gb(ctx)
                        alt_total = alt_size + alt_kv + overhead_gb
                        if alt_total <= usable_gb:
                            recommended_quant = q
                            recommended_ctx = ctx
                            warnings.append(
                                f"Recommend reducing context to {ctx} "
                                f"with {q} quantization to fit."
                            )
                            break
                    else:
                        continue
                    break
                else:
                    warnings.append(
                        "Model cannot fit on this device at any quantization level. "
                        "Consider using a smaller model."
                    )

    return FitReport(
        fits=fits,
        model_size_gb=round(model_size_gb, 2),
        available_memory_gb=round(device.free_memory_gb, 2),
        overhead_gb=round(overhead_gb, 2),
        headroom_gb=round(headroom_gb, 2),
        recommended_quantization=recommended_quant,
        recommended_context_length=recommended_ctx,
        alternatives=alternatives,
        warnings=warnings,
    )


class ModelTooLargeError(RuntimeError):
    """Raised when a model is too large for the target device."""

    def __init__(self, message: str, fit_report: FitReport) -> None:
        super().__init__(message)
        self.fit_report = fit_report
