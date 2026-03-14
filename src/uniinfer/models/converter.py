"""Model conversion utilities.

For v0.1, UniInfer relies on pre-converted GGUF files from HuggingFace.
Full conversion (safetensors -> GGUF) is planned for a future release.
"""

from __future__ import annotations

import logging
from pathlib import Path

from uniinfer.hal.interface import DeviceInfo
from uniinfer.models.quantization import select_quantization

logger = logging.getLogger(__name__)


def convert_to_gguf(
    model_path: Path,
    output_path: Path,
    quantization: str = "q4_k_m",
) -> Path:
    """Convert a HuggingFace model to GGUF format.

    Note: In v0.1, direct conversion is not supported. This function raises
    a clear error directing users to use pre-converted GGUF files.

    Args:
        model_path: Path to the HuggingFace model directory.
        output_path: Desired output path for the GGUF file.
        quantization: Target quantization level.

    Returns:
        Path to the converted GGUF file.

    Raises:
        NotImplementedError: Always, in v0.1.
    """
    raise NotImplementedError(
        f"Direct model conversion is not yet supported in UniInfer v0.1.\n\n"
        f"The model at '{model_path}' needs to be converted to GGUF format.\n"
        f"Options:\n"
        f"  1. Search HuggingFace for a pre-converted GGUF version of this model\n"
        f"  2. Use llama.cpp's convert_hf_to_gguf.py script manually:\n"
        f"     python convert_hf_to_gguf.py {model_path} --outfile {output_path}\n"
        f"  3. Use a model ID that already has GGUF files in the repo\n"
    )


def select_quantization_for_device(
    device: DeviceInfo,
    requested: str = "auto",
) -> str:
    """Select the appropriate quantization level for a device.

    Args:
        device: Target device info.
        requested: User-requested quantization. "auto" for automatic selection.

    Returns:
        Quantization string (f16, q8_0, or q4_k_m).
    """
    if requested != "auto":
        logger.info("Using user-specified quantization: %s", requested)
        return requested

    recommendation = select_quantization(device)
    logger.info(
        "Auto-selected quantization: %s (%s)",
        recommendation.quantization,
        recommendation.reason,
    )
    return recommendation.quantization
