"""Backend registry — factory for selecting execution backends."""

from __future__ import annotations

import logging

from uniinfer.backends.interface import ExecutionBackend
from uniinfer.hal.interface import DeviceType

logger = logging.getLogger(__name__)


def get_backend(backend_name: str, device_type: DeviceType) -> ExecutionBackend:
    """Create an execution backend by name.

    Args:
        backend_name: Backend identifier ("llamacpp" or "onnxruntime").
        device_type: Target device type.

    Returns:
        An ExecutionBackend instance.
    """
    if backend_name == "llamacpp":
        from uniinfer.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend(device_type=device_type)

    if backend_name == "onnxruntime":
        from uniinfer.backends.onnxrt import OnnxRuntimeBackend

        return OnnxRuntimeBackend(device_type=device_type)

    raise ValueError(
        f"Unknown backend: '{backend_name}'. Available backends: llamacpp, onnxruntime"
    )


def detect_backend(model_path: str) -> str:
    """Detect the appropriate backend from a model file path.

    Args:
        model_path: Path to the model file.

    Returns:
        Backend name string.
    """
    lower = model_path.lower()

    if lower.endswith(".gguf"):
        return "llamacpp"
    if lower.endswith(".onnx"):
        return "onnxruntime"

    # Default to llamacpp for unknown extensions (most common case)
    logger.warning(
        "Cannot determine backend from file extension '%s', defaulting to llamacpp",
        model_path,
    )
    return "llamacpp"
