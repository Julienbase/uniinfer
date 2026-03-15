"""Backend registry — factory for selecting execution backends."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from uniinfer.backends.interface import ExecutionBackend
from uniinfer.hal.interface import DeviceType

logger = logging.getLogger(__name__)

# GGUF magic: bytes 0-3 = "GGUF"
_GGUF_MAGIC = b"GGUF"


def get_backend(backend_name: str, device_type: DeviceType) -> ExecutionBackend:
    """Create an execution backend by name.

    Args:
        backend_name: Backend identifier ("llamacpp", "onnxruntime", or "transformers").
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

    if backend_name == "transformers":
        from uniinfer.backends.transformers_backend import TransformersBackend

        return TransformersBackend(device_type=device_type)

    raise ValueError(
        f"Unknown backend: '{backend_name}'. "
        f"Available backends: llamacpp, onnxruntime, transformers"
    )


def detect_backend_from_magic(model_path: str) -> Optional[str]:
    """Detect backend by reading the file's magic bytes.

    Args:
        model_path: Path to the model file.

    Returns:
        Backend name string, or None if format is unrecognized.
    """
    try:
        with open(model_path, "rb") as f:
            header = f.read(8)
    except (OSError, IOError):
        return None

    if header[:4] == _GGUF_MAGIC:
        return "llamacpp"

    # ONNX files start with a protobuf header (field 1, varint)
    # but there's no reliable single magic — rely on extension for ONNX

    return None


def _has_safetensors(directory: Path) -> bool:
    """Check if a directory contains SafeTensors model files."""
    return any(directory.glob("*.safetensors"))


def detect_backend(model_path: str) -> str:
    """Detect the appropriate backend from a model file or directory.

    Detection order:
    1. File extension (.gguf, .onnx)
    2. Directory containing .safetensors files → transformers backend
    3. Magic bytes (for files without recognized extensions)
    4. Default to llamacpp

    Args:
        model_path: Path to the model file or directory.

    Returns:
        Backend name string.
    """
    path = Path(model_path)
    lower = model_path.lower()

    # 1. Extension-based detection
    if lower.endswith(".gguf"):
        return "llamacpp"
    if lower.endswith(".onnx"):
        return "onnxruntime"

    # 2. Directory detection
    if path.is_dir():
        # Check for ONNX files first
        if any(path.rglob("*.onnx")):
            logger.info("Detected ONNX model in directory: %s", model_path)
            return "onnxruntime"
        if _has_safetensors(path):
            logger.info("Detected SafeTensors model in directory: %s", model_path)
            return "transformers"
        # Default for directories
        logger.warning(
            "Directory '%s' does not contain recognized model files",
            model_path,
        )
        return "transformers"

    # 3. Magic byte detection
    if path.is_file():
        magic_result = detect_backend_from_magic(model_path)
        if magic_result:
            logger.info("Detected backend '%s' from magic bytes", magic_result)
            return magic_result

    # 4. Default to llamacpp for unknown
    logger.warning(
        "Cannot determine backend for '%s', defaulting to llamacpp",
        model_path,
    )
    return "llamacpp"
