"""GGUF file metadata parser — reads model info from GGUF binary headers."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# GGUF format constants
_GGUF_MAGIC = b"GGUF"
_GGUF_VALUE_TYPE_STRING = 8
_GGUF_VALUE_TYPE_UINT32 = 4
_GGUF_VALUE_TYPE_INT32 = 5
_GGUF_VALUE_TYPE_UINT64 = 10

# Map of file_type ID to quantization name (from llama.cpp ggml_ftype enum)
_FILE_TYPE_NAMES: dict[int, str] = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    7: "q8_0",
    8: "q5_0",
    9: "q5_1",
    10: "q2_k",
    11: "q3_k_s",
    12: "q3_k_m",
    13: "q3_k_l",
    14: "q4_k_s",
    15: "q4_k_m",
    16: "q5_k_s",
    17: "q5_k_m",
    18: "q6_k",
}


@dataclass(frozen=True)
class GGUFMetadata:
    """Parsed metadata from a GGUF file header."""

    architecture: str = ""
    model_name: str = ""
    file_type: int = 0
    quantization_name: str = ""
    context_length: int = 0
    tensor_count: int = 0
    file_size_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return self.file_size_bytes / (1024**3)


def _read_string(f: Any) -> str:
    """Read a GGUF string (uint64 length prefix + bytes)."""
    length = struct.unpack("<Q", f.read(8))[0]
    if length > 1024 * 1024:  # Safety: skip strings > 1MB
        f.seek(length, 1)
        return ""
    return f.read(length).decode("utf-8", errors="replace")


def _read_value(f: Any, value_type: int) -> Any:
    """Read a GGUF metadata value of the given type."""
    if value_type == 0:  # uint8
        return struct.unpack("<B", f.read(1))[0]
    elif value_type == 1:  # int8
        return struct.unpack("<b", f.read(1))[0]
    elif value_type == 2:  # uint16
        return struct.unpack("<H", f.read(2))[0]
    elif value_type == 3:  # int16
        return struct.unpack("<h", f.read(2))[0]
    elif value_type == _GGUF_VALUE_TYPE_UINT32:  # uint32
        return struct.unpack("<I", f.read(4))[0]
    elif value_type == _GGUF_VALUE_TYPE_INT32:  # int32
        return struct.unpack("<i", f.read(4))[0]
    elif value_type == 6:  # float32
        return struct.unpack("<f", f.read(4))[0]
    elif value_type == 7:  # bool
        return struct.unpack("<?", f.read(1))[0]
    elif value_type == _GGUF_VALUE_TYPE_STRING:  # string
        return _read_string(f)
    elif value_type == 9:  # array
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        return [_read_value(f, arr_type) for _ in range(min(arr_len, 1024))]
    elif value_type == _GGUF_VALUE_TYPE_UINT64:  # uint64
        return struct.unpack("<Q", f.read(8))[0]
    elif value_type == 11:  # int64
        return struct.unpack("<q", f.read(8))[0]
    elif value_type == 12:  # float64
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unknown GGUF value type: {value_type}")


def parse_gguf_metadata(path: Path) -> GGUFMetadata:
    """Parse metadata from a GGUF file header.

    Reads only the header (metadata key-value pairs), not tensor data.

    Args:
        path: Path to the GGUF file.

    Returns:
        GGUFMetadata with parsed fields.

    Raises:
        ValueError: If the file is not a valid GGUF file.
    """
    file_size = path.stat().st_size

    with open(path, "rb") as f:
        # Magic
        magic = f.read(4)
        if magic != _GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: invalid magic bytes {magic!r}")

        # Version
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")

        # Counts
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

        # Parse metadata key-value pairs
        metadata: dict[str, Any] = {}
        keys_to_extract = {
            "general.architecture",
            "general.name",
            "general.file_type",
        }

        for _ in range(metadata_kv_count):
            try:
                key = _read_string(f)
                value_type = struct.unpack("<I", f.read(4))[0]
                value = _read_value(f, value_type)
                metadata[key] = value
            except (struct.error, UnicodeDecodeError, ValueError):
                break  # Stop parsing on error, we have what we need

        # Extract fields
        architecture = str(metadata.get("general.architecture", ""))
        model_name = str(metadata.get("general.name", ""))
        file_type = int(metadata.get("general.file_type", 0))
        quant_name = _FILE_TYPE_NAMES.get(file_type, f"unknown({file_type})")

        # Try to get context length from architecture-specific key
        ctx_key = f"{architecture}.context_length" if architecture else ""
        context_length = int(metadata.get(ctx_key, 0))

    return GGUFMetadata(
        architecture=architecture,
        model_name=model_name,
        file_type=file_type,
        quantization_name=quant_name,
        context_length=context_length,
        tensor_count=tensor_count,
        file_size_bytes=file_size,
        extra={k: str(v) for k, v in metadata.items() if k in keys_to_extract},
    )


def estimate_param_count_from_name(model_id: str) -> Optional[float]:
    """Extract parameter count from model name heuristics.

    Looks for patterns like "7B", "13B", "70B", "1.1B", "3.8B" in the model ID.

    Args:
        model_id: HuggingFace model ID or alias.

    Returns:
        Estimated parameter count in billions, or None if not detected.
    """
    import re

    # Match patterns like "7B", "70B", "1.1B", "3.8B", "0.5B"
    match = re.search(r"(\d+\.?\d*)\s*[Bb]", model_id)
    if match:
        return float(match.group(1))
    return None
