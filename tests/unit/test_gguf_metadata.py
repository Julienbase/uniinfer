"""Tests for the GGUF metadata parser."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest

from uniinfer.models.gguf_metadata import (
    GGUFMetadata,
    estimate_param_count_from_name,
    parse_gguf_metadata,
)


def _create_minimal_gguf(
    architecture: str = "llama",
    model_name: str = "TestModel-7B",
    file_type: int = 15,  # q4_k_m
    context_length: int = 4096,
) -> bytes:
    """Create a minimal valid GGUF v3 binary for testing."""
    buf = bytearray()

    # Magic
    buf.extend(b"GGUF")
    # Version 3
    buf.extend(struct.pack("<I", 3))
    # Tensor count
    buf.extend(struct.pack("<Q", 42))

    # Build metadata KV pairs
    kvs: list[tuple[str, int, bytes]] = []

    def _pack_string(s: str) -> bytes:
        encoded = s.encode("utf-8")
        return struct.pack("<Q", len(encoded)) + encoded

    # general.architecture (string, type=8)
    key_data = _pack_string("general.architecture")
    val_data = struct.pack("<I", 8) + _pack_string(architecture)
    kvs.append((key_data, 8, val_data))

    # general.name (string, type=8)
    key_data = _pack_string("general.name")
    val_data = struct.pack("<I", 8) + _pack_string(model_name)
    kvs.append((key_data, 8, val_data))

    # general.file_type (uint32, type=4)
    key_data = _pack_string("general.file_type")
    val_data = struct.pack("<I", 4) + struct.pack("<I", file_type)
    kvs.append((key_data, 4, val_data))

    # {arch}.context_length (uint32, type=4)
    key_data = _pack_string(f"{architecture}.context_length")
    val_data = struct.pack("<I", 4) + struct.pack("<I", context_length)
    kvs.append((key_data, 4, val_data))

    # Metadata KV count
    buf.extend(struct.pack("<Q", len(kvs)))

    # Write KV pairs
    for key_bytes, _, val_bytes in kvs:
        buf.extend(key_bytes)
        buf.extend(val_bytes)

    return bytes(buf)


class TestParseGGUFMetadata:
    def test_valid_gguf(self, tmp_path: Path) -> None:
        data = _create_minimal_gguf()
        gguf_file = tmp_path / "test.gguf"
        gguf_file.write_bytes(data)

        meta = parse_gguf_metadata(gguf_file)

        assert meta.architecture == "llama"
        assert meta.model_name == "TestModel-7B"
        assert meta.file_type == 15
        assert meta.quantization_name == "q4_k_m"
        assert meta.context_length == 4096
        assert meta.tensor_count == 42
        assert meta.file_size_bytes == len(data)

    def test_size_gb_property(self, tmp_path: Path) -> None:
        data = _create_minimal_gguf()
        gguf_file = tmp_path / "test.gguf"
        gguf_file.write_bytes(data)

        meta = parse_gguf_metadata(gguf_file)
        expected_gb = len(data) / (1024**3)
        assert meta.size_gb == pytest.approx(expected_gb, abs=0.001)

    def test_invalid_magic(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.gguf"
        bad_file.write_bytes(b"NOTAGGUF" + b"\x00" * 100)

        with pytest.raises(ValueError, match="Not a GGUF file"):
            parse_gguf_metadata(bad_file)

    def test_unsupported_version(self, tmp_path: Path) -> None:
        buf = bytearray()
        buf.extend(b"GGUF")
        buf.extend(struct.pack("<I", 99))  # version 99
        buf.extend(b"\x00" * 100)

        bad_file = tmp_path / "badver.gguf"
        bad_file.write_bytes(bytes(buf))

        with pytest.raises(ValueError, match="Unsupported GGUF version"):
            parse_gguf_metadata(bad_file)

    def test_f16_file_type(self, tmp_path: Path) -> None:
        data = _create_minimal_gguf(file_type=1)
        gguf_file = tmp_path / "f16.gguf"
        gguf_file.write_bytes(data)

        meta = parse_gguf_metadata(gguf_file)
        assert meta.quantization_name == "f16"

    def test_q8_0_file_type(self, tmp_path: Path) -> None:
        data = _create_minimal_gguf(file_type=7)
        gguf_file = tmp_path / "q8.gguf"
        gguf_file.write_bytes(data)

        meta = parse_gguf_metadata(gguf_file)
        assert meta.quantization_name == "q8_0"

    def test_custom_context_length(self, tmp_path: Path) -> None:
        data = _create_minimal_gguf(context_length=8192)
        gguf_file = tmp_path / "8k.gguf"
        gguf_file.write_bytes(data)

        meta = parse_gguf_metadata(gguf_file)
        assert meta.context_length == 8192


class TestEstimateParamCountFromName:
    def test_7b(self) -> None:
        assert estimate_param_count_from_name("Llama-7B-Chat-GGUF") == 7.0

    def test_70b(self) -> None:
        # Regex picks up the first number+B pattern — "3.1" is not followed by B,
        # so it correctly finds "70B"
        assert estimate_param_count_from_name("Meta-Llama-3.1-70B") == 70.0

    def test_1_1b(self) -> None:
        assert estimate_param_count_from_name("TinyLlama-1.1B-Chat") == 1.1

    def test_no_match(self) -> None:
        assert estimate_param_count_from_name("some-model-no-params") is None

    def test_lowercase_b(self) -> None:
        assert estimate_param_count_from_name("model-7b-instruct") == 7.0

    def test_decimal_params(self) -> None:
        assert estimate_param_count_from_name("phi-3.8B-mini") == 3.8
