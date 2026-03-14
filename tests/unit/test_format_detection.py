"""Tests for multi-format model detection in backend registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from uniinfer.backends.registry import (
    detect_backend,
    detect_backend_from_magic,
)


class TestDetectBackendByExtension:
    def test_gguf_extension(self) -> None:
        assert detect_backend("model.gguf") == "llamacpp"

    def test_gguf_extension_case_insensitive(self) -> None:
        assert detect_backend("Model.GGUF") == "llamacpp"

    def test_onnx_extension(self) -> None:
        assert detect_backend("model.onnx") == "onnxruntime"

    def test_onnx_extension_uppercase(self) -> None:
        assert detect_backend("model.ONNX") == "onnxruntime"


class TestDetectBackendFromMagic:
    def test_gguf_magic_bytes(self, tmp_path: Path) -> None:
        gguf_file = tmp_path / "model.bin"
        # GGUF magic: "GGUF" followed by version bytes
        gguf_file.write_bytes(b"GGUF\x03\x00\x00\x00")
        assert detect_backend_from_magic(str(gguf_file)) == "llamacpp"

    def test_unknown_magic_returns_none(self, tmp_path: Path) -> None:
        unknown_file = tmp_path / "model.bin"
        unknown_file.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")
        assert detect_backend_from_magic(str(unknown_file)) is None

    def test_nonexistent_file_returns_none(self) -> None:
        assert detect_backend_from_magic("/nonexistent/path.bin") is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.bin"
        empty_file.write_bytes(b"")
        assert detect_backend_from_magic(str(empty_file)) is None


class TestDetectBackendDirectory:
    def test_directory_with_safetensors(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"fake")
        assert detect_backend(str(tmp_path)) == "transformers"

    def test_directory_without_model_files(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("hello")
        assert detect_backend(str(tmp_path)) == "transformers"


class TestDetectBackendFallback:
    def test_unknown_extension_with_gguf_magic(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.bin"
        model_file.write_bytes(b"GGUF\x03\x00\x00\x00")
        assert detect_backend(str(model_file)) == "llamacpp"

    def test_completely_unknown_defaults_to_llamacpp(self) -> None:
        # Non-existent path with no extension
        assert detect_backend("/fake/path/model") == "llamacpp"
