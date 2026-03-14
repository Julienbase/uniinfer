"""Tests for backend registry."""

from __future__ import annotations

import pytest

from uniinfer.backends.registry import detect_backend


class TestDetectBackend:
    def test_gguf_detects_llamacpp(self) -> None:
        assert detect_backend("model.gguf") == "llamacpp"
        assert detect_backend("/path/to/model.GGUF") == "llamacpp"

    def test_onnx_detects_onnxruntime(self) -> None:
        assert detect_backend("model.onnx") == "onnxruntime"
        assert detect_backend("/path/to/model.ONNX") == "onnxruntime"

    def test_unknown_extension_defaults_to_llamacpp(self) -> None:
        assert detect_backend("model.bin") == "llamacpp"
        assert detect_backend("model.safetensors") == "llamacpp"

    def test_no_extension_defaults_to_llamacpp(self) -> None:
        assert detect_backend("model") == "llamacpp"
