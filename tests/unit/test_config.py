"""Tests for engine configuration validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from uniinfer.config.engine_config import EngineConfig


class TestEngineConfig:
    """Test suite for EngineConfig Pydantic model."""

    def test_minimal_config(self) -> None:
        """Model ID is the only required field."""
        config = EngineConfig(model="meta-llama/Llama-3.1-8B-Instruct")
        assert config.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert config.device == "auto"
        assert config.quantization == "auto"
        assert config.max_tokens == 2048
        assert config.context_length == 4096

    def test_default_cache_dir(self) -> None:
        """Cache dir defaults to ~/.uniinfer/cache."""
        config = EngineConfig(model="test/model")
        expected = str(Path(os.path.expanduser("~")) / ".uniinfer" / "cache")
        assert config.cache_dir == expected

    def test_custom_cache_dir(self) -> None:
        config = EngineConfig(model="test/model", cache_dir="/tmp/my_cache")
        assert config.cache_dir == "/tmp/my_cache"

    def test_valid_devices(self) -> None:
        """All supported device strings should pass validation."""
        for device in ["auto", "cuda:0", "cuda:1", "rocm:0", "vulkan:0", "cpu"]:
            config = EngineConfig(model="test/model", device=device)
            assert config.device == device

    def test_invalid_device_type(self) -> None:
        with pytest.raises(ValueError, match="Device type"):
            EngineConfig(model="test/model", device="tpu:0")

    def test_invalid_device_id(self) -> None:
        with pytest.raises(ValueError, match="Device ID must be an integer"):
            EngineConfig(model="test/model", device="cuda:abc")

    def test_valid_quantizations(self) -> None:
        for quant in ["auto", "f16", "q8_0", "q4_k_m"]:
            config = EngineConfig(model="test/model", quantization=quant)
            assert config.quantization == quant

    def test_invalid_quantization(self) -> None:
        with pytest.raises(ValueError, match="Quantization must be one of"):
            EngineConfig(model="test/model", quantization="q2_k")

    def test_max_tokens_bounds(self) -> None:
        config = EngineConfig(model="test/model", max_tokens=1)
        assert config.max_tokens == 1

        with pytest.raises(ValueError):
            EngineConfig(model="test/model", max_tokens=0)

        with pytest.raises(ValueError):
            EngineConfig(model="test/model", max_tokens=100000)

    def test_context_length_bounds(self) -> None:
        config = EngineConfig(model="test/model", context_length=256)
        assert config.context_length == 256

        with pytest.raises(ValueError):
            EngineConfig(model="test/model", context_length=100)

    def test_is_local_model_false(self) -> None:
        config = EngineConfig(model="meta-llama/Llama-3.1-8B-Instruct")
        assert config.is_local_model is False

    def test_cache_path_property(self) -> None:
        config = EngineConfig(model="test/model", cache_dir="/tmp/test_cache")
        assert config.cache_path == Path("/tmp/test_cache")

    def test_full_config(self) -> None:
        config = EngineConfig(
            model="test/model",
            device="cuda:0",
            quantization="q4_k_m",
            max_tokens=1024,
            context_length=8192,
            cache_dir="/tmp/cache",
            n_gpu_layers=32,
            n_threads=8,
        )
        assert config.n_gpu_layers == 32
        assert config.n_threads == 8
