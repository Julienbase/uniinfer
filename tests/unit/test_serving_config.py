"""Tests for serving configuration."""

from __future__ import annotations

import pytest

from uniinfer.config.serving_config import ServingConfig


class TestServingConfig:
    def test_defaults(self) -> None:
        config = ServingConfig(model="test/model")
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.device == "auto"
        assert config.quantization == "auto"
        assert config.context_length == 4096
        assert config.max_concurrent_requests == 64
        assert config.timeout_seconds == 300.0
        assert config.api_key is None

    def test_custom_values(self) -> None:
        config = ServingConfig(
            model="test/model",
            host="127.0.0.1",
            port=9000,
            device="cuda:0",
            quantization="q4_k_m",
            api_key="sk-test-123",
            max_concurrent_requests=128,
        )
        assert config.port == 9000
        assert config.api_key == "sk-test-123"

    def test_invalid_port_too_high(self) -> None:
        with pytest.raises(Exception):
            ServingConfig(model="test", port=99999)

    def test_invalid_port_zero(self) -> None:
        with pytest.raises(Exception):
            ServingConfig(model="test", port=0)

    def test_invalid_device(self) -> None:
        with pytest.raises(Exception):
            ServingConfig(model="test", device="tpu:0")

    def test_invalid_quantization(self) -> None:
        with pytest.raises(Exception):
            ServingConfig(model="test", quantization="q3_k_s")

    def test_valid_device_formats(self) -> None:
        for device in ["auto", "cuda:0", "rocm:0", "vulkan:0", "cpu"]:
            config = ServingConfig(model="test", device=device)
            assert config.device == device

    def test_model_required(self) -> None:
        with pytest.raises(Exception):
            ServingConfig()  # type: ignore[call-arg]
