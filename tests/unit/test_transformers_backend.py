"""Tests for the transformers backend with mocked dependencies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from uniinfer.hal.interface import DeviceType


class TestTransformersBackendInit:
    def test_name_property(self) -> None:
        from uniinfer.backends.transformers_backend import TransformersBackend

        backend = TransformersBackend(device_type=DeviceType.CPU)
        assert backend.name == "transformers"

    def test_device_map_cpu(self) -> None:
        from uniinfer.backends.transformers_backend import TransformersBackend

        backend = TransformersBackend(device_type=DeviceType.CPU)
        assert backend._get_device_map() == "cpu"

    def test_device_map_cuda(self) -> None:
        from uniinfer.backends.transformers_backend import TransformersBackend

        backend = TransformersBackend(device_type=DeviceType.CUDA)
        assert backend._get_device_map() == "auto"

    def test_device_map_rocm(self) -> None:
        from uniinfer.backends.transformers_backend import TransformersBackend

        backend = TransformersBackend(device_type=DeviceType.ROCM)
        assert backend._get_device_map() == "auto"


class TestTransformersBackendGenKwargs:
    def test_greedy_sampling(self) -> None:
        from uniinfer.backends.transformers_backend import TransformersBackend

        backend = TransformersBackend(device_type=DeviceType.CPU)
        kwargs = backend._build_gen_kwargs(max_tokens=100, temperature=0.0, top_p=0.9)
        assert kwargs["do_sample"] is False
        assert kwargs["max_new_tokens"] == 100

    def test_sampling_with_temperature(self) -> None:
        from uniinfer.backends.transformers_backend import TransformersBackend

        backend = TransformersBackend(device_type=DeviceType.CPU)
        kwargs = backend._build_gen_kwargs(max_tokens=50, temperature=0.7, top_p=0.95)
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.95


class TestTransformersBackendChatFormat:
    def test_fallback_chat_format(self) -> None:
        from uniinfer.backends.transformers_backend import TransformersBackend

        backend = TransformersBackend(device_type=DeviceType.CPU)

        # Create a mock handle with a tokenizer that doesn't support chat templates
        mock_tokenizer = MagicMock()
        del mock_tokenizer.apply_chat_template  # Remove the attribute
        handle = MagicMock()
        handle.internal = {"tokenizer": mock_tokenizer}

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = backend._format_chat(handle, messages)

        assert "<|system|>" in result
        assert "You are helpful." in result
        assert "<|user|>" in result
        assert "Hello" in result
        assert "<|assistant|>" in result
