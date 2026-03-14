"""Integration tests for the REST API using FastAPI TestClient.

All tests mock the Engine to avoid loading real models.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from uniinfer.api.server import UniInferServer
from uniinfer.backends.interface import GenerationResult, StreamChunk
from uniinfer.config.serving_config import ServingConfig


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock Engine."""
    engine = MagicMock()
    engine.info.return_value = {
        "model": "test-model",
        "device": "cpu",
        "device_name": "Test CPU",
        "quantization": "q4_k_m",
        "context_length": 4096,
        "backend": "llama.cpp",
        "model_path": "/fake/model.gguf",
        "loaded": True,
    }
    engine.generate.return_value = GenerationResult(
        text="Generated text",
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
    )
    engine.chat.return_value = GenerationResult(
        text="Chat response",
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
    )

    def mock_stream(prompt, **kwargs):  # type: ignore[no-untyped-def]
        yield StreamChunk(text="Hello", finished=False)
        yield StreamChunk(text=" world", finished=True)

    def mock_chat_stream(messages, **kwargs):  # type: ignore[no-untyped-def]
        yield StreamChunk(text="Chat", finished=False)
        yield StreamChunk(text=" reply", finished=True)

    engine.stream.side_effect = mock_stream
    engine.chat_stream.side_effect = mock_chat_stream
    return engine


@pytest.fixture
def client(mock_engine: MagicMock) -> TestClient:
    """Create a TestClient with a mocked Engine."""
    config = ServingConfig(model="test-model")

    with patch("uniinfer.api.server.Engine", return_value=mock_engine):
        server = UniInferServer(config)
        server.engine = mock_engine

        # Create scheduler with mock engine
        with TestClient(server.app) as tc:
            yield tc


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"] == "test-model"


class TestModelsEndpoint:
    def test_list_models(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["object"] == "model"


class TestChatCompletions:
    def test_chat_completion(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert "created" in data
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] >= 0
        assert data["usage"]["total_tokens"] >= 0

    def test_chat_streaming(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        lines = response.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 1

        # Last data line should be [DONE]
        assert data_lines[-1] == "data: [DONE]"

        # Parse a non-DONE chunk
        for line in data_lines:
            if line == "data: [DONE]":
                continue
            chunk_data = json.loads(line[len("data: "):])
            assert chunk_data["object"] == "chat.completion.chunk"
            assert "choices" in chunk_data

    def test_invalid_request_empty_messages(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [],
            },
        )
        assert response.status_code == 422


class TestCompletions:
    def test_text_completion(self, client: TestClient) -> None:
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello world",
                "max_tokens": 50,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "text_completion"
        assert data["id"].startswith("cmpl-")
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_completion_streaming(self, client: TestClient) -> None:
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = response.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert data_lines[-1] == "data: [DONE]"

    def test_missing_prompt(self, client: TestClient) -> None:
        response = client.post(
            "/v1/completions",
            json={"model": "test-model"},
        )
        assert response.status_code == 422
