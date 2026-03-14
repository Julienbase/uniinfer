"""Tests for OpenAI-compatible API schemas."""

from __future__ import annotations

import json
import time

import pytest

from uniinfer.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatDelta,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    CompletionStreamResponse,
    ErrorDetail,
    ErrorResponse,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
    _generate_id,
)


class TestChatCompletionRequest:
    def test_valid_request(self) -> None:
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert req.model == "test-model"
        assert len(req.messages) == 1
        assert req.temperature == 0.7
        assert req.stream is False
        assert req.n == 1

    def test_stop_list_from_string(self) -> None:
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            stop="</s>",
        )
        assert req.get_stop_list() == ["</s>"]

    def test_stop_list_from_list(self) -> None:
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            stop=["</s>", "<|end|>"],
        )
        assert req.get_stop_list() == ["</s>", "<|end|>"]

    def test_stop_list_none(self) -> None:
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
        )
        assert req.get_stop_list() is None

    def test_to_messages_dicts(self) -> None:
        req = ChatCompletionRequest(
            model="test",
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Hi"),
            ],
        )
        dicts = req.to_messages_dicts()
        assert len(dicts) == 2
        assert dicts[0] == {"role": "system", "content": "You are helpful."}

    def test_invalid_temperature(self) -> None:
        with pytest.raises(Exception):
            ChatCompletionRequest(
                model="test",
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=3.0,
            )

    def test_empty_messages(self) -> None:
        with pytest.raises(Exception):
            ChatCompletionRequest(model="test", messages=[])


class TestCompletionRequest:
    def test_valid_request(self) -> None:
        req = CompletionRequest(model="test", prompt="Hello world")
        assert req.prompt == "Hello world"
        assert req.max_tokens == 512

    def test_stop_string_to_list(self) -> None:
        req = CompletionRequest(model="test", prompt="Hi", stop="\n")
        assert req.get_stop_list() == ["\n"]


class TestChatCompletionResponse:
    def test_response_format(self) -> None:
        resp = ChatCompletionResponse(
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )
        assert resp.object == "chat.completion"
        assert resp.id.startswith("chatcmpl-")
        assert resp.model == "test-model"
        assert resp.choices[0].message.content == "Hello!"
        assert resp.usage.total_tokens == 6

    def test_serialization(self) -> None:
        resp = ChatCompletionResponse(
            model="test",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content="Hi"),
                )
            ],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        data = json.loads(resp.model_dump_json())
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data


class TestCompletionResponse:
    def test_response_format(self) -> None:
        resp = CompletionResponse(
            model="test",
            choices=[CompletionChoice(text="world")],
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        assert resp.object == "text_completion"
        assert resp.id.startswith("cmpl-")


class TestStreamingResponses:
    def test_chat_stream_chunk(self) -> None:
        chunk = ChatCompletionStreamResponse(
            id="chatcmpl-test",
            model="test",
            choices=[
                ChatCompletionStreamChoice(
                    delta=ChatDelta(content="Hello"),
                    finish_reason=None,
                )
            ],
        )
        assert chunk.object == "chat.completion.chunk"
        data = json.loads(chunk.model_dump_json())
        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert data["choices"][0]["finish_reason"] is None

    def test_completion_stream_chunk(self) -> None:
        chunk = CompletionStreamResponse(
            id="cmpl-test",
            model="test",
            choices=[CompletionStreamChoice(text="Hi")],
        )
        assert chunk.object == "text_completion.chunk"


class TestModelListResponse:
    def test_models_list(self) -> None:
        resp = ModelListResponse(
            data=[ModelInfo(id="test-model")]
        )
        assert resp.object == "list"
        assert len(resp.data) == 1
        assert resp.data[0].id == "test-model"
        assert resp.data[0].object == "model"
        assert resp.data[0].owned_by == "uniinfer"


class TestErrorResponse:
    def test_error_format(self) -> None:
        resp = ErrorResponse(
            error=ErrorDetail(
                message="Not found",
                type="invalid_request_error",
                code="model_not_found",
            )
        )
        data = json.loads(resp.model_dump_json())
        assert data["error"]["message"] == "Not found"
        assert data["error"]["type"] == "invalid_request_error"


class TestGenerateId:
    def test_chat_id_format(self) -> None:
        id_ = _generate_id("chatcmpl")
        assert id_.startswith("chatcmpl-")
        assert len(id_) == len("chatcmpl-") + 24

    def test_unique_ids(self) -> None:
        ids = {_generate_id() for _ in range(100)}
        assert len(ids) == 100
