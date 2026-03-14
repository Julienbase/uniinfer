"""OpenAI-compatible request and response schemas."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


def _generate_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="The role of the message author."
    )
    content: str = Field(..., description="The message content.")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(..., description="Model identifier.")
    messages: list[ChatMessage] = Field(
        ..., min_length=1, description="List of messages in the conversation."
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=512, ge=1)
    stream: bool = Field(default=False)
    stop: Optional[Union[str, list[str]]] = Field(default=None)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    n: int = Field(default=1, ge=1, le=1, description="Only n=1 is supported.")

    def get_stop_list(self) -> Optional[list[str]]:
        if self.stop is None:
            return None
        if isinstance(self.stop, str):
            return [self.stop]
        return self.stop

    def to_messages_dicts(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]


class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request."""

    model: str = Field(..., description="Model identifier.")
    prompt: str = Field(..., description="The prompt to generate from.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=512, ge=1)
    stream: bool = Field(default=False)
    stop: Optional[Union[str, list[str]]] = Field(default=None)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    n: int = Field(default=1, ge=1, le=1, description="Only n=1 is supported.")

    def get_stop_list(self) -> Optional[list[str]]:
        if self.stop is None:
            return None
        if isinstance(self.stop, str):
            return [self.stop]
        return self.stop


# ---------------------------------------------------------------------------
# Response models — non-streaming
# ---------------------------------------------------------------------------


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: _generate_id("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class CompletionChoice(BaseModel):
    """A single choice in a text completion response."""

    index: int = 0
    text: str
    finish_reason: Optional[str] = "stop"


class CompletionResponse(BaseModel):
    """OpenAI-compatible text completion response."""

    id: str = Field(default_factory=lambda: _generate_id("cmpl"))
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Response models — streaming (SSE)
# ---------------------------------------------------------------------------


class ChatDelta(BaseModel):
    """Delta content for streaming chat completions."""

    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    """A single choice in a streaming chat completion chunk."""

    index: int = 0
    delta: ChatDelta
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming chat completion chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamChoice]


class CompletionStreamChoice(BaseModel):
    """A single choice in a streaming text completion chunk."""

    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = None


class CompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming text completion chunk."""

    id: str
    object: Literal["text_completion.chunk"] = "text_completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionStreamChoice]


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "uniinfer"


class ModelListResponse(BaseModel):
    """Response for the models listing endpoint."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Error detail matching OpenAI error format."""

    message: str
    type: str
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""

    error: ErrorDetail
