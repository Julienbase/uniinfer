"""SSE (Server-Sent Events) helpers for OpenAI-compatible streaming."""

from __future__ import annotations

from typing import AsyncGenerator

from uniinfer.api.schemas import (
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatDelta,
    CompletionStreamChoice,
    CompletionStreamResponse,
    _generate_id,
)
from uniinfer.backends.interface import StreamChunk


async def chat_stream_to_sse(
    request_id: str,
    model: str,
    chunks: AsyncGenerator[StreamChunk, None],
) -> AsyncGenerator[str, None]:
    """Convert an async stream of StreamChunks into OpenAI-format SSE data lines.

    Args:
        request_id: Unique request identifier for the response.
        model: Model name to include in each chunk.
        chunks: Async generator of StreamChunk objects.

    Yields:
        SSE-formatted data lines (``data: {json}\\n\\n``).
    """
    # First chunk includes the role delta
    first = True
    async for chunk in chunks:
        delta = ChatDelta()
        if first:
            delta.role = "assistant"
            first = False
        if chunk.text:
            delta.content = chunk.text

        finish_reason = "stop" if chunk.finished else None

        response = ChatCompletionStreamResponse(
            id=request_id,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason,
                )
            ],
        )
        yield f"data: {response.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


async def completion_stream_to_sse(
    request_id: str,
    model: str,
    chunks: AsyncGenerator[StreamChunk, None],
) -> AsyncGenerator[str, None]:
    """Convert an async stream of StreamChunks into OpenAI-format SSE data lines
    for text completions.

    Args:
        request_id: Unique request identifier for the response.
        model: Model name to include in each chunk.
        chunks: Async generator of StreamChunk objects.

    Yields:
        SSE-formatted data lines.
    """
    async for chunk in chunks:
        finish_reason = "stop" if chunk.finished else None

        response = CompletionStreamResponse(
            id=request_id,
            model=model,
            choices=[
                CompletionStreamChoice(
                    index=0,
                    text=chunk.text,
                    finish_reason=finish_reason,
                )
            ],
        )
        yield f"data: {response.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"
