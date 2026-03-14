"""OpenAI-compatible completion and chat completion endpoints."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from uniinfer.api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorDetail,
    ErrorResponse,
    UsageInfo,
    _generate_id,
)
from uniinfer.api.streaming import chat_stream_to_sse, completion_stream_to_sse
from uniinfer.engine.request import InferenceRequest

if TYPE_CHECKING:
    from uniinfer.api.server import UniInferServer

logger = logging.getLogger(__name__)


def create_completions_router(server: UniInferServer) -> APIRouter:
    """Create the completions router with endpoints bound to the server.

    Args:
        server: The UniInferServer instance providing engine, scheduler, and metrics.

    Returns:
        FastAPI APIRouter with /v1/completions and /v1/chat/completions.
    """
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):  # type: ignore[no-untyped-def]
        """Create a chat completion (OpenAI-compatible)."""
        start_time = time.time()
        request_id = _generate_id("chatcmpl")
        model_name = server.engine.info()["model"] if server.engine else request.model

        inference_req = InferenceRequest(
            request_id=request_id,
            messages=request.to_messages_dicts(),
            is_chat=True,
            stream=request.stream,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.get_stop_list(),
        )

        try:
            await server.scheduler.add_request(inference_req)
        except RuntimeError:
            raise HTTPException(
                status_code=503,
                detail="Server is overloaded. Try again later.",
            )

        if request.stream:
            chunks = server.scheduler.stream_result(request_id)
            sse_stream = chat_stream_to_sse(request_id, model_name, chunks)
            return StreamingResponse(
                sse_stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming
        try:
            result = await server.scheduler.get_result(request_id)
        except Exception as exc:
            duration = time.time() - start_time
            server.metrics.record_request(
                "/v1/chat/completions", "error", 0, 0, duration
            )
            raise HTTPException(status_code=500, detail=str(exc))

        duration = time.time() - start_time
        server.metrics.record_request(
            "/v1/chat/completions",
            "success",
            result.prompt_tokens,
            result.completion_tokens,
            duration,
        )

        return ChatCompletionResponse(
            id=request_id,
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result.text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            ),
        )

    @router.post("/v1/completions")
    async def create_completion(request: CompletionRequest):  # type: ignore[no-untyped-def]
        """Create a text completion (OpenAI-compatible)."""
        start_time = time.time()
        request_id = _generate_id("cmpl")
        model_name = server.engine.info()["model"] if server.engine else request.model

        inference_req = InferenceRequest(
            request_id=request_id,
            prompt=request.prompt,
            is_chat=False,
            stream=request.stream,
            max_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.get_stop_list(),
        )

        try:
            await server.scheduler.add_request(inference_req)
        except RuntimeError:
            raise HTTPException(
                status_code=503,
                detail="Server is overloaded. Try again later.",
            )

        if request.stream:
            chunks = server.scheduler.stream_result(request_id)
            sse_stream = completion_stream_to_sse(request_id, model_name, chunks)
            return StreamingResponse(
                sse_stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming
        try:
            result = await server.scheduler.get_result(request_id)
        except Exception as exc:
            duration = time.time() - start_time
            server.metrics.record_request("/v1/completions", "error", 0, 0, duration)
            raise HTTPException(status_code=500, detail=str(exc))

        duration = time.time() - start_time
        server.metrics.record_request(
            "/v1/completions",
            "success",
            result.prompt_tokens,
            result.completion_tokens,
            duration,
        )

        return CompletionResponse(
            id=request_id,
            model=model_name,
            choices=[
                CompletionChoice(
                    index=0,
                    text=result.text,
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            ),
        )

    return router
