"""OpenAI-compatible completion and chat completion endpoints."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from uniinfer.api.chat_store import ChatMessage as StoreChatMessage
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

    def _get_session_id(http_request: Request, model_name: str) -> str:
        """Get or create a chat session from the request headers."""
        session_id = http_request.headers.get("x-uniinfer-session")
        source = http_request.headers.get("x-uniinfer-source", "api")
        return server.chat_store.get_or_create_session(session_id, model_name, source)

    def _store_chat_messages(
        session_id: str,
        user_messages: list[dict[str, str]],
        assistant_text: str,
        completion_tokens: int,
        elapsed: float,
    ) -> None:
        """Store user messages and assistant response in the chat store."""
        # Store user messages (skip system messages)
        for msg in user_messages:
            if msg["role"] in ("user", "system"):
                server.chat_store.add_message(
                    session_id,
                    StoreChatMessage(role=msg["role"], content=msg["content"]),
                )

        # Store assistant response
        tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0
        server.chat_store.add_message(
            session_id,
            StoreChatMessage(
                role="assistant",
                content=assistant_text,
                tokens=completion_tokens,
                tokens_per_second=round(tok_s, 1),
            ),
        )

    @router.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest, http_request: Request):  # type: ignore[no-untyped-def]
        """Create a chat completion (OpenAI-compatible)."""
        if server.engine is None or server.scheduler is None:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. Load a model first via the dashboard or API.",
            )
        start_time = time.time()
        request_id = _generate_id("chatcmpl")
        model_name = server.engine.info()["model"]

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
            session_id = _get_session_id(http_request, model_name)
            chunks = server.scheduler.stream_result(request_id)

            # Wrap SSE stream to capture full response for chat store
            async def tracked_stream():
                full_text = ""
                token_count = 0
                stream_start = time.time()
                async for line in chat_stream_to_sse(request_id, model_name, chunks):
                    yield line
                    # Extract text from SSE data
                    if line.startswith("data: ") and "[DONE]" not in line:
                        try:
                            import json
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_text += content
                                token_count += 1
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass

                # Store in chat history after stream completes
                elapsed = time.time() - stream_start
                _store_chat_messages(
                    session_id,
                    request.to_messages_dicts(),
                    full_text,
                    token_count,
                    elapsed,
                )

            return StreamingResponse(
                tracked_stream(),
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

        # Store in chat history
        session_id = _get_session_id(http_request, model_name)
        _store_chat_messages(
            session_id,
            request.to_messages_dicts(),
            result.text,
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
        if server.engine is None or server.scheduler is None:
            raise HTTPException(
                status_code=503,
                detail="No model loaded. Load a model first via the dashboard or API.",
            )
        start_time = time.time()
        request_id = _generate_id("cmpl")
        model_name = server.engine.info()["model"]

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
