"""Async scheduler for concurrent inference request handling.

Bridges the async FastAPI layer with the synchronous Engine by running
inference in a thread pool. Requests are queued and processed sequentially
(llama.cpp is single-threaded per model), but the async interface allows
the server to handle many concurrent connections cleanly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Optional

from uniinfer.backends.interface import GenerationResult, StreamChunk
from uniinfer.engine.engine import Engine
from uniinfer.engine.request import InferenceRequest, RequestStatus

logger = logging.getLogger(__name__)


class Scheduler:
    """Async request scheduler wrapping a synchronous Engine.

    Accepts inference requests via async methods, processes them
    sequentially through the Engine in a thread pool, and delivers
    results via futures (blocking) or async queues (streaming).
    """

    def __init__(self, engine: Engine, max_waiting: int = 64) -> None:
        self._engine = engine
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=max_waiting)
        self._results: dict[str, asyncio.Future[GenerationResult]] = {}
        self._stream_queues: dict[str, asyncio.Queue[Optional[StreamChunk]]] = {}
        self._active_request: Optional[InferenceRequest] = None
        self._running = False
        self._task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._max_waiting = max_waiting

    @property
    def queue_depth(self) -> int:
        """Number of requests currently waiting in the queue."""
        return self._queue.qsize()

    @property
    def is_processing(self) -> bool:
        """Whether a request is currently being processed."""
        return self._active_request is not None

    async def start(self) -> None:
        """Start the scheduler background loop."""
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started (max_waiting=%d)", self._max_waiting)

    async def stop(self) -> None:
        """Stop the scheduler and wait for the background loop to exit."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Scheduler stopped")

    async def add_request(self, request: InferenceRequest) -> str:
        """Submit an inference request to the queue.

        Returns the request ID. Raises RuntimeError if the queue is full.
        """
        if self._queue.full():
            raise RuntimeError(
                f"Request queue is full ({self._max_waiting} requests). "
                f"Try again later."
            )

        loop = asyncio.get_running_loop()

        if request.stream:
            self._stream_queues[request.request_id] = asyncio.Queue()
        else:
            self._results[request.request_id] = loop.create_future()

        await self._queue.put(request)
        logger.debug("Request %s queued (depth=%d)", request.request_id, self._queue.qsize())
        return request.request_id

    async def get_result(self, request_id: str) -> GenerationResult:
        """Wait for and return a non-streaming result."""
        try:
            return await self._results[request_id]
        finally:
            self._results.pop(request_id, None)

    async def stream_result(self, request_id: str) -> AsyncGenerator[StreamChunk, None]:
        """Yield streaming chunks as they become available."""
        queue = self._stream_queues.get(request_id)
        if queue is None:
            raise RuntimeError(f"No stream queue for request {request_id}")

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            self._stream_queues.pop(request_id, None)

    async def _run_loop(self) -> None:
        """Main scheduler loop — pulls requests and processes them."""
        loop = asyncio.get_running_loop()

        while self._running:
            try:
                request = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            self._active_request = request
            request.status = RequestStatus.RUNNING

            try:
                if request.stream:
                    await loop.run_in_executor(None, self._process_stream, request)
                else:
                    await loop.run_in_executor(None, self._process_generate, request)
            except Exception as exc:
                logger.error("Error processing request %s: %s", request.request_id, exc)
                self._handle_error(request, exc)
            finally:
                request.status = RequestStatus.FINISHED
                self._active_request = None

    def _process_generate(self, request: InferenceRequest) -> None:
        """Process a non-streaming request (runs in thread pool)."""
        loop = self._loop
        if loop is None:
            return

        try:
            if request.is_chat and request.messages is not None:
                result = self._engine.chat(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                )
            elif request.prompt is not None:
                result = self._engine.generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                )
            else:
                raise ValueError("Request has neither prompt nor messages")

            future = self._results.get(request.request_id)
            if future is not None and not future.done():
                loop.call_soon_threadsafe(future.set_result, result)

        except Exception as exc:
            future = self._results.get(request.request_id)
            if future is not None and not future.done():
                loop.call_soon_threadsafe(future.set_exception, exc)

    def _process_stream(self, request: InferenceRequest) -> None:
        """Process a streaming request (runs in thread pool)."""
        loop = self._loop
        queue = self._stream_queues.get(request.request_id)
        if queue is None or loop is None:
            return

        try:
            if request.is_chat and request.messages is not None:
                chunks = self._engine.chat_stream(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                )
            elif request.prompt is not None:
                chunks = self._engine.stream(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                )
            else:
                raise ValueError("Request has neither prompt nor messages")

            for chunk in chunks:
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)

        except Exception as exc:
            logger.error("Stream error for request %s: %s", request.request_id, exc)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    def _handle_error(self, request: InferenceRequest, exc: Exception) -> None:
        """Handle errors for both streaming and non-streaming requests."""
        loop = self._loop
        request.status = RequestStatus.ABORTED

        if request.stream:
            queue = self._stream_queues.get(request.request_id)
            if queue is not None and loop is not None:
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                except Exception:
                    pass
        else:
            future = self._results.get(request.request_id)
            if future is not None and not future.done() and loop is not None:
                loop.call_soon_threadsafe(future.set_exception, exc)
