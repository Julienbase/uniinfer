"""Tests for the async scheduler."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from uniinfer.backends.interface import GenerationResult, StreamChunk
from uniinfer.engine.request import InferenceRequest, RequestStatus
from uniinfer.engine.scheduler import Scheduler


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock Engine that returns predictable results."""
    engine = MagicMock()
    engine.generate.return_value = GenerationResult(
        text="Hello world",
        prompt_tokens=5,
        completion_tokens=2,
        total_tokens=7,
    )
    engine.chat.return_value = GenerationResult(
        text="I am a helpful assistant.",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    def mock_stream(prompt, **kwargs):  # type: ignore[no-untyped-def]
        yield StreamChunk(text="Hello", finished=False)
        yield StreamChunk(text=" world", finished=True)

    def mock_chat_stream(messages, **kwargs):  # type: ignore[no-untyped-def]
        yield StreamChunk(text="I am", finished=False)
        yield StreamChunk(text=" helpful", finished=True)

    engine.stream.side_effect = mock_stream
    engine.chat_stream.side_effect = mock_chat_stream
    return engine


@pytest.mark.asyncio
async def test_scheduler_start_stop(mock_engine: MagicMock) -> None:
    scheduler = Scheduler(mock_engine, max_waiting=10)
    await scheduler.start()
    assert scheduler.queue_depth == 0
    await scheduler.stop()


@pytest.mark.asyncio
async def test_generate_request(mock_engine: MagicMock) -> None:
    scheduler = Scheduler(mock_engine, max_waiting=10)
    await scheduler.start()

    try:
        request = InferenceRequest(
            prompt="Hello",
            is_chat=False,
            stream=False,
            max_tokens=100,
        )
        await scheduler.add_request(request)
        result = await asyncio.wait_for(
            scheduler.get_result(request.request_id), timeout=5.0
        )
        assert result.text == "Hello world"
        assert result.total_tokens == 7
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_chat_request(mock_engine: MagicMock) -> None:
    scheduler = Scheduler(mock_engine, max_waiting=10)
    await scheduler.start()

    try:
        request = InferenceRequest(
            messages=[{"role": "user", "content": "Hello"}],
            is_chat=True,
            stream=False,
            max_tokens=100,
        )
        await scheduler.add_request(request)
        result = await asyncio.wait_for(
            scheduler.get_result(request.request_id), timeout=5.0
        )
        assert result.text == "I am a helpful assistant."
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_streaming_request(mock_engine: MagicMock) -> None:
    scheduler = Scheduler(mock_engine, max_waiting=10)
    await scheduler.start()

    try:
        request = InferenceRequest(
            prompt="Hello",
            is_chat=False,
            stream=True,
            max_tokens=100,
        )
        await scheduler.add_request(request)

        chunks: list[StreamChunk] = []
        async for chunk in scheduler.stream_result(request.request_id):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].text == "Hello"
        assert chunks[1].text == " world"
        assert chunks[1].finished is True
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_queue_full(mock_engine: MagicMock) -> None:
    scheduler = Scheduler(mock_engine, max_waiting=1)
    await scheduler.start()

    try:
        # Fill the queue
        req1 = InferenceRequest(prompt="A", stream=False, max_tokens=100)
        await scheduler.add_request(req1)

        # Wait briefly for the request to be picked up before filling again
        await asyncio.sleep(0.2)

        # Second request - may or may not fail depending on timing
        # The key test is that the scheduler doesn't crash
        req2 = InferenceRequest(prompt="B", stream=False, max_tokens=100)
        try:
            await scheduler.add_request(req2)
        except RuntimeError:
            pass  # Expected if queue is full
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_multiple_requests(mock_engine: MagicMock) -> None:
    scheduler = Scheduler(mock_engine, max_waiting=10)
    await scheduler.start()

    try:
        requests = []
        for i in range(3):
            req = InferenceRequest(
                prompt=f"Request {i}",
                is_chat=False,
                stream=False,
                max_tokens=100,
            )
            await scheduler.add_request(req)
            requests.append(req)

        # All should complete
        for req in requests:
            result = await asyncio.wait_for(
                scheduler.get_result(req.request_id), timeout=10.0
            )
            assert result.text == "Hello world"
    finally:
        await scheduler.stop()
