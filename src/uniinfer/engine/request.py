"""Internal inference request representation."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RequestStatus(str, Enum):
    """Status of an inference request in the scheduler."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class InferenceRequest:
    """An inference request tracked by the scheduler.

    Attributes:
        request_id: Unique identifier for this request.
        prompt: Text prompt for completion requests.
        messages: Message list for chat requests.
        is_chat: Whether this is a chat request.
        stream: Whether to stream the response.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        stop: Stop sequences.
        arrival_time: Unix timestamp when the request was submitted.
        status: Current request status.
    """

    request_id: str = field(default_factory=lambda: f"req-{uuid.uuid4().hex[:16]}")
    prompt: Optional[str] = None
    messages: Optional[list[dict[str, str]]] = None
    is_chat: bool = False
    stream: bool = False
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[list[str]] = None
    arrival_time: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.WAITING
