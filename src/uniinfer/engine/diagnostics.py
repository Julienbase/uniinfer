"""Inference diagnostics — timing, throughput, and session metrics."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Metrics for a single inference call."""

    start_time: float = 0.0
    end_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    method: str = ""  # "generate", "stream", "chat", "chat_stream"

    @property
    def elapsed_seconds(self) -> float:
        if self.end_time <= 0 or self.start_time <= 0:
            return 0.0
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed <= 0 or self.completion_tokens <= 0:
            return 0.0
        return self.completion_tokens / elapsed

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": round(self.tokens_per_second, 1),
        }


@dataclass
class SessionDiagnostics:
    """Aggregated diagnostics for an engine session."""

    model_load_time: float = 0.0
    inference_history: list[InferenceMetrics] = field(default_factory=list)

    @property
    def total_inferences(self) -> int:
        return len(self.inference_history)

    @property
    def total_tokens_generated(self) -> int:
        return sum(m.completion_tokens for m in self.inference_history)

    @property
    def total_inference_time(self) -> float:
        return sum(m.elapsed_seconds for m in self.inference_history)

    @property
    def average_tokens_per_second(self) -> float:
        total_time = self.total_inference_time
        total_tokens = self.total_tokens_generated
        if total_time <= 0 or total_tokens <= 0:
            return 0.0
        return total_tokens / total_time

    @property
    def peak_tokens_per_second(self) -> float:
        if not self.inference_history:
            return 0.0
        return max(m.tokens_per_second for m in self.inference_history)

    def record(self, metrics: InferenceMetrics) -> None:
        """Record an inference call's metrics."""
        self.inference_history.append(metrics)
        logger.debug(
            "Inference [%s]: %d tokens in %.2fs (%.1f tok/s)",
            metrics.method,
            metrics.completion_tokens,
            metrics.elapsed_seconds,
            metrics.tokens_per_second,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_load_time_seconds": round(self.model_load_time, 3),
            "total_inferences": self.total_inferences,
            "total_tokens_generated": self.total_tokens_generated,
            "total_inference_time_seconds": round(self.total_inference_time, 3),
            "average_tokens_per_second": round(self.average_tokens_per_second, 1),
            "peak_tokens_per_second": round(self.peak_tokens_per_second, 1),
            "last_inference": (
                self.inference_history[-1].to_dict()
                if self.inference_history
                else None
            ),
        }
