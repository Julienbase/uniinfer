"""Prometheus metrics for UniInfer server monitoring."""

from __future__ import annotations

import time
from typing import Optional

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks inference server metrics using Prometheus client.

    Falls back to no-ops if prometheus_client is not installed.
    """

    def __init__(self) -> None:
        if not _PROMETHEUS_AVAILABLE:
            logger.info("prometheus_client not installed, metrics disabled")
            return

        self._registry = CollectorRegistry()

        self.requests_total = Counter(
            "uniinfer_requests_total",
            "Total inference requests",
            ["endpoint", "status"],
            registry=self._registry,
        )
        self.tokens_generated = Counter(
            "uniinfer_tokens_generated_total",
            "Total tokens generated",
            registry=self._registry,
        )
        self.prompt_tokens = Counter(
            "uniinfer_prompt_tokens_total",
            "Total prompt tokens processed",
            registry=self._registry,
        )
        self.request_latency = Histogram(
            "uniinfer_request_duration_seconds",
            "Request duration in seconds",
            ["endpoint"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
            registry=self._registry,
        )
        self.time_to_first_token = Histogram(
            "uniinfer_time_to_first_token_seconds",
            "Time to first token for streaming requests",
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self._registry,
        )
        self.queue_depth = Gauge(
            "uniinfer_queue_depth",
            "Current number of waiting requests",
            registry=self._registry,
        )
        self.active_requests = Gauge(
            "uniinfer_active_requests",
            "Number of requests currently being processed",
            registry=self._registry,
        )

    def record_request(
        self,
        endpoint: str,
        status: str,
        prompt_tokens_count: int,
        completion_tokens_count: int,
        duration: float,
    ) -> None:
        """Record a completed request."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.requests_total.labels(endpoint=endpoint, status=status).inc()
        self.prompt_tokens.inc(prompt_tokens_count)
        self.tokens_generated.inc(completion_tokens_count)
        self.request_latency.labels(endpoint=endpoint).observe(duration)

    def record_first_token(self, duration: float) -> None:
        """Record time to first token for a streaming request."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.time_to_first_token.observe(duration)

    def set_queue_depth(self, depth: int) -> None:
        """Update the current queue depth gauge."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.queue_depth.set(depth)

    def set_active_requests(self, count: int) -> None:
        """Update the active requests gauge."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self.active_requests.set(count)

    def get_metrics(self) -> bytes:
        """Return Prometheus metrics in exposition format."""
        if not _PROMETHEUS_AVAILABLE:
            return b"# prometheus_client not installed\n"
        return generate_latest(self._registry)
