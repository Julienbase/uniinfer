"""Download manager with progress tracking for the dashboard API."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    """Snapshot of a download in progress."""

    status: str  # "checking", "downloading", "complete", "error"
    message: str = ""
    progress: float = 0.0  # 0.0 - 1.0
    downloaded_gb: float = 0.0
    total_gb: float = 0.0
    path: str = ""

    def to_sse(self) -> str:
        data = {
            "status": self.status,
            "message": self.message,
            "progress": round(self.progress, 3),
            "downloaded_gb": round(self.downloaded_gb, 2),
            "total_gb": round(self.total_gb, 2),
        }
        if self.path:
            data["path"] = self.path
        return f"data: {json.dumps(data)}\n\n"


class _ProgressCallback:
    """tqdm-compatible callback that pushes progress to a thread-safe queue."""

    def __init__(self, queue: asyncio.Queue[DownloadProgress], loop: asyncio.AbstractEventLoop) -> None:
        self._queue = queue
        self._loop = loop
        self._total = 0
        self._downloaded = 0
        self._last_reported = 0.0

    def __call__(self, total: int, downloaded: int, _filename: str = "") -> None:
        self._total = total
        self._downloaded = downloaded
        total_gb = total / (1024**3) if total > 0 else 0
        downloaded_gb = downloaded / (1024**3)
        progress = downloaded / total if total > 0 else 0

        # Throttle: only report every 2% change
        if progress - self._last_reported < 0.02 and progress < 1.0:
            return
        self._last_reported = progress

        event = DownloadProgress(
            status="downloading",
            message=f"Downloading... {progress:.0%}",
            progress=progress,
            downloaded_gb=downloaded_gb,
            total_gb=total_gb,
        )
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)


@dataclass
class _ActiveDownload:
    model_id: str
    quantization: str
    task: Optional[asyncio.Task] = None


class DownloadManager:
    """Manages model downloads with progress tracking.

    Ensures only one download per model/quant runs at a time.
    """

    def __init__(self) -> None:
        self._active: dict[str, _ActiveDownload] = {}
        self._lock = threading.Lock()

    def _key(self, model_id: str, quantization: str) -> str:
        return f"{model_id}::{quantization}"

    def is_downloading(self, model_id: str, quantization: str) -> bool:
        return self._key(model_id, quantization) in self._active

    async def download_with_progress(
        self,
        model_id: str,
        quantization: str = "q4_k_m",
        cache_dir: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Download a model and yield SSE progress events.

        Supports GGUF, ONNX, and SafeTensors formats. Auto-detects the
        repo format on HuggingFace.

        Args:
            model_id: HuggingFace model ID.
            quantization: Quantization level (used for GGUF only).
            cache_dir: Cache directory override.

        Yields:
            SSE-formatted progress strings.
        """
        key = self._key(model_id, quantization)

        with self._lock:
            if key in self._active:
                yield DownloadProgress(
                    status="error",
                    message=f"Download already in progress for {model_id} ({quantization})",
                ).to_sse()
                return
            self._active[key] = _ActiveDownload(model_id=model_id, quantization=quantization)

        try:
            yield DownloadProgress(
                status="checking",
                message="Checking cache and resolving model...",
            ).to_sse()

            # Check if already cached (any format)
            from uniinfer.models.registry import is_cached

            if is_cached(model_id, quantization, cache_dir):
                from uniinfer.models.registry import get_cached_path

                existing = get_cached_path(model_id, quantization, cache_dir)
                yield DownloadProgress(
                    status="complete",
                    message="Model already cached",
                    progress=1.0,
                    path=str(existing or ""),
                ).to_sse()
                return

            # Detect format and report
            yield DownloadProgress(
                status="checking",
                message="Detecting model format on HuggingFace...",
            ).to_sse()

            # Run download in thread pool
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[DownloadProgress] = asyncio.Queue()

            download_future = loop.run_in_executor(
                None,
                self._run_download,
                model_id,
                quantization,
                cache_dir,
                queue,
                loop,
            )

            # Yield progress events as they arrive
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield event.to_sse()
                except asyncio.TimeoutError:
                    pass

                if download_future.done():
                    while not queue.empty():
                        event = queue.get_nowait()
                        yield event.to_sse()
                    break

            # Get result or error
            try:
                result_path = download_future.result()
                yield DownloadProgress(
                    status="complete",
                    message="Download complete",
                    progress=1.0,
                    path=str(result_path),
                ).to_sse()
            except Exception as exc:
                yield DownloadProgress(
                    status="error",
                    message=str(exc),
                ).to_sse()

        finally:
            with self._lock:
                self._active.pop(key, None)

    def _run_download(
        self,
        model_id: str,
        quantization: str,
        cache_dir: Optional[str],
        queue: asyncio.Queue[DownloadProgress],
        loop: asyncio.AbstractEventLoop,
    ) -> Path:
        """Run the actual download in a worker thread.

        Delegates to registry.download_model() which handles all formats.
        """
        from uniinfer.models.registry import detect_repo_format

        fmt = detect_repo_format(model_id)

        # Notify about detected format
        event = DownloadProgress(
            status="downloading",
            message=f"Detected {fmt.upper()} format, downloading...",
            progress=0.0,
        )
        loop.call_soon_threadsafe(queue.put_nowait, event)

        from uniinfer.models.registry import download_model

        result = download_model(
            model_id=model_id,
            quantization=quantization,
            cache_dir=cache_dir,
        )

        return result
