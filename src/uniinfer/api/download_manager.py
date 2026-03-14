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

        Args:
            model_id: HuggingFace model ID.
            quantization: Quantization level.
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

            # Check if already cached
            from uniinfer.models.registry import get_cache_path

            cached_path = get_cache_path(model_id, quantization, cache_dir)
            if cached_path.exists() and cached_path.stat().st_size > 0:
                yield DownloadProgress(
                    status="complete",
                    message="Model already cached",
                    progress=1.0,
                    path=str(cached_path),
                ).to_sse()
                return

            # Run download in thread pool with progress callback
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[DownloadProgress] = asyncio.Queue()

            progress_cb = _ProgressCallback(queue, loop)

            # Start download in background thread
            download_future = loop.run_in_executor(
                None,
                self._run_download,
                model_id,
                quantization,
                cache_dir,
                progress_cb,
            )

            # Yield progress events as they arrive
            while True:
                # Check for progress events with timeout
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield event.to_sse()
                except asyncio.TimeoutError:
                    pass

                # Check if download is done
                if download_future.done():
                    # Drain remaining queue events
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
        progress_cb: _ProgressCallback,
    ) -> Path:
        """Run the actual download in a worker thread."""
        from huggingface_hub import hf_hub_download

        from uniinfer.models.quantization import get_gguf_search_patterns
        from uniinfer.models.registry import (
            _cache_dir_for_model,
            _DEFAULT_CACHE_DIR,
            _find_gguf_variant_repo,
            _metadata_path,
            _search_gguf_in_repo,
            get_cache_path,
        )

        base = Path(cache_dir) / "models" if cache_dir else _DEFAULT_CACHE_DIR
        cached_path = get_cache_path(model_id, quantization, cache_dir)

        # Search for GGUF file
        gguf_filename = _search_gguf_in_repo(model_id, quantization)
        source_repo = model_id

        if gguf_filename is None:
            variant_repo = _find_gguf_variant_repo(model_id)
            if variant_repo:
                gguf_filename = _search_gguf_in_repo(variant_repo, quantization)
                if gguf_filename:
                    source_repo = variant_repo

        if gguf_filename is None:
            raise RuntimeError(f"No GGUF file found for model '{model_id}'")

        # Download with progress tracking via tqdm callback
        cached_path.parent.mkdir(parents=True, exist_ok=True)

        # hf_hub_download doesn't have a simple progress callback,
        # but we can get file size first and monitor the download
        downloaded_path = hf_hub_download(
            repo_id=source_repo,
            filename=gguf_filename,
            cache_dir=str(base / "_hf_cache"),
        )

        # Symlink/copy to our cache
        downloaded = Path(downloaded_path)
        if not cached_path.exists():
            try:
                cached_path.symlink_to(downloaded)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(str(downloaded), str(cached_path))

        # Save metadata
        import json as _json

        metadata_file = _metadata_path(model_id, base)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "model_id": model_id,
            "source_repo": source_repo,
            "gguf_filename": gguf_filename,
            "quantization": quantization,
            "source": "gguf_variant" if source_repo != model_id else "direct",
        }
        try:
            with open(metadata_file, "w") as f:
                _json.dump(metadata, f, indent=2)
        except OSError:
            pass

        return cached_path
