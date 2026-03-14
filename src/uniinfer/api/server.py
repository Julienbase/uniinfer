"""UniInfer REST API server — OpenAI-compatible inference endpoint."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from uniinfer.api.chat_store import ChatStore
from uniinfer.api.routes_completions import create_completions_router
from uniinfer.api.routes_dashboard import create_dashboard_router
from uniinfer.api.routes_models import create_models_router
from uniinfer.api.schemas import ErrorDetail, ErrorResponse
from uniinfer.config.serving_config import ServingConfig
from uniinfer.engine.engine import Engine
from uniinfer.engine.scheduler import Scheduler
from uniinfer.metrics.prometheus import MetricsTracker

logger = logging.getLogger(__name__)


class UniInferServer:
    """OpenAI-compatible REST API server for UniInfer.

    Loads a model via the Engine, starts a scheduler for concurrent
    request handling, and serves the FastAPI application.
    """

    def __init__(self, config: ServingConfig) -> None:
        self.config = config
        self.engine: Optional[Engine] = None
        self.scheduler: Optional[Scheduler] = None
        self.metrics = MetricsTracker()
        self.chat_store = ChatStore()
        self._start_time = time.time()
        self._swap_lock = asyncio.Lock()

        self.app = FastAPI(
            title="UniInfer",
            description="Hardware-agnostic AI inference runtime — OpenAI-compatible API",
            version="0.5.0",
            lifespan=self._lifespan,
        )
        self._setup_middleware()
        self._setup_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None, None]:
        """FastAPI lifespan: load model and start scheduler on startup."""
        if self.config.model:
            logger.info("Loading model: %s", self.config.model)
            self.engine = Engine(
                model=self.config.model,
                device=self.config.device,
                quantization=self.config.quantization,
                context_length=self.config.context_length,
            )
            logger.info("Model loaded. Starting scheduler...")
            self.scheduler = Scheduler(
                engine=self.engine,
                max_waiting=self.config.max_concurrent_requests,
            )
            await self.scheduler.start()
        else:
            logger.info("Starting without a model. Load one via the dashboard or API.")
        logger.info(
            "UniInfer server ready at http://%s:%d", self.config.host, self.config.port
        )
        yield
        # Shutdown
        logger.info("Shutting down...")
        if self.scheduler is not None:
            await self.scheduler.stop()
        if self.engine is not None:
            self.engine.close()

    def _setup_middleware(self) -> None:
        """Configure CORS and optional API key auth."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        if self.config.api_key is not None:
            @self.app.middleware("http")
            async def api_key_auth(request: Request, call_next):  # type: ignore[no-untyped-def]
                # Skip auth for health, metrics, and dashboard endpoints
                if (
                    request.url.path in ("/health", "/metrics")
                    or request.url.path.startswith("/api/dashboard")
                    or request.url.path.startswith("/dashboard")
                ):
                    return await call_next(request)

                auth_header = request.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    return JSONResponse(
                        status_code=401,
                        content=ErrorResponse(
                            error=ErrorDetail(
                                message="Missing or invalid Authorization header. "
                                "Use 'Bearer <api_key>'.",
                                type="authentication_error",
                                code="invalid_api_key",
                            )
                        ).model_dump(),
                    )

                token = auth_header[len("Bearer "):]
                if token != self.config.api_key:
                    return JSONResponse(
                        status_code=401,
                        content=ErrorResponse(
                            error=ErrorDetail(
                                message="Invalid API key.",
                                type="authentication_error",
                                code="invalid_api_key",
                            )
                        ).model_dump(),
                    )

                return await call_next(request)

    def _setup_routes(self) -> None:
        """Register all route handlers."""
        self.app.include_router(create_completions_router(self))
        self.app.include_router(create_models_router(self))
        self.app.include_router(create_dashboard_router(self, self._start_time))

        # Serve React dashboard build if available
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists() and (static_dir / "index.html").exists():
            # Mount assets subdirectory for JS/CSS bundles
            assets_dir = static_dir / "assets"
            if assets_dir.exists():
                self.app.mount(
                    "/dashboard/assets",
                    StaticFiles(directory=str(assets_dir)),
                    name="dashboard-assets",
                )

            @self.app.get("/dashboard/{full_path:path}")
            async def serve_spa(full_path: str) -> FileResponse:
                """Serve the React SPA for all dashboard routes."""
                # Check if it's a real file first
                file_path = static_dir / full_path
                if file_path.is_file() and ".." not in full_path:
                    return FileResponse(str(file_path))
                return FileResponse(str(static_dir / "index.html"))

            @self.app.get("/dashboard")
            async def serve_dashboard() -> FileResponse:
                """Serve the dashboard index page."""
                return FileResponse(str(static_dir / "index.html"))

        @self.app.get("/health")
        async def health() -> dict:
            """Health check endpoint."""
            model = ""
            if self.engine is not None:
                model = self.engine.info().get("model", "")
            return {
                "status": "ok",
                "model": model,
                "loaded": self.engine is not None,
            }

        @self.app.get("/metrics")
        async def metrics() -> PlainTextResponse:
            """Prometheus metrics endpoint."""
            if self.scheduler is not None:
                self.metrics.set_queue_depth(self.scheduler.queue_depth)
                self.metrics.set_active_requests(1 if self.scheduler.is_processing else 0)
            return PlainTextResponse(
                content=self.metrics.get_metrics(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

    async def swap_model(
        self,
        model: str,
        device: str = "auto",
        quantization: str = "auto",
        context_length: int = 4096,
    ) -> dict:
        """Hot-swap the loaded model.

        Stops the scheduler, closes the old engine, creates a new engine
        and scheduler with the new model.

        Args:
            model: Model identifier (alias, HuggingFace ID, or local path).
            device: Device preference.
            quantization: Quantization level.
            context_length: Context window size.

        Returns:
            Info dict from the new engine.

        Raises:
            RuntimeError: If a swap is already in progress.
        """
        if self._swap_lock.locked():
            raise RuntimeError("Model swap already in progress")

        async with self._swap_lock:
            logger.info("Swapping model to: %s", model)

            # Wait for active request to finish (up to 60s)
            if self.scheduler is not None and self.scheduler.is_processing:
                logger.info("Waiting for active request to complete...")
                for _ in range(600):
                    if not self.scheduler.is_processing:
                        break
                    await asyncio.sleep(0.1)

            # Stop old scheduler
            if self.scheduler is not None:
                await self.scheduler.stop()
                self.scheduler = None

            # Close old engine
            if self.engine is not None:
                self.engine.close()
                self.engine = None

            # Create new engine (blocking — runs in executor)
            loop = asyncio.get_running_loop()
            new_engine = await loop.run_in_executor(
                None,
                lambda: Engine(
                    model=model,
                    device=device,
                    quantization=quantization,
                    context_length=context_length,
                ),
            )

            # Create and start new scheduler
            new_scheduler = Scheduler(
                engine=new_engine,
                max_waiting=self.config.max_concurrent_requests,
            )
            await new_scheduler.start()

            # Atomic swap
            self.engine = new_engine
            self.scheduler = new_scheduler

            logger.info("Model swapped successfully to: %s", model)
            return self.engine.info()

    def run(self) -> None:
        """Start the uvicorn server."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )
