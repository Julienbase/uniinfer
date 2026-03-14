"""UniInfer REST API server — OpenAI-compatible inference endpoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from uniinfer.api.routes_completions import create_completions_router
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
                # Skip auth for health and metrics endpoints
                if request.url.path in ("/health", "/metrics"):
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

        @self.app.get("/health")
        async def health() -> dict[str, str]:
            """Health check endpoint."""
            return {
                "status": "ok",
                "model": self.config.model,
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

    def run(self) -> None:
        """Start the uvicorn server."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )
