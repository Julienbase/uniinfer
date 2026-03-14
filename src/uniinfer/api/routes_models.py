"""Model listing endpoint (OpenAI-compatible)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter

from uniinfer.api.schemas import ModelInfo, ModelListResponse

if TYPE_CHECKING:
    from uniinfer.api.server import UniInferServer


def create_models_router(server: UniInferServer) -> APIRouter:
    """Create the models router.

    Args:
        server: The UniInferServer instance.

    Returns:
        FastAPI APIRouter with /v1/models endpoint.
    """
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models() -> ModelListResponse:
        """List available models (OpenAI-compatible)."""
        models: list[ModelInfo] = []

        if server.engine is not None:
            info = server.engine.info()
            models.append(
                ModelInfo(
                    id=info["model"],
                    owned_by="uniinfer",
                )
            )

        return ModelListResponse(data=models)

    return router
