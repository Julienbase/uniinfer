"""Serving configuration for the UniInfer API server."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ServingConfig(BaseModel):
    """Configuration for the UniInfer REST API server."""

    model: str = Field(
        default="",
        description="HuggingFace model ID or local path to GGUF file. Empty to start without a model.",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Address to bind the server to.",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port to listen on.",
    )
    device: str = Field(
        default="auto",
        description="Device to use. 'auto' for best available, or 'cuda:0', 'rocm:0', 'vulkan:0', 'cpu'.",
    )
    quantization: str = Field(
        default="auto",
        description="Quantization level. 'auto' selects based on available VRAM.",
    )
    context_length: int = Field(
        default=4096,
        ge=256,
        le=131072,
        description="Context window size in tokens.",
    )
    max_concurrent_requests: int = Field(
        default=64,
        ge=1,
        le=1024,
        description="Maximum number of requests that can be queued.",
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Request timeout in seconds.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for Bearer token authentication.",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v == "auto":
            return v
        parts = v.split(":")
        valid_types = {"cuda", "rocm", "vulkan", "cpu"}
        if parts[0] not in valid_types:
            raise ValueError(
                f"Device type must be one of {valid_types}, got '{parts[0]}'"
            )
        if len(parts) > 1:
            try:
                int(parts[1])
            except ValueError:
                raise ValueError(f"Device ID must be an integer, got '{parts[1]}'")
        return v

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str) -> str:
        allowed = {"auto", "f16", "q8_0", "q4_k_m"}
        if v not in allowed:
            raise ValueError(f"Quantization must be one of {allowed}, got '{v}'")
        return v
