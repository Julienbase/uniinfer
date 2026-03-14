"""Engine configuration using Pydantic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class EngineConfig(BaseModel):
    """Configuration for the UniInfer engine."""

    model: str = Field(
        ...,
        description="HuggingFace model ID (e.g., 'meta-llama/Llama-3.1-8B-Instruct') or local path to GGUF file.",
    )
    device: str = Field(
        default="auto",
        description="Device to use. 'auto' for best available, or 'cuda:0', 'rocm:0', 'vulkan:0', 'cpu'.",
    )
    quantization: str = Field(
        default="auto",
        description="Quantization level. 'auto' selects based on available VRAM. Options: f16, q8_0, q4_k_m.",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        le=65536,
        description="Maximum number of tokens to generate.",
    )
    context_length: int = Field(
        default=4096,
        ge=256,
        le=131072,
        description="Context window size in tokens.",
    )
    cache_dir: str = Field(
        default="",
        description="Directory for model cache. Defaults to ~/.uniinfer/cache.",
    )
    n_gpu_layers: Optional[int] = Field(
        default=None,
        description="Override number of GPU layers. None = auto-detect from device.",
    )
    n_threads: Optional[int] = Field(
        default=None,
        description="Number of CPU threads. None = auto-detect.",
    )

    @model_validator(mode="after")
    def set_default_cache_dir(self) -> EngineConfig:
        if not self.cache_dir:
            self.cache_dir = str(Path(os.path.expanduser("~")) / ".uniinfer" / "cache")
        return self

    @field_validator("quantization")
    @classmethod
    def validate_quantization(cls, v: str) -> str:
        allowed = {"auto", "f16", "q8_0", "q4_k_m"}
        if v not in allowed:
            raise ValueError(f"Quantization must be one of {allowed}, got '{v}'")
        return v

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

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)

    @property
    def is_local_model(self) -> bool:
        """Check if model refers to a local file path."""
        path = Path(self.model)
        return path.exists() and path.is_file()
