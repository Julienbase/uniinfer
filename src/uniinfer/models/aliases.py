"""Model aliases — short names for popular HuggingFace models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelAlias:
    """Metadata for a model alias."""

    repo_id: str
    display_name: str
    param_count_billions: float
    default_quant: str = "q4_k_m"
    default_context_length: int = 4096


MODEL_ALIASES: dict[str, ModelAlias] = {
    "tinyllama-1b": ModelAlias(
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        display_name="TinyLlama 1.1B Chat",
        param_count_billions=1.1,
        default_quant="q4_k_m",
        default_context_length=2048,
    ),
    "phi-3-mini": ModelAlias(
        repo_id="bartowski/Phi-3.1-mini-4k-instruct-GGUF",
        display_name="Phi-3.1 Mini 4K Instruct",
        param_count_billions=3.8,
        default_quant="q4_k_m",
        default_context_length=4096,
    ),
    "gemma-2b": ModelAlias(
        repo_id="bartowski/gemma-2-2b-it-GGUF",
        display_name="Gemma 2 2B Instruct",
        param_count_billions=2.6,
        default_quant="q4_k_m",
        default_context_length=8192,
    ),
    "mistral-7b": ModelAlias(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        display_name="Mistral 7B Instruct v0.2",
        param_count_billions=7.24,
        default_quant="q4_k_m",
        default_context_length=4096,
    ),
    "llama-3.1-8b": ModelAlias(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        display_name="Llama 3.1 8B Instruct",
        param_count_billions=8.03,
        default_quant="q4_k_m",
        default_context_length=8192,
    ),
    "qwen-2.5-7b": ModelAlias(
        repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
        display_name="Qwen 2.5 7B Instruct",
        param_count_billions=7.62,
        default_quant="q4_k_m",
        default_context_length=4096,
    ),
    "llama-3.3-70b": ModelAlias(
        repo_id="bartowski/Llama-3.3-70B-Instruct-GGUF",
        display_name="Llama 3.3 70B Instruct",
        param_count_billions=70.6,
        default_quant="q4_k_m",
        default_context_length=4096,
    ),
}


def resolve_alias(model: str) -> str:
    """Resolve a model alias to a full HuggingFace repo ID.

    If the model is a known alias, returns the full repo ID.
    Otherwise, returns the model string unchanged (passthrough).
    """
    alias = MODEL_ALIASES.get(model.lower())
    if alias:
        return alias.repo_id
    return model


def get_alias_info(model: str) -> Optional[ModelAlias]:
    """Get alias metadata for a model, if it is a known alias."""
    return MODEL_ALIASES.get(model.lower())


def list_aliases() -> list[tuple[str, ModelAlias]]:
    """Return all known aliases sorted by parameter count."""
    return sorted(MODEL_ALIASES.items(), key=lambda x: x[1].param_count_billions)
