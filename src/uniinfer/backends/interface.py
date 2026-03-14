"""Execution backend interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator, Optional


@dataclass(frozen=True)
class GenerationResult:
    """Result of a text generation call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class StreamChunk:
    """A single chunk from a streaming generation."""

    text: str
    finished: bool


@dataclass
class ModelHandle:
    """Opaque handle to a loaded model in a backend."""

    backend_name: str
    model_path: str
    internal: Any = None  # Backend-specific model object


class ExecutionBackend(ABC):
    """Abstract base class for inference execution backends.

    Each backend wraps a specific inference library (e.g., llama-cpp-python)
    and provides a unified interface for loading models, generating text,
    and streaming tokens.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'llama.cpp')."""
        ...

    @abstractmethod
    def load_model(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelHandle:
        """Load a model from disk.

        Args:
            model_path: Path to the model file (GGUF).
            n_gpu_layers: Number of layers to offload to GPU. -1 = all.
            n_ctx: Context window size.
            n_threads: Number of CPU threads. None = auto.
            **kwargs: Backend-specific options.

        Returns:
            A ModelHandle for use in generate/stream calls.
        """
        ...

    @abstractmethod
    def generate(
        self,
        handle: ModelHandle,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text from a prompt (blocking).

        Args:
            handle: Model handle from load_model.
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Returns:
            GenerationResult with the completed text.
        """
        ...

    @abstractmethod
    def stream(
        self,
        handle: ModelHandle,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream tokens from a prompt.

        Args:
            handle: Model handle from load_model.
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Yields:
            StreamChunk with each generated token.
        """
        ...

    @abstractmethod
    def chat(
        self,
        handle: ModelHandle,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate a chat response from a list of messages.

        Uses the model's built-in chat template from GGUF metadata.

        Args:
            handle: Model handle from load_model.
            messages: List of dicts with 'role' and 'content' keys.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Returns:
            GenerationResult with the assistant's response.
        """
        ...

    @abstractmethod
    def chat_stream(
        self,
        handle: ModelHandle,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat response from a list of messages.

        Args:
            handle: Model handle from load_model.
            messages: List of dicts with 'role' and 'content' keys.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Yields:
            StreamChunk with each generated token.
        """
        ...

    @abstractmethod
    def unload(self, handle: ModelHandle) -> None:
        """Unload a model and free resources.

        Args:
            handle: Model handle to unload.
        """
        ...
