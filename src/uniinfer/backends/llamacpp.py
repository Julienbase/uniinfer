"""llama.cpp backend via llama-cpp-python."""

from __future__ import annotations

import logging
from typing import Any, Generator, Optional

from uniinfer.backends.interface import (
    ExecutionBackend,
    GenerationResult,
    ModelHandle,
    StreamChunk,
)
from uniinfer.hal.interface import DeviceType

logger = logging.getLogger(__name__)


def _gpu_layers_for_device(device_type: DeviceType, n_gpu_layers: Optional[int] = None) -> int:
    """Determine the number of GPU layers based on device type.

    Args:
        device_type: The target device type.
        n_gpu_layers: User override. None = auto.

    Returns:
        Number of GPU layers to offload. -1 = all layers.
    """
    if n_gpu_layers is not None:
        return n_gpu_layers

    device_to_layers = {
        DeviceType.CUDA: -1,    # All layers on GPU
        DeviceType.ROCM: -1,    # All layers on GPU (llama-cpp-python with ROCm)
        DeviceType.VULKAN: -1,  # All layers on GPU (llama-cpp-python with Vulkan)
        DeviceType.CPU: 0,      # No GPU layers
    }
    return device_to_layers.get(device_type, 0)


class LlamaCppBackend(ExecutionBackend):
    """Execution backend wrapping llama-cpp-python.

    Supports CUDA, ROCm, Vulkan, and CPU through llama.cpp's
    built-in backend selection.
    """

    def __init__(self, device_type: DeviceType = DeviceType.CPU) -> None:
        self._device_type = device_type

    @property
    def name(self) -> str:
        return "llama.cpp"

    def load_model(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelHandle:
        """Load a GGUF model using llama-cpp-python.

        Args:
            model_path: Path to the GGUF file.
            n_gpu_layers: Number of layers to offload. -1 = all.
            n_ctx: Context window size.
            n_threads: CPU threads. None = auto.
            **kwargs: Additional llama-cpp-python parameters.

        Returns:
            ModelHandle wrapping the Llama instance.
        """
        try:
            from llama_cpp import Llama  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is required but not installed.\n"
                "Install it with: pip install llama-cpp-python\n"
                "For GPU support, see: https://github.com/abetlen/llama-cpp-python#installation"
            ) from exc

        effective_gpu_layers = _gpu_layers_for_device(self._device_type, n_gpu_layers)

        llama_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_gpu_layers": effective_gpu_layers,
            "n_ctx": n_ctx,
            "verbose": kwargs.get("verbose", False),
        }

        if n_threads is not None:
            llama_kwargs["n_threads"] = n_threads

        # Pass through any extra kwargs
        for key in ("seed", "n_batch", "rope_freq_base", "rope_freq_scale"):
            if key in kwargs:
                llama_kwargs[key] = kwargs[key]

        logger.info(
            "Loading model: %s (gpu_layers=%d, n_ctx=%d, device=%s)",
            model_path,
            effective_gpu_layers,
            n_ctx,
            self._device_type.value,
        )

        try:
            model = Llama(**llama_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{model_path}' with llama.cpp: {exc}\n"
                f"Device: {self._device_type.value}, GPU layers: {effective_gpu_layers}"
            ) from exc

        logger.info("Model loaded successfully")

        return ModelHandle(
            backend_name=self.name,
            model_path=model_path,
            internal=model,
        )

    def generate(
        self,
        handle: ModelHandle,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text using llama-cpp-python's create_completion.

        Args:
            handle: Model handle from load_model.
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Returns:
            GenerationResult with completed text and token counts.
        """
        model = handle.internal
        if model is None:
            raise RuntimeError("Model handle has no loaded model")

        try:
            result = model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Generation failed: {exc}") from exc

        # Extract results from llama-cpp-python response
        text = result["choices"][0]["text"]
        usage = result.get("usage", {})

        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def stream(
        self,
        handle: ModelHandle,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream tokens using llama-cpp-python's streaming completion.

        Args:
            handle: Model handle from load_model.
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Yields:
            StreamChunk for each generated token.
        """
        model = handle.internal
        if model is None:
            raise RuntimeError("Model handle has no loaded model")

        try:
            stream_iter = model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                stream=True,
            )

            for chunk in stream_iter:
                choice = chunk["choices"][0]
                text = choice.get("text", "")
                finish_reason = choice.get("finish_reason")
                finished = finish_reason is not None

                yield StreamChunk(text=text, finished=finished)

        except Exception as exc:
            raise RuntimeError(f"Streaming generation failed: {exc}") from exc

    def chat(
        self,
        handle: ModelHandle,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate a chat response using the model's built-in chat template."""
        model = handle.internal
        if model is None:
            raise RuntimeError("Model handle has no loaded model")

        try:
            result = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Chat generation failed: {exc}") from exc

        text = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})

        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def chat_stream(
        self,
        handle: ModelHandle,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat response using the model's built-in chat template."""
        model = handle.internal
        if model is None:
            raise RuntimeError("Model handle has no loaded model")

        try:
            stream_iter = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                stream=True,
            )

            for chunk in stream_iter:
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content", "")
                finish_reason = chunk["choices"][0].get("finish_reason")
                finished = finish_reason is not None

                if text or finished:
                    yield StreamChunk(text=text, finished=finished)

        except Exception as exc:
            raise RuntimeError(f"Chat streaming failed: {exc}") from exc

    def unload(self, handle: ModelHandle) -> None:
        """Unload the model and free resources.

        Args:
            handle: Model handle to unload.
        """
        if handle.internal is not None:
            # llama-cpp-python cleans up on deletion
            try:
                del handle.internal
            except Exception as exc:
                logger.warning("Error unloading model: %s", exc)
            handle.internal = None
            logger.info("Model unloaded: %s", handle.model_path)
