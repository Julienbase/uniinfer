"""Transformers backend — wraps HuggingFace transformers for SafeTensors/HF models."""

from __future__ import annotations

import logging
from threading import Thread
from typing import Any, Generator, Optional

from uniinfer.backends.interface import (
    ExecutionBackend,
    GenerationResult,
    ModelHandle,
    StreamChunk,
)
from uniinfer.hal.interface import DeviceType

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class TransformersBackend(ExecutionBackend):
    """Execution backend using HuggingFace transformers.

    Supports SafeTensors and other HuggingFace model formats.
    Requires: pip install torch transformers
    """

    def __init__(self, device_type: DeviceType) -> None:
        self._device_type = device_type

    @property
    def name(self) -> str:
        return "transformers"

    def load_model(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelHandle:
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers backend requires 'torch' and 'transformers'. "
                "Install with: pip install torch transformers"
            )

        device_map = self._get_device_map()
        dtype = self._get_dtype()

        logger.info(
            "Loading model from '%s' with transformers (device_map=%s, dtype=%s)",
            model_path,
            device_map,
            dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=False,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=False,
        )

        return ModelHandle(
            backend_name=self.name,
            model_path=model_path,
            internal={
                "model": model,
                "tokenizer": tokenizer,
                "n_ctx": n_ctx,
            },
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
        model = handle.internal["model"]
        tokenizer = handle.internal["tokenizer"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        gen_kwargs = self._build_gen_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        # Extract only generated tokens (exclude prompt)
        new_tokens = output_ids[0][prompt_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completion_tokens = len(new_tokens)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
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
        model = handle.internal["model"]
        tokenizer = handle.internal["tokenizer"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        gen_kwargs = self._build_gen_kwargs(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs["streamer"] = streamer
        thread = Thread(
            target=lambda: model.generate(**inputs, **gen_kwargs),
            daemon=True,
        )
        thread.start()

        for text in streamer:
            if text:
                yield StreamChunk(text=text, finished=False)

        yield StreamChunk(text="", finished=True)
        thread.join()

    def chat(
        self,
        handle: ModelHandle,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        prompt = self._format_chat(handle, messages)
        return self.generate(
            handle=handle,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
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
        prompt = self._format_chat(handle, messages)
        yield from self.stream(
            handle=handle,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

    def unload(self, handle: ModelHandle) -> None:
        if handle.internal:
            model = handle.internal.get("model")
            if model is not None:
                del model
            handle.internal.clear()
            if _TRANSFORMERS_AVAILABLE:
                torch.cuda.empty_cache()
        logger.info("Transformers model unloaded")

    def _get_device_map(self) -> str:
        if self._device_type in (DeviceType.CUDA, DeviceType.ROCM):
            return "auto"
        return "cpu"

    def _get_dtype(self) -> Any:
        if not _TRANSFORMERS_AVAILABLE:
            return None
        if self._device_type in (DeviceType.CUDA, DeviceType.ROCM):
            return torch.float16
        return torch.float32

    def _build_gen_kwargs(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
        }
        if temperature <= 0.01:
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        return kwargs

    def _format_chat(
        self,
        handle: ModelHandle,
        messages: list[dict[str, str]],
    ) -> str:
        tokenizer = handle.internal["tokenizer"]

        # Try using the tokenizer's built-in chat template
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                logger.debug("apply_chat_template failed, using fallback format")

        # Fallback: simple ChatML-style format
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)
