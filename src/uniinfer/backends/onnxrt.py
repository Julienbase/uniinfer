"""ONNX Runtime execution backend.

Supports CUDA, ROCm, and CPU execution providers for running
ONNX-format models. Implements autoregressive text generation
with configurable sampling.
"""

from __future__ import annotations

import logging
from typing import Any, Generator, Optional

import numpy as np

from uniinfer.backends.interface import (
    ExecutionBackend,
    GenerationResult,
    ModelHandle,
    StreamChunk,
)
from uniinfer.hal.interface import DeviceType

logger = logging.getLogger(__name__)

# Execution provider mapping per device type
_EP_MAP: dict[DeviceType, list[str]] = {
    DeviceType.CUDA: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    DeviceType.ROCM: ["ROCMExecutionProvider", "CPUExecutionProvider"],
    DeviceType.VULKAN: ["CPUExecutionProvider"],  # No native Vulkan EP in ORT
    DeviceType.CPU: ["CPUExecutionProvider"],
}


class OnnxRuntimeBackend(ExecutionBackend):
    """Execution backend wrapping ONNX Runtime.

    Supports ONNX-format models with automatic execution provider
    selection based on the target device type.
    """

    def __init__(self, device_type: DeviceType = DeviceType.CPU) -> None:
        self._device_type = device_type

    @property
    def name(self) -> str:
        return "onnxruntime"

    def load_model(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        **kwargs: Any,
    ) -> ModelHandle:
        """Load an ONNX model.

        Args:
            model_path: Path to the .onnx model file.
            n_gpu_layers: Ignored for ONNX Runtime (uses execution providers).
            n_ctx: Context window size (stored for generation loop).
            n_threads: CPU thread count for intra-op parallelism.
            **kwargs: Additional session options.
        """
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required but not installed.\n"
                "Install it with: pip install onnxruntime\n"
                "For GPU support: pip install onnxruntime-gpu"
            ) from exc

        providers = _EP_MAP.get(self._device_type, ["CPUExecutionProvider"])

        # Filter to only available providers
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        if n_threads is not None:
            session_options.intra_op_num_threads = n_threads

        # If given a directory, find the .onnx file inside
        from pathlib import Path as _Path
        resolved_path = model_path
        if _Path(model_path).is_dir():
            onnx_files = list(_Path(model_path).rglob("*.onnx"))
            if not onnx_files:
                raise RuntimeError(
                    f"No .onnx files found in directory '{model_path}'"
                )
            # Prefer model.onnx, otherwise use the largest
            for f in onnx_files:
                if f.name == "model.onnx":
                    resolved_path = str(f)
                    break
            else:
                resolved_path = str(max(onnx_files, key=lambda p: p.stat().st_size))
            logger.info("Resolved ONNX model in directory: %s", resolved_path)

        logger.info(
            "Loading ONNX model: %s (providers=%s, device=%s)",
            resolved_path,
            providers,
            self._device_type.value,
        )

        try:
            session = ort.InferenceSession(
                resolved_path,
                sess_options=session_options,
                providers=providers,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load ONNX model '{resolved_path}': {exc}\n"
                f"Providers attempted: {providers}"
            ) from exc

        # Try to load the tokenizer from the model directory
        tokenizer = self._load_tokenizer(resolved_path, **kwargs)

        logger.info("ONNX model loaded successfully (providers=%s)", providers)

        return ModelHandle(
            backend_name=self.name,
            model_path=resolved_path,
            internal={
                "session": session,
                "tokenizer": tokenizer,
                "n_ctx": n_ctx,
            },
        )

    def _load_tokenizer(self, model_path: str, **kwargs: Any) -> Any:
        """Attempt to load a tokenizer for the model.

        Looks for tokenizer files in the same directory as the ONNX model,
        or uses a tokenizer_path kwarg if provided.
        """
        from pathlib import Path

        tokenizer_path = kwargs.get("tokenizer_path")

        if tokenizer_path is None:
            model_dir = Path(model_path).parent
            # Look for tokenizer.json in the model directory
            candidate = model_dir / "tokenizer.json"
            if candidate.exists():
                tokenizer_path = str(candidate)

        if tokenizer_path is None:
            logger.warning(
                "No tokenizer found for ONNX model. "
                "Chat and generate methods require a tokenizer. "
                "Provide tokenizer_path kwarg or place tokenizer.json alongside the model."
            )
            return None

        try:
            from tokenizers import Tokenizer  # type: ignore[import-untyped]

            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            logger.info("Tokenizer loaded from: %s", tokenizer_path)
            return tokenizer
        except Exception as exc:
            logger.warning("Failed to load tokenizer from '%s': %s", tokenizer_path, exc)
            return None

    def generate(
        self,
        handle: ModelHandle,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text using autoregressive decoding with ONNX Runtime."""
        state = handle.internal
        if state is None:
            raise RuntimeError("Model handle has no loaded model")

        session = state["session"]
        tokenizer = state["tokenizer"]
        if tokenizer is None:
            raise RuntimeError(
                "ONNX Runtime backend requires a tokenizer for text generation. "
                "Place tokenizer.json alongside the ONNX model or pass tokenizer_path."
            )

        # Encode prompt
        encoding = tokenizer.encode(prompt)
        input_ids = list(encoding.ids)
        prompt_token_count = len(input_ids)

        # Autoregressive decode loop
        generated_tokens: list[int] = []
        stop_sequences = stop or []

        for _ in range(max_tokens):
            # Prepare input
            input_array = np.array([input_ids + generated_tokens], dtype=np.int64)
            attention_mask = np.ones_like(input_array, dtype=np.int64)

            feeds: dict[str, Any] = {}
            input_names = [inp.name for inp in session.get_inputs()]

            if "input_ids" in input_names:
                feeds["input_ids"] = input_array
            if "attention_mask" in input_names:
                feeds["attention_mask"] = attention_mask

            # Run forward pass
            outputs = session.run(None, feeds)
            logits = outputs[0]  # Shape: [batch, seq_len, vocab_size]

            # Get logits for the last token
            next_logits = logits[0, -1, :]

            # Sample next token
            next_token = self._sample_token(next_logits, temperature, top_p)
            generated_tokens.append(int(next_token))

            # Check EOS
            decoded_so_far = tokenizer.decode(generated_tokens)
            if _check_stop(decoded_so_far, stop_sequences):
                break

        output_text = tokenizer.decode(generated_tokens)
        # Trim stop sequences from output
        output_text = _trim_stop_sequences(output_text, stop_sequences)

        return GenerationResult(
            text=output_text,
            prompt_tokens=prompt_token_count,
            completion_tokens=len(generated_tokens),
            total_tokens=prompt_token_count + len(generated_tokens),
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
        """Stream tokens using autoregressive decoding."""
        state = handle.internal
        if state is None:
            raise RuntimeError("Model handle has no loaded model")

        session = state["session"]
        tokenizer = state["tokenizer"]
        if tokenizer is None:
            raise RuntimeError(
                "ONNX Runtime backend requires a tokenizer for text generation."
            )

        encoding = tokenizer.encode(prompt)
        input_ids = list(encoding.ids)
        generated_tokens: list[int] = []
        stop_sequences = stop or []
        prev_text = ""

        for i in range(max_tokens):
            input_array = np.array([input_ids + generated_tokens], dtype=np.int64)
            attention_mask = np.ones_like(input_array, dtype=np.int64)

            feeds: dict[str, Any] = {}
            input_names = [inp.name for inp in session.get_inputs()]
            if "input_ids" in input_names:
                feeds["input_ids"] = input_array
            if "attention_mask" in input_names:
                feeds["attention_mask"] = attention_mask

            outputs = session.run(None, feeds)
            logits = outputs[0]
            next_logits = logits[0, -1, :]

            next_token = self._sample_token(next_logits, temperature, top_p)
            generated_tokens.append(int(next_token))

            decoded = tokenizer.decode(generated_tokens)
            new_text = decoded[len(prev_text):]
            prev_text = decoded

            if _check_stop(decoded, stop_sequences):
                new_text = _trim_stop_sequences(new_text, stop_sequences)
                if new_text:
                    yield StreamChunk(text=new_text, finished=True)
                else:
                    yield StreamChunk(text="", finished=True)
                return

            if new_text:
                yield StreamChunk(text=new_text, finished=False)

        # Final chunk
        yield StreamChunk(text="", finished=True)

    def chat(
        self,
        handle: ModelHandle,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate a chat response by formatting messages into a prompt."""
        prompt = _format_chat_prompt(messages)
        return self.generate(handle, prompt, max_tokens, temperature, top_p, stop)

    def chat_stream(
        self,
        handle: ModelHandle,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat response."""
        prompt = _format_chat_prompt(messages)
        yield from self.stream(handle, prompt, max_tokens, temperature, top_p, stop)

    def unload(self, handle: ModelHandle) -> None:
        """Unload the model and free resources."""
        if handle.internal is not None:
            try:
                handle.internal["session"] = None
                handle.internal["tokenizer"] = None
            except Exception as exc:
                logger.warning("Error unloading ONNX model: %s", exc)
            handle.internal = None
            logger.info("ONNX model unloaded: %s", handle.model_path)

    @staticmethod
    def _sample_token(logits: Any, temperature: float, top_p: float) -> int:
        """Sample a token from logits using temperature and nucleus sampling."""
        if temperature <= 0:
            return int(np.argmax(logits))

        # Apply temperature
        logits = logits / temperature

        # Softmax
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits)

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative = np.cumsum(sorted_probs)

            # Find cutoff
            cutoff_idx = np.searchsorted(cumulative, top_p) + 1
            cutoff_idx = min(cutoff_idx, len(sorted_probs))

            # Zero out tokens below cutoff
            mask = np.zeros_like(probs)
            mask[sorted_indices[:cutoff_idx]] = 1.0
            probs = probs * mask
            probs = probs / np.sum(probs)

        return int(np.random.choice(len(probs), p=probs))


def _format_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Format chat messages into a simple prompt string.

    Uses a basic ChatML-style format. For production use, the model's
    actual chat template should be used instead.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _check_stop(text: str, stop_sequences: list[str]) -> bool:
    """Check if any stop sequence appears in the generated text."""
    return any(seq in text for seq in stop_sequences)


def _trim_stop_sequences(text: str, stop_sequences: list[str]) -> str:
    """Remove stop sequences from the end of generated text."""
    for seq in stop_sequences:
        idx = text.find(seq)
        if idx != -1:
            text = text[:idx]
    return text
