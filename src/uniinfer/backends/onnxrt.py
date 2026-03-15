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
            # Filter out models that need external data but are missing it.
            # Small .onnx files (< 10MB) contain only the graph and need a
            # companion <name>.onnx_data file with the actual weights.
            def _is_usable(f: _Path) -> bool:
                if f.stat().st_size >= 10 * 1024 * 1024:
                    return True  # Self-contained model
                data_file = _Path(str(f) + "_data")
                return data_file.exists()

            usable = [f for f in onnx_files if _is_usable(f)]
            candidates = usable if usable else onnx_files
            # Prefer model.onnx, otherwise use the largest
            for f in candidates:
                if f.name == "model.onnx":
                    resolved_path = str(f)
                    break
            else:
                resolved_path = str(max(candidates, key=lambda p: p.stat().st_size))
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

        # Load model config (for EOS tokens, chat template, etc.)
        model_config = self._load_model_config(resolved_path)
        chat_template = self._load_chat_template(resolved_path)

        logger.info("ONNX model loaded successfully (providers=%s)", providers)

        return ModelHandle(
            backend_name=self.name,
            model_path=resolved_path,
            internal={
                "session": session,
                "tokenizer": tokenizer,
                "n_ctx": n_ctx,
                "model_config": model_config,
                "chat_template": chat_template,
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
            # Search the model directory and parent directories for tokenizer.json
            # (ONNX repos often have tokenizer at a higher level than the .onnx files)
            search_dir = Path(model_path).parent
            for _ in range(4):
                candidate = search_dir / "tokenizer.json"
                if candidate.exists():
                    tokenizer_path = str(candidate)
                    break
                parent = search_dir.parent
                if parent == search_dir:
                    break
                search_dir = parent

        # If not found locally, try to download from HuggingFace
        if tokenizer_path is None:
            tokenizer_path = self._download_tokenizer(model_path)

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

    def _download_tokenizer(self, model_path: str) -> Optional[str]:
        """Try to download tokenizer.json from HuggingFace for a cached model.

        Infers the repo ID from the cache directory name (e.g.
        'onnx-community--gemma-3-1b-it-ONNX' -> 'onnx-community/gemma-3-1b-it-ONNX').
        """
        from pathlib import Path

        # Walk up to find the model cache root (directory whose name contains '--')
        model_dir = Path(model_path).parent
        repo_id = None
        target_dir = None
        for _ in range(5):
            if "--" in model_dir.name:
                repo_id = model_dir.name.replace("--", "/", 1)
                target_dir = model_dir
                break
            parent = model_dir.parent
            if parent == model_dir:
                break
            model_dir = parent

        if repo_id is None or target_dir is None:
            return None

        try:
            from huggingface_hub import hf_hub_download

            logger.info("Downloading tokenizer for '%s'...", repo_id)
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                local_dir=str(target_dir),
            )
            logger.info("Tokenizer downloaded to: %s", local_path)
            return str(local_path)
        except Exception as exc:
            logger.warning("Failed to download tokenizer for '%s': %s", repo_id, exc)
            return None

    def _load_model_config(self, model_path: str) -> dict[str, Any]:
        """Load config.json from the model directory tree."""
        import json
        from pathlib import Path

        search_dir = Path(model_path).parent
        for _ in range(4):
            candidate = search_dir / "config.json"
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        return json.load(f)
                except Exception as exc:
                    logger.warning("Failed to load config.json: %s", exc)
                    return {}
            parent = search_dir.parent
            if parent == search_dir:
                break
            search_dir = parent
        return {}

    def _load_chat_template(self, model_path: str) -> Optional[str]:
        """Load chat_template.jinja from the model directory tree."""
        from pathlib import Path

        search_dir = Path(model_path).parent
        for _ in range(4):
            candidate = search_dir / "chat_template.jinja"
            if candidate.exists():
                try:
                    return candidate.read_text(encoding="utf-8")
                except Exception as exc:
                    logger.warning("Failed to load chat template: %s", exc)
                    return None
            parent = search_dir.parent
            if parent == search_dir:
                break
            search_dir = parent
        return None

    def _build_feeds(
        self,
        session: Any,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        past_kv: Optional[dict[str, np.ndarray]] = None,
    ) -> dict[str, Any]:
        """Build the input feed dict, including KV cache if the model requires it."""
        feeds: dict[str, Any] = {}
        inputs = {inp.name: inp for inp in session.get_inputs()}

        if "input_ids" in inputs:
            feeds["input_ids"] = input_ids
        if "attention_mask" in inputs:
            feeds["attention_mask"] = attention_mask

        # Populate past_key_values if the model expects them
        kv_inputs = [n for n in inputs if n.startswith("past_key_values")]
        if kv_inputs:
            if past_kv is not None:
                for name in kv_inputs:
                    feeds[name] = past_kv[name]
            else:
                # First pass: initialise with zero-length sequence dim
                for name in kv_inputs:
                    shape = inputs[name].shape  # e.g. [batch, heads, seq, head_dim]
                    # Replace symbolic dims with concrete values
                    concrete: list[int] = []
                    for i, d in enumerate(shape):
                        if isinstance(d, int):
                            concrete.append(d)
                        elif i == 0:
                            concrete.append(1)  # batch
                        elif i == 2:
                            concrete.append(0)  # past sequence length = 0
                        else:
                            # head_count or head_dim — read from model config if available
                            concrete.append(d if isinstance(d, int) else 0)
                    # For typical [batch, heads, past_seq, head_dim] shapes,
                    # we need the actual head count and head dim from the model.
                    # Use the output shapes to infer dimensions.
                    feeds[name] = np.zeros(concrete, dtype=np.float32)

            # Infer correct shapes from model outputs on first call
            if past_kv is None and feeds.get(kv_inputs[0]) is not None:
                # Check if zero-dim shapes have unresolved symbolic dims
                # by looking at output shapes for concrete values
                out_kv = [o for o in session.get_outputs() if o.name.startswith("present")]
                if out_kv:
                    for out in out_kv:
                        past_name = out.name.replace("present", "past_key_values")
                        if past_name in feeds:
                            shape = list(out.shape)
                            concrete = []
                            for i, d in enumerate(shape):
                                if isinstance(d, int):
                                    concrete.append(d)
                                elif i == 0:
                                    concrete.append(1)
                                elif i == 2:
                                    concrete.append(0)
                                else:
                                    concrete.append(1)  # placeholder
                            feeds[past_name] = np.zeros(concrete, dtype=np.float32)

        return feeds

    def _extract_kv_cache(
        self, session: Any, outputs: list[np.ndarray]
    ) -> Optional[dict[str, np.ndarray]]:
        """Extract updated KV cache from model outputs."""
        output_names = [o.name for o in session.get_outputs()]
        kv_cache: dict[str, np.ndarray] = {}
        for i, name in enumerate(output_names):
            if name.startswith("present"):
                past_name = name.replace("present", "past_key_values")
                kv_cache[past_name] = outputs[i]
        return kv_cache if kv_cache else None

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
        past_kv: Optional[dict[str, np.ndarray]] = None
        has_kv_cache = any(
            inp.name.startswith("past_key_values") for inp in session.get_inputs()
        )

        # Gather EOS token IDs from model config
        config = (state.get("model_config") or {})
        eos_ids = _get_eos_token_ids(config)

        for _ in range(max_tokens):
            # With KV cache: only feed new token(s); without: feed full sequence
            if has_kv_cache and past_kv is not None:
                cur_ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
                seq_len = prompt_token_count + len(generated_tokens)
                cur_mask = np.ones((1, seq_len), dtype=np.int64)
            else:
                cur_ids = np.array([input_ids + generated_tokens], dtype=np.int64)
                cur_mask = np.ones_like(cur_ids, dtype=np.int64)

            feeds = self._build_feeds(session, cur_ids, cur_mask, past_kv)
            outputs = session.run(None, feeds)
            logits = outputs[0]  # Shape: [batch, seq_len, vocab_size]

            if has_kv_cache:
                past_kv = self._extract_kv_cache(session, outputs)

            # Get logits for the last token
            next_logits = logits[0, -1, :]

            # Sample next token
            next_token = self._sample_token(next_logits, temperature, top_p)

            # Check EOS token IDs before appending
            if int(next_token) in eos_ids:
                break

            generated_tokens.append(int(next_token))

            # Check text-based stop sequences
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
        past_kv: Optional[dict[str, np.ndarray]] = None
        has_kv_cache = any(
            inp.name.startswith("past_key_values") for inp in session.get_inputs()
        )

        config = (state.get("model_config") or {})
        eos_ids = _get_eos_token_ids(config)

        for i in range(max_tokens):
            if has_kv_cache and past_kv is not None:
                cur_ids = np.array([[generated_tokens[-1]]], dtype=np.int64)
                seq_len = len(input_ids) + len(generated_tokens)
                cur_mask = np.ones((1, seq_len), dtype=np.int64)
            else:
                cur_ids = np.array([input_ids + generated_tokens], dtype=np.int64)
                cur_mask = np.ones_like(cur_ids, dtype=np.int64)

            feeds = self._build_feeds(session, cur_ids, cur_mask, past_kv)
            outputs = session.run(None, feeds)
            logits = outputs[0]

            if has_kv_cache:
                past_kv = self._extract_kv_cache(session, outputs)

            next_logits = logits[0, -1, :]
            next_token = self._sample_token(next_logits, temperature, top_p)

            # Check EOS token IDs before appending
            if int(next_token) in eos_ids:
                yield StreamChunk(text="", finished=True)
                return

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
        state = handle.internal or {}
        template = state.get("chat_template")
        config = state.get("model_config", {})
        prompt = _format_chat_prompt(messages, template, config)

        # Add EOS-based stop tokens
        eos_stops = _get_eos_stop_strings(state.get("tokenizer"), config)
        all_stops = list(stop or []) + eos_stops

        return self.generate(handle, prompt, max_tokens, temperature, top_p, all_stops)

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
        state = handle.internal or {}
        template = state.get("chat_template")
        config = state.get("model_config", {})
        prompt = _format_chat_prompt(messages, template, config)

        eos_stops = _get_eos_stop_strings(state.get("tokenizer"), config)
        all_stops = list(stop or []) + eos_stops

        yield from self.stream(handle, prompt, max_tokens, temperature, top_p, all_stops)

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


def _format_chat_prompt(
    messages: list[dict[str, str]],
    template: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> str:
    """Format chat messages using the model's Jinja chat template if available."""
    if template is not None:
        try:
            from jinja2 import Environment, BaseLoader

            env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
            # Add raise_exception helper used by some templates
            env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(
                ValueError(msg)
            )
            tmpl = env.from_string(template)

            bos_token = ""
            if config:
                bos_id = config.get("bos_token_id")
                if bos_id is not None:
                    bos_token = "<bos>"

            result = tmpl.render(
                messages=messages,
                bos_token=bos_token,
                eos_token="<eos>",
                add_generation_prompt=True,
            )
            return result
        except ImportError:
            logger.warning("jinja2 not available, falling back to basic chat format")
        except Exception as exc:
            logger.warning("Failed to render chat template: %s", exc)

    # Fallback: basic format
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        else:
            parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def _get_eos_token_ids(config: dict[str, Any]) -> set[int]:
    """Extract EOS token IDs from model config."""
    eos = config.get("eos_token_id", [])
    if isinstance(eos, int):
        return {eos}
    if isinstance(eos, list):
        return set(eos)
    return set()


def _get_eos_stop_strings(tokenizer: Any, config: dict[str, Any]) -> list[str]:
    """Get text representations of EOS tokens for stop sequence matching."""
    stops = []
    # Common end-of-turn markers
    stops.append("<end_of_turn>")
    stops.append("<eos>")
    return stops


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
