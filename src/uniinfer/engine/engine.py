"""Main Engine class — the primary user-facing interface for UniInfer."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Generator, Optional

from uniinfer.backends.interface import ExecutionBackend, GenerationResult, StreamChunk
from uniinfer.backends.registry import detect_backend, get_backend
from uniinfer.config.engine_config import EngineConfig
from uniinfer.engine.diagnostics import InferenceMetrics, SessionDiagnostics
from uniinfer.engine.fallback import FallbackResult, try_with_fallback
from uniinfer.hal.discovery import devices as discover_devices
from uniinfer.hal.discovery import select_best_device
from uniinfer.hal.interface import DeviceInfo
from uniinfer.models.aliases import ModelAlias, get_alias_info, resolve_alias
from uniinfer.models.converter import select_quantization_for_device
from uniinfer.models.fitting import FitReport, check_model_fit
from uniinfer.models.quantization import select_quantization
from uniinfer.models.registry import download_model, get_cache_path, get_cached_path, is_cached

logger = logging.getLogger(__name__)


class Engine:
    """UniInfer inference engine.

    The main entry point for loading models and running inference.
    Handles hardware detection, model downloading, quantization selection,
    and backend management.

    Usage:
        engine = Engine(model="meta-llama/Llama-3.1-8B-Instruct")
        result = engine.generate("What is gravity?")
        print(result.text)

        for chunk in engine.stream("Write a poem"):
            print(chunk.text, end="", flush=True)
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        quantization: str = "auto",
        max_tokens: int = 2048,
        context_length: int = 4096,
        cache_dir: str = "",
        **kwargs: Any,
    ) -> None:
        """Initialize the engine.

        Args:
            model: HuggingFace model ID or local path to a GGUF file.
            device: Device to use. "auto" for automatic selection.
            quantization: Quantization level. "auto" for automatic selection.
            max_tokens: Default maximum tokens for generation.
            context_length: Context window size.
            cache_dir: Custom cache directory.
            **kwargs: Additional backend options.
        """
        self._config = EngineConfig(
            model=model,
            device=device,
            quantization=quantization,
            max_tokens=max_tokens,
            context_length=context_length,
            cache_dir=cache_dir,
        )
        self._kwargs = kwargs
        self._device_info: Optional[DeviceInfo] = None
        self._backend: Optional[ExecutionBackend] = None
        self._handle: Optional[Any] = None
        self._resolved_quantization: str = "q4_k_m"
        self._model_path: Optional[Path] = None
        self._loaded = False
        self._alias_info: Optional[ModelAlias] = None
        self._fit_report: Optional[FitReport] = None
        self._fallback_result: Optional[FallbackResult] = None
        self._available_devices: list[DeviceInfo] = []
        self._diagnostics = SessionDiagnostics()

        self._setup()

    def _setup(self) -> None:
        """Run the full initialization pipeline."""
        # 1. Discover hardware
        logger.info("Discovering hardware...")
        self._available_devices = discover_devices()
        self._device_info = select_best_device(
            preferred=self._config.device,
            available=self._available_devices,
        )
        logger.info(
            "Selected device: %s (%s, %.1f GB free)",
            self._device_info.name,
            self._device_info.device_string,
            self._device_info.free_memory_gb,
        )

        # 2. Select quantization
        self._resolved_quantization = select_quantization_for_device(
            self._device_info,
            self._config.quantization,
        )

        # Check if context length should be reduced
        if self._config.quantization == "auto":
            recommendation = select_quantization(self._device_info)
            if recommendation.reduce_context and recommendation.suggested_context_length > 0:
                self._config.context_length = min(
                    self._config.context_length,
                    recommendation.suggested_context_length,
                )
                logger.info(
                    "Reduced context length to %d due to memory constraints",
                    self._config.context_length,
                )

        # 3. Run pre-load fit check if we have alias metadata
        self._alias_info = get_alias_info(self._config.model)
        if self._alias_info and self._device_info:
            from uniinfer.models.fitting import estimate_model_size_gb

            est_size = estimate_model_size_gb(
                self._alias_info.param_count_billions,
                self._resolved_quantization,
            )
            self._fit_report = check_model_fit(
                device=self._device_info,
                model_size_gb=est_size,
                context_length=self._config.context_length,
                quantization=self._resolved_quantization,
                param_count_billions=self._alias_info.param_count_billions,
            )
            if self._fit_report.warnings:
                for w in self._fit_report.warnings:
                    logger.warning("Fit check: %s", w)

        # 4. Resolve model path
        self._model_path = self._resolve_model()

        # 5. Post-download fit check (use actual file/directory size)
        if self._model_path and self._device_info:
            actual_size_gb = 0.0
            if self._model_path.is_file():
                actual_size_gb = self._model_path.stat().st_size / (1024**3)
            elif self._model_path.is_dir():
                # Sum all model files in directory (ONNX/SafeTensors)
                total_bytes = sum(
                    f.stat().st_size for f in self._model_path.rglob("*")
                    if f.is_file() and f.suffix.lower() in (
                        ".onnx", ".safetensors", ".bin", ".pt",
                    )
                )
                actual_size_gb = total_bytes / (1024**3)

            if actual_size_gb > 0:
                param_count = (
                    self._alias_info.param_count_billions if self._alias_info else 0.0
                )
                self._fit_report = check_model_fit(
                    device=self._device_info,
                    model_size_gb=actual_size_gb,
                    context_length=self._config.context_length,
                    quantization=self._resolved_quantization,
                    param_count_billions=param_count,
                )
                if self._fit_report.warnings:
                    for w in self._fit_report.warnings:
                        logger.warning("Fit check (actual): %s", w)

        # 6. Load model via backend with fallback chain
        self._load_model_with_fallback()

    def _resolve_model(self) -> Path:
        """Resolve the model to a local file path.

        Checks:
        1. Is it a model alias? Resolve to full repo ID.
        2. Is it a local file path?
        3. Is it cached?
        4. Download from HuggingFace.
        """
        # Resolve alias first (may already be set from _setup fit check)
        if self._alias_info is None:
            self._alias_info = get_alias_info(self._config.model)
        if self._alias_info:
            logger.info(
                "Resolved alias '%s' → %s (%s, %.1fB params)",
                self._config.model,
                self._alias_info.repo_id,
                self._alias_info.display_name,
                self._alias_info.param_count_billions,
            )
            self._config.model = self._alias_info.repo_id

        model = self._config.model

        # Check if it's a local file
        local_path = Path(model)
        if local_path.exists() and local_path.is_file():
            supported = {".gguf", ".onnx", ".safetensors"}
            if local_path.suffix.lower() in supported:
                logger.info("Using local model file: %s", local_path)
                return local_path
            else:
                raise RuntimeError(
                    f"Unsupported model format: '{local_path.suffix}'. "
                    f"Supported formats: {', '.join(sorted(supported))}"
                )

        # Check if it's a local directory (e.g., SafeTensors/HuggingFace model)
        if local_path.exists() and local_path.is_dir():
            logger.info("Using local model directory: %s", local_path)
            return local_path

        # Check cache (all formats: GGUF, ONNX, SafeTensors)
        cache_dir = self._config.cache_dir if self._config.cache_dir else None
        cached = get_cached_path(model, self._resolved_quantization, cache_dir)
        if cached is not None:
            logger.info("Using cached model: %s", cached)
            return cached

        # Download from HuggingFace
        logger.info("Downloading model '%s' from HuggingFace...", model)
        param_count = (
            self._alias_info.param_count_billions if self._alias_info else 0.0
        )
        return download_model(
            model_id=model,
            quantization=self._resolved_quantization,
            cache_dir=cache_dir,
            device=self._device_info,
            param_count_billions=param_count,
        )

    def _load_model_with_fallback(self) -> None:
        """Load the model with automatic fallback to alternative devices."""
        if self._model_path is None or self._device_info is None:
            raise RuntimeError("Model path or device info not resolved")

        def _try_load(device: DeviceInfo) -> Any:
            """Attempt to load the model on a specific device."""
            backend_name = detect_backend(str(self._model_path))
            backend = get_backend(backend_name, device.device_type)

            n_gpu_layers = self._config.n_gpu_layers
            n_threads = self._config.n_threads

            handle = backend.load_model(
                model_path=str(self._model_path),
                n_gpu_layers=n_gpu_layers if n_gpu_layers is not None else -1,
                n_ctx=self._config.context_length,
                n_threads=n_threads,
                **self._kwargs,
            )
            return backend, handle

        load_start = time.perf_counter()
        result_tuple, final_device, fallback_result = try_with_fallback(
            preferred=self._device_info,
            available=self._available_devices,
            load_fn=_try_load,
        )
        self._diagnostics.model_load_time = time.perf_counter() - load_start

        self._backend, self._handle = result_tuple
        self._device_info = final_device
        self._fallback_result = fallback_result
        self._loaded = True

        logger.info("Model loaded in %.2fs", self._diagnostics.model_load_time)
        if fallback_result.fell_back:
            logger.warning("Device fallback occurred: %s", fallback_result.summary)

    def _load_model(self) -> None:
        """Load the model into the backend (no fallback)."""
        if self._model_path is None or self._device_info is None:
            raise RuntimeError("Model path or device info not resolved")

        backend_name = detect_backend(str(self._model_path))
        self._backend = get_backend(backend_name, self._device_info.device_type)

        n_gpu_layers = self._config.n_gpu_layers
        n_threads = self._config.n_threads

        self._handle = self._backend.load_model(
            model_path=str(self._model_path),
            n_gpu_layers=n_gpu_layers if n_gpu_layers is not None else -1,
            n_ctx=self._config.context_length,
            n_threads=n_threads,
            **self._kwargs,
        )

        self._loaded = True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text from a prompt (blocking).

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Returns:
            GenerationResult with the completed text.
        """
        self._ensure_loaded()
        metrics = InferenceMetrics(start_time=time.perf_counter(), method="generate")
        result = self._backend.generate(  # type: ignore[union-attr]
            handle=self._handle,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        metrics.end_time = time.perf_counter()
        metrics.prompt_tokens = result.prompt_tokens
        metrics.completion_tokens = result.completion_tokens
        self._diagnostics.record(metrics)
        return result

    def stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream tokens from a prompt.

        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Yields:
            StreamChunk with each generated token.
        """
        self._ensure_loaded()
        metrics = InferenceMetrics(start_time=time.perf_counter(), method="stream")
        token_count = 0
        for chunk in self._backend.stream(  # type: ignore[union-attr]
            handle=self._handle,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            token_count += 1
            yield chunk
        metrics.end_time = time.perf_counter()
        metrics.completion_tokens = token_count
        self._diagnostics.record(metrics)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate a chat response from a list of messages.

        Uses the model's built-in chat template from GGUF metadata
        for correct formatting (Mistral, Llama, ChatML, etc.).

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Returns:
            GenerationResult with the assistant's response.
        """
        self._ensure_loaded()
        metrics = InferenceMetrics(start_time=time.perf_counter(), method="chat")
        result = self._backend.chat(  # type: ignore[union-attr]
            handle=self._handle,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        metrics.end_time = time.perf_counter()
        metrics.prompt_tokens = result.prompt_tokens
        metrics.completion_tokens = result.completion_tokens
        self._diagnostics.record(metrics)
        return result

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat response from a list of messages.

        Uses the model's built-in chat template from GGUF metadata.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Stop sequences.

        Yields:
            StreamChunk with each generated token.
        """
        self._ensure_loaded()
        metrics = InferenceMetrics(start_time=time.perf_counter(), method="chat_stream")
        token_count = 0
        for chunk in self._backend.chat_stream(  # type: ignore[union-attr]
            handle=self._handle,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        ):
            token_count += 1
            yield chunk
        metrics.end_time = time.perf_counter()
        metrics.completion_tokens = token_count
        self._diagnostics.record(metrics)

    @property
    def available_devices(self) -> list[DeviceInfo]:
        """Return the list of available hardware devices."""
        return list(self._available_devices)

    def info(self) -> dict[str, Any]:
        """Return information about the loaded model and device.

        Returns:
            Dict with model, device, quantization, and backend info.
        """
        result: dict[str, Any] = {
            "model": self._config.model,
            "device": self._device_info.device_string if self._device_info else "unknown",
            "device_name": self._device_info.name if self._device_info else "unknown",
            "quantization": self._resolved_quantization,
            "context_length": self._config.context_length,
            "backend": self._backend.name if self._backend else "none",
            "model_path": str(self._model_path) if self._model_path else "none",
            "loaded": self._loaded,
        }

        if self._device_info:
            result["device_memory_total_gb"] = round(self._device_info.total_memory_gb, 2)
            result["device_memory_free_gb"] = round(self._device_info.free_memory_gb, 2)

        if self._fit_report:
            result["fit"] = {
                "fits": self._fit_report.fits,
                "model_size_gb": self._fit_report.model_size_gb,
                "headroom_gb": self._fit_report.headroom_gb,
                "warnings": self._fit_report.warnings,
            }

        if self._fallback_result:
            result["fallback"] = {
                "fell_back": self._fallback_result.fell_back,
                "summary": self._fallback_result.summary,
                "events": [
                    {
                        "from": e.from_device,
                        "to": e.to_device,
                        "reason": e.reason,
                        "success": e.success,
                    }
                    for e in self._fallback_result.events
                ],
            }

        result["diagnostics"] = self._diagnostics.to_dict()

        return result

    def close(self) -> None:
        """Unload the model and free resources."""
        if self._backend and self._handle:
            self._backend.unload(self._handle)
        self._handle = None
        self._loaded = False
        logger.info("Engine closed")

    def _ensure_loaded(self) -> None:
        if not self._loaded or self._backend is None or self._handle is None:
            raise RuntimeError(
                "Model is not loaded. Make sure Engine initialization completed successfully."
            )

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
