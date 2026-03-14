"""UniInfer — Hardware-agnostic AI inference runtime.

Usage:
    import uniinfer

    # One-liner chat
    response = uniinfer.chat("mistral-7b", "What is gravity?")

    # One-liner generation
    text = uniinfer.generate("meta-llama/Llama-3.1-8B-Instruct", "What is gravity?")

    # Engine for repeated use
    engine = uniinfer.Engine(model="mistral-7b")
    result = engine.generate("Explain quantum computing.")
    print(result.text)

    # Streaming
    for chunk in engine.stream("Write a haiku"):
        print(chunk.text, end="", flush=True)

    # Hardware discovery
    print(uniinfer.devices())
"""

from __future__ import annotations

from typing import Any, Generator

__version__ = "0.5.0"


def _lazy_engine() -> type:
    from uniinfer.engine.engine import Engine
    return Engine


def devices() -> list:
    """Discover all available hardware devices.

    Returns:
        List of DeviceInfo objects for all discovered devices.
    """
    from uniinfer.hal.discovery import devices as _devices
    return _devices()


def generate(model: str, prompt: str, **kwargs: Any) -> str:
    """One-liner text generation.

    Creates a temporary Engine, generates text, and returns the result.

    Args:
        model: HuggingFace model ID or local GGUF path.
        prompt: Input prompt.
        **kwargs: Additional arguments passed to Engine and generate().

    Returns:
        Generated text string.
    """
    from uniinfer.engine.engine import Engine

    # Split kwargs between Engine init and generate call
    engine_keys = {"device", "quantization", "max_tokens", "context_length", "cache_dir"}
    engine_kwargs = {k: v for k, v in kwargs.items() if k in engine_keys}
    gen_kwargs = {k: v for k, v in kwargs.items() if k not in engine_keys}

    engine = Engine(model=model, **engine_kwargs)
    try:
        result = engine.generate(prompt=prompt, **gen_kwargs)
        return result.text
    finally:
        engine.close()


def stream(model: str, prompt: str, **kwargs: Any) -> Generator:
    """One-liner streaming generation.

    Creates a temporary Engine and yields StreamChunk objects.

    Args:
        model: HuggingFace model ID, model alias, or local path.
        prompt: Input prompt.
        **kwargs: Additional arguments passed to Engine and stream().

    Yields:
        StreamChunk objects with .text and .finished attributes.
    """
    from uniinfer.engine.engine import Engine

    engine_keys = {"device", "quantization", "max_tokens", "context_length", "cache_dir"}
    engine_kwargs = {k: v for k, v in kwargs.items() if k in engine_keys}
    stream_kwargs = {k: v for k, v in kwargs.items() if k not in engine_keys}

    engine = Engine(model=model, **engine_kwargs)
    try:
        yield from engine.stream(prompt=prompt, **stream_kwargs)
    finally:
        engine.close()


def chat(model: str, message: str, **kwargs: Any) -> str:
    """One-liner chat. Returns assistant response text.

    Creates a temporary Engine, sends a chat message, and returns the response.

    Args:
        model: HuggingFace model ID, model alias (e.g. "mistral-7b"), or local path.
        message: User message string.
        **kwargs: Additional arguments. 'system' sets the system prompt.
            Engine args: device, quantization, max_tokens, context_length, cache_dir.
            Chat args: temperature, top_p, stop.

    Returns:
        Assistant response text string.
    """
    from uniinfer.engine.engine import Engine

    engine_keys = {"device", "quantization", "max_tokens", "context_length", "cache_dir"}
    engine_kwargs = {k: v for k, v in kwargs.items() if k in engine_keys}
    chat_kwargs = {k: v for k, v in kwargs.items() if k not in engine_keys and k != "system"}

    messages: list[dict[str, str]] = []
    if "system" in kwargs:
        messages.append({"role": "system", "content": kwargs["system"]})
    messages.append({"role": "user", "content": message})

    engine = Engine(model=model, **engine_kwargs)
    try:
        result = engine.chat(messages=messages, **chat_kwargs)
        return result.text
    finally:
        engine.close()


def chat_stream(model: str, message: str, **kwargs: Any) -> Generator:
    """One-liner streaming chat. Yields StreamChunk objects.

    Args:
        model: HuggingFace model ID, model alias, or local path.
        message: User message string.
        **kwargs: Additional arguments. 'system' sets the system prompt.

    Yields:
        StreamChunk objects with .text and .finished attributes.
    """
    from uniinfer.engine.engine import Engine

    engine_keys = {"device", "quantization", "max_tokens", "context_length", "cache_dir"}
    engine_kwargs = {k: v for k, v in kwargs.items() if k in engine_keys}
    chat_kwargs = {k: v for k, v in kwargs.items() if k not in engine_keys and k != "system"}

    messages: list[dict[str, str]] = []
    if "system" in kwargs:
        messages.append({"role": "system", "content": kwargs["system"]})
    messages.append({"role": "user", "content": message})

    engine = Engine(model=model, **engine_kwargs)
    try:
        yield from engine.chat_stream(messages=messages, **chat_kwargs)
    finally:
        engine.close()


# Lazy import for Engine to avoid heavy imports on `import uniinfer`
def __getattr__(name: str) -> Any:
    if name == "Engine":
        from uniinfer.engine.engine import Engine
        return Engine
    raise AttributeError(f"module 'uniinfer' has no attribute '{name}'")
