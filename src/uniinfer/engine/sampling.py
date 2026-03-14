"""Sampling parameters for text generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplingParams:
    """Parameters controlling token sampling during generation.

    Attributes:
        temperature: Controls randomness. 0 = greedy, higher = more random.
        top_p: Nucleus sampling — keep tokens with cumulative probability >= top_p.
        top_k: Keep only the top_k most probable tokens. 0 = disabled.
        max_tokens: Maximum number of tokens to generate.
        stop: List of stop sequences. Generation halts when any is produced.
        repeat_penalty: Penalty for repeating tokens. 1.0 = no penalty.
        presence_penalty: Penalize tokens based on presence in output so far.
        frequency_penalty: Penalize tokens based on frequency in output so far.
    """

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_tokens: int = 512
    stop: Optional[list[str]] = None
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be between 0 and 1, got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.repeat_penalty < 0:
            raise ValueError(f"repeat_penalty must be >= 0, got {self.repeat_penalty}")
        if self.stop is None:
            self.stop = []
