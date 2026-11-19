"""Model access through the official Anthropic SDK.

Two providers share one code path:
- "anthropic": the Claude API. Credentials come from the environment
  (ANTHROPIC_API_KEY); nothing is read from files.
- "ollama": a local Ollama server, which serves the same Messages API shape at
  http://localhost:11434. No key leaves the machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-haiku-5-5"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"

# USD per million tokens (input, output). Haiku 5.5 rates apply to prompts up to 100K tokens.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    calls: int = 0

    def add(self, raw: Any) -> None:
        self.calls += 1
        self.input_tokens += int(getattr(raw, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(raw, "output_tokens", 0) or 0)
        self.cache_creation_input_tokens += int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
        self.cache_read_input_tokens += int(getattr(raw, "cache_read_input_tokens", 0) or 0)

    def merge(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.calls += other.calls

    def cost_usd(self, model: str) -> float:
        if model not in PRICES:
            return 0.0
        price_in, price_out = PRICES[model]
        return (
            self.input_tokens * price_in
            + self.cache_creation_input_tokens * price_in * CACHE_WRITE_MULTIPLIER
            + self.cache_read_input_tokens * price_in * CACHE_READ_MULTIPLIER
            + self.output_tokens * price_out
        ) / 1_000_000

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "calls": self.calls,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int] | None) -> Usage:
        return cls(**data) if data else cls()


@dataclass
class ModelConfig:
    provider: str = "anthropic"
    model: str = DEFAULT_MODEL
    effort: str | None = "low"
    max_tokens: int = 8000
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_claude(self) -> bool:
        return self.provider == "anthropic"

    @classmethod
    def from_env(cls) -> ModelConfig:
        provider = os.environ.get("INVESTIGATOR_PROVIDER", "anthropic")
        if provider == "ollama":
            return cls(provider="ollama", model=os.environ.get("INVESTIGATOR_MODEL", DEFAULT_OLLAMA_MODEL), effort=None)
        return cls(
            provider="anthropic",
            model=os.environ.get("INVESTIGATOR_MODEL", DEFAULT_MODEL),
            effort=os.environ.get("INVESTIGATOR_EFFORT", "low") or None,
        )


def make_client(config: ModelConfig) -> anthropic.Anthropic:
    if config.provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return anthropic.Anthropic(base_url=base_url, api_key="ollama", timeout=600.0, max_retries=0)
    if config.provider != "anthropic":
        raise ValueError(f"unknown provider {config.provider!r}")
    return anthropic.Anthropic(timeout=120.0, max_retries=3)
