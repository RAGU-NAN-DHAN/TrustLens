"""Centralized, validated configuration for M1.

All configuration is read from environment variables so that no secrets or
machine-specific values are hardcoded. Call ``GeneratorConfig.from_env()`` /
``EmbeddingConfig.from_env()`` at the point of use (not at import time) so
that unit tests can construct configs without real environment variables
and without triggering network or model-download side effects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class GeneratorConfig:
    # Default writer/generator model. Groq no longer serves Llama 3.3 70B
    # (the model originally planned), so this defaults to a currently
    # available Groq-hosted model instead. This is NOT the same model as
    # M3's second verifier (Qwen, via Groq) — the generator and the
    # verifier must stay different models for the pipeline's adversarial-
    # independence design to hold. Override with GROQ_MODEL if Groq's
    # available model lineup changes again.
    api_key: str
    model: str = "openai/gpt-oss-120b"
    timeout_seconds: float = 30.0
    max_output_tokens: int = 1024
    temperature: float = 0.0
    max_retries: int = 2

    @staticmethod
    def from_env() -> GeneratorConfig:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Export it before calling the "
                "generator (see .env.example)."
            )
        model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip()
        if not model:
            raise ConfigError("GROQ_MODEL must not be empty if set.")

        timeout_raw = os.environ.get("GROQ_TIMEOUT_SECONDS", "30")
        try:
            timeout_seconds = float(timeout_raw)
        except ValueError as exc:
            raise ConfigError(
                f"GROQ_TIMEOUT_SECONDS must be numeric, got {timeout_raw!r}"
            ) from exc
        if timeout_seconds <= 0:
            raise ConfigError("GROQ_TIMEOUT_SECONDS must be > 0")

        max_tokens_raw = os.environ.get("GROQ_MAX_OUTPUT_TOKENS", "1024")
        try:
            max_output_tokens = int(max_tokens_raw)
        except ValueError as exc:
            raise ConfigError(
                f"GROQ_MAX_OUTPUT_TOKENS must be an integer, got {max_tokens_raw!r}"
            ) from exc
        if max_output_tokens <= 0:
            raise ConfigError("GROQ_MAX_OUTPUT_TOKENS must be > 0")

        max_retries_raw = os.environ.get("GROQ_MAX_RETRIES", "2")
        try:
            max_retries = int(max_retries_raw)
        except ValueError as exc:
            raise ConfigError(
                f"GROQ_MAX_RETRIES must be an integer, got {max_retries_raw!r}"
            ) from exc
        if max_retries < 0 or max_retries > 5:
            raise ConfigError("GROQ_MAX_RETRIES must be between 0 and 5 (no unbounded retries)")

        return GeneratorConfig(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
        )


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 32

    @staticmethod
    def from_env() -> EmbeddingConfig:
        model_name = os.environ.get(
            "TRUSTLENS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()
        if not model_name:
            raise ConfigError("TRUSTLENS_EMBEDDING_MODEL must not be empty if set.")
        device = os.environ.get("TRUSTLENS_EMBEDDING_DEVICE", "auto").strip() or "auto"
        batch_size_raw = os.environ.get("TRUSTLENS_EMBEDDING_BATCH_SIZE", "32")
        try:
            batch_size = int(batch_size_raw)
        except ValueError as exc:
            raise ConfigError(
                f"TRUSTLENS_EMBEDDING_BATCH_SIZE must be an integer, got {batch_size_raw!r}"
            ) from exc
        if batch_size <= 0:
            raise ConfigError("TRUSTLENS_EMBEDDING_BATCH_SIZE must be > 0")
        return EmbeddingConfig(model_name=model_name, device=device, batch_size=batch_size)


@dataclass(frozen=True)
class SelfConsistencyConfig:
    sample_count: int = 3
    temperature: float = 0.9
    max_output_tokens: int = 512

    @staticmethod
    def from_env() -> SelfConsistencyConfig:
        count_raw = os.environ.get("TRUSTLENS_SC_SAMPLE_COUNT", "3")
        try:
            sample_count = int(count_raw)
        except ValueError as exc:
            raise ConfigError(
                f"TRUSTLENS_SC_SAMPLE_COUNT must be an integer, got {count_raw!r}"
            ) from exc
        if not (3 <= sample_count <= 5):
            raise ConfigError("TRUSTLENS_SC_SAMPLE_COUNT must be between 3 and 5")

        temp_raw = os.environ.get("TRUSTLENS_SC_TEMPERATURE", "0.9")
        try:
            temperature = float(temp_raw)
        except ValueError as exc:
            raise ConfigError(
                f"TRUSTLENS_SC_TEMPERATURE must be numeric, got {temp_raw!r}"
            ) from exc
        if not (0.0 <= temperature <= 2.0):
            raise ConfigError("TRUSTLENS_SC_TEMPERATURE must be between 0.0 and 2.0")

        tokens_raw = os.environ.get("TRUSTLENS_SC_MAX_OUTPUT_TOKENS", "512")
        try:
            max_output_tokens = int(tokens_raw)
        except ValueError as exc:
            raise ConfigError(
                f"TRUSTLENS_SC_MAX_OUTPUT_TOKENS must be an integer, got {tokens_raw!r}"
            ) from exc
        if max_output_tokens <= 0:
            raise ConfigError("TRUSTLENS_SC_MAX_OUTPUT_TOKENS must be > 0")

        return SelfConsistencyConfig(
            sample_count=sample_count,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
