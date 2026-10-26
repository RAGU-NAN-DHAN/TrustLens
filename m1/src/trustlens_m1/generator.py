"""Phase 1 — Groq-based initial answer generation.

Owned by M1. Provides ``generate_answer`` (and the lower-level
``generate_answer_with_client`` used by self_consistency.py and tests) as the
single way the rest of the team gets an initial answer out of Groq / Llama
3.3 70B.

Design notes
------------
- The Groq client is created lazily per call (or injected for tests) so that
  importing this module never requires GROQ_API_KEY to be set.
- Retries are bounded (``GeneratorConfig.max_retries``, default 2) and only
  applied to retriable failures (timeouts, 429s, 5xxs) with capped backoff.
  Authentication failures and invalid input are never retried.
- No credential value is ever included in a log line or an exception
  message; see ``logging_utils.log_event`` for redaction.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .config import ConfigError, GeneratorConfig
from .errors import (
    AuthenticationError,
    GenerationError,
    GenerationTimeoutError,
    InvalidInputError,
    NetworkError,
    RateLimitError,
    UnexpectedResponseError,
)
from .logging_utils import get_logger, log_event

logger = get_logger("generator")


@dataclass(frozen=True)
class GenerationResult:
    """The data contract M2/M3 consume for a single generated answer.

    Attributes:
        question: The original, validated question text.
        answer: The generated answer text.
        model: The Groq model identifier that produced the answer.
        temperature: Sampling temperature used for this call.
        created_at: Unix timestamp (seconds) when the result was produced.
        duration_seconds: Wall-clock time for the request.
        prompt_tokens, completion_tokens, total_tokens: Token usage if the
            API reported it, else ``None``.
        request_id: Groq's request id if available, for debugging/support.
        attempt: 1-indexed attempt number that succeeded (>1 means retried).
    """

    question: str
    answer: str
    model: str
    temperature: float
    created_at: float
    duration_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    request_id: str | None = None
    attempt: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "model": self.model,
            "temperature": self.temperature,
            "created_at": self.created_at,
            "duration_seconds": self.duration_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_id": self.request_id,
            "attempt": self.attempt,
        }


class SupportsChatCompletion(Protocol):
    """Minimal interface M1 depends on, so tests can supply a fake client."""

    async def create_chat_completion(
        self,
        *,
        model: str,
        question: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> dict[str, Any]:
        ...


class GroqChatClient:
    """Thin async wrapper around the official ``groq`` Python client.

    Imports the ``groq`` package lazily inside ``__init__`` so that modules
    which only need ``GenerationResult`` (or test doubles) never need the
    dependency installed.
    """

    def __init__(self, api_key: str) -> None:
        try:
            from groq import AsyncGroq  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via tests with stub
            raise GenerationError(
                "The 'groq' package is not installed. Run `pip install groq`."
            ) from exc
        self._client = AsyncGroq(api_key=api_key)

    async def create_chat_completion(
        self,
        *,
        model: str,
        question: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> dict[str, Any]:
        from groq import (  # type: ignore[import-not-found]
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
        )
        from groq import (
            AuthenticationError as GroqAuthError,
        )
        from groq import (
            RateLimitError as GroqRateLimitError,
        )

        try:
            completion = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise GenerationTimeoutError(
                f"Groq request exceeded {timeout}s timeout"
            ) from exc
        except GroqAuthError as exc:
            raise AuthenticationError(
                "Groq rejected the API key (authentication failed)"
            ) from exc
        except GroqRateLimitError as exc:
            raise RateLimitError("Groq rate limit exceeded") from exc
        except APITimeoutError as exc:
            raise GenerationTimeoutError(f"Groq request timed out: {exc}") from exc
        except APIConnectionError as exc:
            raise NetworkError(f"Network error contacting Groq: {exc}") from exc
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status == 401 or status == 403:
                raise AuthenticationError("Groq rejected the API key") from exc
            if status == 429:
                raise RateLimitError("Groq rate limit exceeded") from exc
            if status is not None and 500 <= status < 600:
                raise NetworkError(f"Groq server error (status {status})") from exc
            raise UnexpectedResponseError(
                f"Groq returned an unexpected status: {status}"
            ) from exc

        try:
            choice = completion.choices[0]
            answer = choice.message.content
            if answer is None:
                raise UnexpectedResponseError("Groq response had no message content")
            usage = getattr(completion, "usage", None)
            return {
                "answer": answer,
                "model": getattr(completion, "model", model),
                "request_id": getattr(completion, "id", None),
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None)
                if usage
                else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            }
        except (IndexError, AttributeError) as exc:
            raise UnexpectedResponseError(
                f"Groq response did not match the expected schema: {exc}"
            ) from exc


def _validate_question(question: str) -> str:
    if question is None:
        raise InvalidInputError("question must not be None")
    if not isinstance(question, str):
        raise InvalidInputError(f"question must be a string, got {type(question).__name__}")
    stripped = question.strip()
    if not stripped:
        raise InvalidInputError("question must not be empty or whitespace-only")
    if len(stripped) > 8000:
        raise InvalidInputError("question exceeds the 8000-character limit")
    return stripped


_RETRIABLE = (RateLimitError, GenerationTimeoutError, NetworkError)


async def generate_answer(
    question: str,
    *,
    config: GeneratorConfig | None = None,
    client: SupportsChatCompletion | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> GenerationResult:
    """Generate the initial answer to ``question`` using Groq (default model:
    openai/gpt-oss-120b; configurable via GROQ_MODEL).

    Args:
        question: The user's question. Must be non-empty after stripping.
        config: Optional pre-built ``GeneratorConfig``. If omitted, is built
            from the environment via ``GeneratorConfig.from_env()`` (requires
            ``GROQ_API_KEY``).
        client: Optional injected client implementing
            ``SupportsChatCompletion`` — used by tests and by M2/M3 if they
            want to share a single underlying client instance. If omitted, a
            real ``GroqChatClient`` is constructed from ``config.api_key``.
        temperature: Overrides ``config.temperature`` for this call (used by
            self-consistency sampling to request higher-temperature samples).
        max_tokens: Overrides ``config.max_output_tokens`` for this call.

    Returns:
        A ``GenerationResult``.

    Raises:
        InvalidInputError: ``question`` is empty/whitespace/non-string/too long.
        trustlens_m1.config.ConfigError: Required env vars are missing/invalid.
        AuthenticationError, RateLimitError, GenerationTimeoutError,
        NetworkError, UnexpectedResponseError: see ``errors.py``.
    """
    validated_question = _validate_question(question)

    try:
        cfg = config or GeneratorConfig.from_env()
    except ConfigError:
        raise

    effective_temperature = cfg.temperature if temperature is None else temperature
    effective_max_tokens = cfg.max_output_tokens if max_tokens is None else max_tokens

    active_client: SupportsChatCompletion = client or GroqChatClient(cfg.api_key)

    log_event(
        logger,
        logging.INFO,
        "generation.start",
        model=cfg.model,
        temperature=effective_temperature,
        max_tokens=effective_max_tokens,
        question_length=len(validated_question),
    )

    last_error: GenerationError | None = None
    start = time.monotonic()
    for attempt in range(1, cfg.max_retries + 2):  # at least one attempt
        attempt_start = time.monotonic()
        try:
            raw = await active_client.create_chat_completion(
                model=cfg.model,
                question=validated_question,
                temperature=effective_temperature,
                max_tokens=effective_max_tokens,
                timeout=cfg.timeout_seconds,
            )
            duration = time.monotonic() - start
            log_event(
                logger,
                logging.INFO,
                "generation.success",
                model=raw.get("model", cfg.model),
                attempt=attempt,
                duration_seconds=duration,
                request_id=raw.get("request_id"),
            )
            return GenerationResult(
                question=validated_question,
                answer=raw["answer"],
                model=raw.get("model", cfg.model),
                temperature=effective_temperature,
                created_at=time.time(),
                duration_seconds=duration,
                prompt_tokens=raw.get("prompt_tokens"),
                completion_tokens=raw.get("completion_tokens"),
                total_tokens=raw.get("total_tokens"),
                request_id=raw.get("request_id"),
                attempt=attempt,
            )
        except AuthenticationError:
            log_event(logger, logging.ERROR, "generation.auth_error", attempt=attempt)
            raise
        except InvalidInputError:
            raise
        except _RETRIABLE as exc:
            last_error = exc
            log_event(
                logger,
                logging.WARNING,
                "generation.retriable_error",
                error_type=type(exc).__name__,
                attempt=attempt,
                duration_seconds=time.monotonic() - attempt_start,
            )
            if attempt <= cfg.max_retries:
                backoff = min(2 ** (attempt - 1), 8) * 0.1
                await asyncio.sleep(backoff)
                continue
            break
        except GenerationError as exc:
            log_event(
                logger,
                logging.ERROR,
                "generation.unexpected_error",
                error_type=type(exc).__name__,
                attempt=attempt,
            )
            raise

    assert last_error is not None
    log_event(
        logger,
        logging.ERROR,
        "generation.exhausted_retries",
        error_type=type(last_error).__name__,
        attempts=cfg.max_retries + 1,
    )
    raise last_error
