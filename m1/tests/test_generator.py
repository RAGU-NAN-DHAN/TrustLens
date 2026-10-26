from __future__ import annotations

import pytest

from tests.conftest import FakeGroqClient
from trustlens_m1.config import ConfigError, GeneratorConfig
from trustlens_m1.errors import (
    AuthenticationError,
    GenerationTimeoutError,
    InvalidInputError,
    NetworkError,
    RateLimitError,
)
from trustlens_m1.generator import generate_answer


@pytest.mark.asyncio
async def test_generate_answer_valid_response(generator_config):
    client = FakeGroqClient(answers="Paris is the capital of France.")
    result = await generate_answer("What is the capital of France?", config=generator_config, client=client)
    assert result.answer == "Paris is the capital of France."
    assert result.model == "fake-model"
    assert result.question == "What is the capital of France?"
    assert result.attempt == 1
    assert result.duration_seconds >= 0
    assert client.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_question", ["", "   ", "\n\t", None, 123])
async def test_generate_answer_rejects_invalid_input(generator_config, bad_question):
    client = FakeGroqClient()
    with pytest.raises(InvalidInputError):
        await generate_answer(bad_question, config=generator_config, client=client)
    assert client.call_count == 0


@pytest.mark.asyncio
async def test_generate_answer_missing_api_credentials(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = FakeGroqClient()
    with pytest.raises(ConfigError):
        await generate_answer("Hello?", client=client)
    assert client.call_count == 0


@pytest.mark.asyncio
async def test_generate_answer_auth_error_not_retried(generator_config):
    client = FakeGroqClient(fail_with=AuthenticationError, fail_times=99)
    with pytest.raises(AuthenticationError):
        await generate_answer("Hello?", config=generator_config, client=client)
    assert client.call_count == 1  # never retried


@pytest.mark.asyncio
async def test_generate_answer_rate_limit_retries_then_succeeds(generator_config):
    client = FakeGroqClient(fail_with=RateLimitError, fail_times=1)
    result = await generate_answer("Hello?", config=generator_config, client=client)
    assert result.answer
    assert client.call_count == 2
    assert result.attempt == 2


@pytest.mark.asyncio
async def test_generate_answer_timeout_exhausts_retries(generator_config):
    client = FakeGroqClient(fail_with=GenerationTimeoutError, fail_times=99)
    with pytest.raises(GenerationTimeoutError):
        await generate_answer("Hello?", config=generator_config, client=client)
    assert client.call_count == generator_config.max_retries + 1


@pytest.mark.asyncio
async def test_generate_answer_network_error_bounded_retries(generator_config):
    client = FakeGroqClient(fail_with=NetworkError, fail_times=99)
    with pytest.raises(NetworkError):
        await generate_answer("Hello?", config=generator_config, client=client)
    assert client.call_count == generator_config.max_retries + 1


@pytest.mark.asyncio
async def test_generate_answer_secrets_never_in_error_message(generator_config, caplog):
    secret_cfg = GeneratorConfig(api_key="super-secret-value-123", model="fake-model", max_retries=0)
    client = FakeGroqClient(fail_with=AuthenticationError, fail_times=1)
    with pytest.raises(AuthenticationError) as exc_info:
        await generate_answer("Hello?", config=secret_cfg, client=client)
    assert "super-secret-value-123" not in str(exc_info.value)
    for record in caplog.records:
        assert "super-secret-value-123" not in record.getMessage()


def test_generator_config_rejects_unbounded_retries(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GROQ_MAX_RETRIES", "999")
    with pytest.raises(ConfigError):
        GeneratorConfig.from_env()


def test_generator_config_from_env_reads_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    cfg = GeneratorConfig.from_env()
    assert cfg.model == "openai/gpt-oss-120b"
    assert cfg.api_key == "x"


def test_generator_config_default_model_is_gpt_oss_120b(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    cfg = GeneratorConfig.from_env()
    assert cfg.model == "openai/gpt-oss-120b"
    # Must never silently default to the Qwen model reserved for M3's
    # second verifier — generator and verifier must stay different models.
    assert "qwen" not in cfg.model.lower()
