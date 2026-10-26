"""Shared test fixtures.

Nothing here makes a real Groq API call or downloads real embedding model
weights: the fake client/model below are deterministic, in-process doubles.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from trustlens_m1.config import GeneratorConfig

try:
    import pytest_asyncio  # noqa: F401

    _HAS_PYTEST_ASYNCIO = True
except ImportError:
    _HAS_PYTEST_ASYNCIO = False

if not _HAS_PYTEST_ASYNCIO:
    # Fallback async-test runner used only when pytest-asyncio isn't
    # installed (e.g. an offline sandbox). Production/CI should install
    # pytest-asyncio per requirements-dev.txt; when it's present this whole
    # block is inert and pytest-asyncio handles `@pytest.mark.asyncio` tests
    # as normal.
    import asyncio
    import inspect

    def pytest_configure(config: pytest.Config) -> None:
        config.addinivalue_line(
            "markers",
            "asyncio: run this async test (fallback runner active; "
            "install pytest-asyncio for the standard plugin)",
        )

    @pytest.hookimpl(tryfirst=True)
    def pytest_pyfunc_call(pyfuncitem: pytest.Function):
        test_func = pyfuncitem.obj
        if inspect.iscoroutinefunction(test_func):
            argnames = pyfuncitem._fixtureinfo.argnames
            kwargs = {name: pyfuncitem.funcargs[name] for name in argnames}
            asyncio.run(test_func(**kwargs))
            return True
        return None


class FakeGroqClient:
    """Deterministic stand-in for GroqChatClient.create_chat_completion.

    ``answers`` is either a single string (returned every call) or a list
    consumed in order (one per call); ``fail_with`` optionally raises that
    exception class on every call instead of succeeding.
    """

    def __init__(
        self,
        answers: list[str] | str | None = None,
        *,
        fail_with: type[Exception] | None = None,
        fail_times: int = 0,
    ) -> None:
        self._answers = answers if answers is not None else "This is a fake answer."
        self._fail_with = fail_with
        self._fail_times = fail_times
        self.call_count = 0

    async def create_chat_completion(
        self,
        *,
        model: str,
        question: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> dict[str, Any]:
        self.call_count += 1
        if self._fail_with is not None and self.call_count <= self._fail_times:
            raise self._fail_with("synthetic failure for testing")

        if isinstance(self._answers, list):
            idx = min(self.call_count - 1, len(self._answers) - 1)
            answer = self._answers[idx]
        else:
            answer = self._answers

        return {
            "answer": answer,
            "model": model,
            "request_id": f"fake-req-{self.call_count}",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }


class FakeEmbeddingModel:
    """Deterministic bag-of-words embedding so similarity tests are stable
    without downloading real sentence-transformers weights.
    """

    _VOCAB_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = self._VOCAB_RE.findall(text.lower())
            for word in words:
                h = hash(word) % self.dim
                vectors[i, h] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] = vectors[i] / norm
        return vectors


@pytest.fixture
def fake_embedding_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel()


@pytest.fixture
def generator_config() -> GeneratorConfig:
    return GeneratorConfig(api_key="fake-test-key", model="fake-model", max_retries=1)
