from __future__ import annotations

import numpy as np
import pytest

from trustlens_m1.embeddings import cosine_similarity, embed_texts
from trustlens_m1.errors import EmbeddingError, InvalidInputError


def test_embed_texts_returns_matrix(fake_embedding_model):
    vectors = embed_texts(["hello world", "goodbye world"], _model=fake_embedding_model)
    assert vectors.shape[0] == 2
    assert vectors.dtype == np.float32


def test_embed_texts_rejects_empty_list(fake_embedding_model):
    with pytest.raises(InvalidInputError):
        embed_texts([], _model=fake_embedding_model)


def test_embed_texts_rejects_empty_string(fake_embedding_model):
    with pytest.raises(InvalidInputError):
        embed_texts(["valid", "   "], _model=fake_embedding_model)


def test_embed_texts_rejects_non_list(fake_embedding_model):
    with pytest.raises(InvalidInputError):
        embed_texts("not a list", _model=fake_embedding_model)  # type: ignore[arg-type]


def test_embed_texts_rejects_non_string_element(fake_embedding_model):
    with pytest.raises(InvalidInputError):
        embed_texts(["ok", 5], _model=fake_embedding_model)  # type: ignore[list-item]


def test_cosine_similarity_identical_text_is_high(fake_embedding_model):
    sim = cosine_similarity(
        "the capital of France is Paris",
        "the capital of France is Paris",
        _model=fake_embedding_model,
    )
    assert sim == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_different_text_is_lower(fake_embedding_model):
    sim_same = cosine_similarity("cats are mammals", "cats are mammals", _model=fake_embedding_model)
    sim_diff = cosine_similarity(
        "cats are mammals", "stock markets fell today", _model=fake_embedding_model
    )
    assert sim_diff < sim_same


def test_cosine_similarity_rejects_empty_text(fake_embedding_model):
    with pytest.raises(InvalidInputError):
        cosine_similarity("", "hello", _model=fake_embedding_model)


def test_embedding_model_load_error_wrapped(monkeypatch):
    import trustlens_m1.embeddings as emb

    emb.reset_model_cache()

    def boom(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(emb, "_resolve_device", lambda d: "cpu")

    class FailingSentenceTransformer:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no network")

    import sys
    import types

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FailingSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    with pytest.raises(EmbeddingError):
        embed_texts(["hello"])

    emb.reset_model_cache()
