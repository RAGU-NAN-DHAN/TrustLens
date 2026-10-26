"""Phase 2A — shared local embedding service.

Single source of truth for sentence embeddings across M1, M2, and M3.
M2 should use ``embed_texts`` / ``cosine_similarity`` for evidence ranking
and M3 may use it for evidence highlighting, rather than re-instantiating
``sentence-transformers`` themselves — this keeps model loading, device
selection, and batching behavior consistent across the pipeline.

The model is downloaded on first use (see README "First-time setup") and
runs entirely locally afterward; no network calls happen at embedding time
once weights are cached.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from .config import EmbeddingConfig
from .errors import EmbeddingError, InvalidInputError
from .logging_utils import get_logger, log_event

logger = get_logger("embeddings")

_model_lock = threading.Lock()
_loaded_model = None  # type: ignore[var-annotated]
_loaded_model_name: str | None = None


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore[import-not-found]

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_model(config: EmbeddingConfig | None = None):
    """Lazily load (once) and cache the sentence-transformers model.

    Thread-safe: concurrent first calls block on a lock rather than loading
    the model twice.
    """
    global _loaded_model, _loaded_model_name

    cfg = config or EmbeddingConfig.from_env()

    if _loaded_model is not None and _loaded_model_name == cfg.model_name:
        return _loaded_model

    with _model_lock:
        if _loaded_model is not None and _loaded_model_name == cfg.model_name:
            return _loaded_model
        try:
            from sentence_transformers import (
                SentenceTransformer,  # type: ignore[import-not-found]
            )
        except ImportError as exc:
            raise EmbeddingError(
                "The 'sentence-transformers' package is not installed. "
                "Run `pip install sentence-transformers`."
            ) from exc

        device = _resolve_device(cfg.device)
        log_event(
            logger,
            logging.INFO,
            "embeddings.model_load.start",
            model=cfg.model_name,
            device=device,
        )
        try:
            model = SentenceTransformer(cfg.model_name, device=device)
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "embeddings.model_load.failed",
                model=cfg.model_name,
                error_type=type(exc).__name__,
            )
            raise EmbeddingError(
                f"Failed to load embedding model '{cfg.model_name}'. "
                "If this is the first run, confirm network access for the "
                "one-time weight download, or pre-download the model. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        log_event(logger, logging.INFO, "embeddings.model_load.success", model=cfg.model_name)
        _loaded_model = model
        _loaded_model_name = cfg.model_name
        return _loaded_model


def reset_model_cache() -> None:
    """Drop the cached model instance. Intended for tests only."""
    global _loaded_model, _loaded_model_name
    with _model_lock:
        _loaded_model = None
        _loaded_model_name = None


def _validate_texts(texts: list[str]) -> list[str]:
    if texts is None:
        raise InvalidInputError("texts must not be None")
    if not isinstance(texts, list):
        raise InvalidInputError(f"texts must be a list, got {type(texts).__name__}")
    cleaned: list[str] = []
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise InvalidInputError(f"texts[{i}] must be a string, got {type(t).__name__}")
        stripped = t.strip()
        if not stripped:
            raise InvalidInputError(f"texts[{i}] must not be empty or whitespace-only")
        cleaned.append(stripped)
    return cleaned


def embed_texts(
    texts: list[str],
    *,
    config: EmbeddingConfig | None = None,
    _model: Any = None,
) -> np.ndarray:
    """Encode a batch of texts into embedding vectors.

    Args:
        texts: Non-empty list of non-empty strings.
        config: Optional ``EmbeddingConfig``; defaults to env-derived config.
        _model: Internal hook for injecting a fake model in tests. Not part
            of the public contract.

    Returns:
        A ``numpy.ndarray`` of shape ``(len(texts), embedding_dim)``, dtype
        float32, L2-normalized (so dot product == cosine similarity).

    Raises:
        InvalidInputError: ``texts`` is empty, not a list, or contains a
            non-string / empty / whitespace-only element.
        EmbeddingError: the model failed to load or encode.
    """
    cleaned = _validate_texts(texts)
    if not cleaned:
        raise InvalidInputError("texts must contain at least one item")

    model = _model if _model is not None else _get_model(config)
    cfg = config or EmbeddingConfig.from_env()

    try:
        vectors = model.encode(
            cleaned,
            batch_size=cfg.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingError(f"Embedding inference failed: {type(exc).__name__}: {exc}") from exc

    return np.asarray(vectors, dtype=np.float32)


def cosine_similarity(
    text_a: str,
    text_b: str,
    *,
    config: EmbeddingConfig | None = None,
    _model: Any = None,
) -> float:
    """Return the cosine similarity between two texts' embeddings, in [-1, 1].

    Raises:
        InvalidInputError: either text is empty/whitespace/non-string.
        EmbeddingError: the model failed to load or encode.
    """
    vectors = embed_texts([text_a, text_b], config=config, _model=_model)
    sim = float(np.dot(vectors[0], vectors[1]))
    return max(-1.0, min(1.0, sim))


def cosine_similarity_matrix(
    texts_a: list[str],
    texts_b: list[str],
    *,
    config: EmbeddingConfig | None = None,
    _model: Any = None,
) -> np.ndarray:
    """Pairwise cosine similarity between two batches: shape (len(a), len(b))."""
    vec_a = embed_texts(texts_a, config=config, _model=_model)
    vec_b = embed_texts(texts_b, config=config, _model=_model)
    sims = vec_a @ vec_b.T
    return np.clip(sims, -1.0, 1.0)
