"""Shared exception hierarchy for M1.

M2 and M3 should catch ``TrustLensM1Error`` (or a specific subclass) rather
than groq / sentence-transformers / sklearn exceptions directly, so that M1
can change its underlying libraries without breaking callers.
"""

from __future__ import annotations


class TrustLensM1Error(Exception):
    """Base class for all errors raised by the M1 package."""


class InvalidInputError(TrustLensM1Error):
    """The caller passed input that failed validation (e.g. empty question)."""


class GenerationError(TrustLensM1Error):
    """Base class for answer-generation failures."""


class AuthenticationError(GenerationError):
    """The Groq API key was missing, malformed, or rejected."""


class RateLimitError(GenerationError):
    """The Groq API returned a rate-limit response."""


class GenerationTimeoutError(GenerationError):
    """The request to Groq exceeded the configured timeout."""


class NetworkError(GenerationError):
    """A network-level failure occurred while contacting Groq."""


class UnexpectedResponseError(GenerationError):
    """Groq returned a response that did not match the expected shape."""


class EmbeddingError(TrustLensM1Error):
    """Raised for embedding-model load or inference failures."""


class DatasetError(TrustLensM1Error):
    """Raised for labelled-dataset schema or validation failures."""


class ModelTrainingError(TrustLensM1Error):
    """Raised for training-pipeline failures (bad data, convergence, etc.)."""


class ModelLoadError(TrustLensM1Error):
    """Raised when a persisted model cannot be loaded or its schema mismatches."""
