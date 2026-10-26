"""Phase 2B — self-consistency sampling (hallucination signal).

Method: ask the generator the same question multiple times at a higher
temperature and measure how similar the resulting answers are to each other
(SelfCheckGPT-style). A low similarity across samples suggests the model
does not reliably "know" the answer, which is a signal correlated with
hallucination — this module reports *consistency*, not factual correctness.
No model is ever asked to judge truth here.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from .config import EmbeddingConfig, GeneratorConfig, SelfConsistencyConfig
from .embeddings import cosine_similarity, embed_texts
from .errors import GenerationError, InvalidInputError
from .generator import GenerationResult, SupportsChatCompletion, generate_answer
from .logging_utils import get_logger, log_event

logger = get_logger("self_consistency")

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ClaimConsistencyResult:
    """Claim-level consistency estimate — distinct from overall-answer consistency.

    Attributes:
        claim: The claim text being scored (supplied by M2's claim splitter
            once available; may also be a full sentence from the canonical
            answer when used standalone).
        best_sentence_similarities: For each sampled answer, the cosine
            similarity between the claim and that sample's single
            most-similar sentence (len == number of samples that had at
            least one sentence).
        consistency_score: Mean of ``best_sentence_similarities``, normalized
            to [0, 1]. Higher means the claim's content recurs consistently
            across independent samples.
        support_count: Number of samples whose best-matching sentence
            exceeded ``match_threshold`` — a coarse "how many samples
            appear to restate this claim" count.
        sample_count: Total number of sampled answers considered.
        match_threshold: The similarity threshold used for ``support_count``.
    """

    claim: str
    best_sentence_similarities: list[float]
    consistency_score: float
    support_count: int
    sample_count: int
    match_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "best_sentence_similarities": self.best_sentence_similarities,
            "consistency_score": self.consistency_score,
            "support_count": self.support_count,
            "sample_count": self.sample_count,
            "match_threshold": self.match_threshold,
        }


@dataclass(frozen=True)
class SelfConsistencyResult:
    """The data contract M2/M3 consume for a self-consistency run.

    Attributes:
        question: The original question.
        canonical_answer: The low-temperature (or configured default
            temperature) reference answer.
        sampled_answers: The additional higher-temperature samples (3-5).
        overall_consistency_score: Mean pairwise cosine similarity across
            {canonical_answer} UNION {sampled_answers}, normalized to
            [0, 1]. This is a consistency estimate, NOT a correctness score.
        sample_count: Number of additional samples requested/produced.
        per_claim_results: Optional list of ``ClaimConsistencyResult``,
            populated only when claims are supplied (claim splitting is
            owned by M2; empty by default).
        duration_seconds: Total wall-clock time for canonical + all samples.
        errors: Any per-sample generation errors encountered (sampling
            continues with the samples that did succeed, as long as at
            least one sample succeeds); never contains credential values.
        temperature: Sampling temperature used for the additional samples.
    """

    question: str
    canonical_answer: str
    sampled_answers: list[str]
    overall_consistency_score: float
    sample_count: int
    temperature: float
    duration_seconds: float
    per_claim_results: list[ClaimConsistencyResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "canonical_answer": self.canonical_answer,
            "sampled_answers": self.sampled_answers,
            "overall_consistency_score": self.overall_consistency_score,
            "sample_count": self.sample_count,
            "temperature": self.temperature,
            "duration_seconds": self.duration_seconds,
            "per_claim_results": [r.to_dict() for r in self.per_claim_results],
            "errors": self.errors,
        }


def _split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _normalize_cosine(value: float) -> float:
    """Map cosine similarity in [-1, 1] to a consistency score in [0, 1]."""
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


async def run_self_consistency(
    question: str,
    *,
    generator_config: GeneratorConfig | None = None,
    sc_config: SelfConsistencyConfig | None = None,
    client: SupportsChatCompletion | None = None,
    claims: list[str] | None = None,
    embedding_config: EmbeddingConfig | None = None,
    _embedding_model: object = None,
) -> SelfConsistencyResult:
    """Generate a canonical answer plus 3-5 extra samples and score consistency.

    Args:
        question: The user's question (validated by ``generate_answer``).
        generator_config: Optional ``GeneratorConfig``; defaults to env.
        sc_config: Optional ``SelfConsistencyConfig`` (sample count 3-5,
            temperature, max tokens); defaults to env.
        client: Optional injected Groq-compatible client (shared across
            calls, or a fake for tests).
        claims: Optional list of already-split claim strings (owned by M2's
            claim splitter). When provided, ``per_claim_results`` is
            populated via ``score_claim_consistency``.
        embedding_config: Optional ``EmbeddingConfig`` override.
        _embedding_model: Internal test hook, passed through to the
            embeddings module.

    Returns:
        A ``SelfConsistencyResult``.

    Raises:
        InvalidInputError: invalid question.
        GenerationError: if the canonical generation fails, or if every
            additional sample fails (at least one successful sample is
            required to compute a score; if zero samples succeed this is
            raised rather than returning a misleading score of 0).
    """
    gen_cfg = generator_config or GeneratorConfig.from_env()
    cfg = sc_config or SelfConsistencyConfig.from_env()

    start = time.monotonic()

    canonical = await generate_answer(question, config=gen_cfg, client=client)

    sample_tasks = [
        generate_answer(
            question,
            config=gen_cfg,
            client=client,
            temperature=cfg.temperature,
            max_tokens=cfg.max_output_tokens,
        )
        for _ in range(cfg.sample_count)
    ]
    results = await asyncio.gather(*sample_tasks, return_exceptions=True)

    sampled_answers: list[str] = []
    errors: list[str] = []
    for r in results:
        if isinstance(r, GenerationResult):
            sampled_answers.append(r.answer)
        elif isinstance(r, GenerationError):
            errors.append(f"{type(r).__name__}: {r}")
        elif isinstance(r, BaseException):
            raise r

    if errors:
        log_event(
            logger,
            logging.WARNING,
            "self_consistency.sample_errors",
            error_count=len(errors),
            succeeded=len(sampled_answers),
            requested=cfg.sample_count,
        )

    if not sampled_answers:
        raise GenerationError(
            "All self-consistency samples failed; cannot compute a consistency "
            f"score. Errors: {errors}"
        )

    all_answers = [canonical.answer] + sampled_answers
    if len(all_answers) >= 2:
        vectors = embed_texts(all_answers, config=embedding_config, _model=_embedding_model)
        sims = vectors @ vectors.T
        n = len(all_answers)
        pair_values = [sims[i, j] for i, j in combinations(range(n), 2)]
        overall_raw = float(np.mean(pair_values)) if pair_values else 1.0
    else:
        overall_raw = 1.0
    overall_score = _normalize_cosine(overall_raw)

    per_claim: list[ClaimConsistencyResult] = []
    if claims:
        for claim in claims:
            per_claim.append(
                score_claim_consistency(
                    claim,
                    sampled_answers,
                    embedding_config=embedding_config,
                    _embedding_model=_embedding_model,
                )
            )

    duration = time.monotonic() - start
    log_event(
        logger,
        logging.INFO,
        "self_consistency.complete",
        sample_count=len(sampled_answers),
        requested_sample_count=cfg.sample_count,
        overall_consistency_score=overall_score,
        duration_seconds=duration,
        error_count=len(errors),
    )

    return SelfConsistencyResult(
        question=canonical.question,
        canonical_answer=canonical.answer,
        sampled_answers=sampled_answers,
        overall_consistency_score=overall_score,
        sample_count=len(sampled_answers),
        temperature=cfg.temperature,
        duration_seconds=duration,
        per_claim_results=per_claim,
        errors=errors,
    )


def score_claim_consistency(
    claim: str,
    sampled_answers: list[str],
    *,
    match_threshold: float = 0.6,
    embedding_config: EmbeddingConfig | None = None,
    _embedding_model: object = None,
) -> ClaimConsistencyResult:
    """Estimate how consistently ``claim`` appears across independent samples.

    For each sampled answer, splits it into sentences and compares the
    claim's embedding against every sentence, keeping the single highest
    similarity ("does this sample say something close to this claim, and how
    closely"). The claim-level consistency score is the mean of those
    per-sample best-match similarities, normalized to [0, 1].

    This is a consistency estimate, NOT a verification of factual
    correctness — a claim can recur consistently across samples and still
    be wrong (e.g. a consistently memorized falsehood). Factual correctness
    is established by the separate evidence-verification stage (M2/M3), not
    by this function.

    Args:
        claim: The claim text to score.
        sampled_answers: The additional self-consistency samples (not
            including the canonical answer) to compare the claim against.
        match_threshold: Cosine-similarity threshold (post float, pre
            normalization is used internally; see implementation) above
            which a sample counts toward ``support_count``.
        embedding_config: Optional ``EmbeddingConfig`` override.
        _embedding_model: Internal test hook.

    Raises:
        InvalidInputError: ``claim`` is empty, or ``sampled_answers`` is
            empty / not a list.
    """
    if not isinstance(claim, str) or not claim.strip():
        raise InvalidInputError("claim must be a non-empty string")
    if not isinstance(sampled_answers, list) or not sampled_answers:
        raise InvalidInputError("sampled_answers must be a non-empty list")

    claim_text = claim.strip()
    best_similarities: list[float] = []

    for answer in sampled_answers:
        sentences = _split_sentences(answer) or ([answer.strip()] if answer.strip() else [])
        if not sentences:
            continue
        sims = [
            cosine_similarity(
                claim_text, s, config=embedding_config, _model=_embedding_model
            )
            for s in sentences
        ]
        best_similarities.append(max(sims))

    if not best_similarities:
        return ClaimConsistencyResult(
            claim=claim_text,
            best_sentence_similarities=[],
            consistency_score=0.0,
            support_count=0,
            sample_count=len(sampled_answers),
            match_threshold=match_threshold,
        )

    normalized = [_normalize_cosine(s) for s in best_similarities]
    consistency_score = float(np.mean(normalized))
    support_count = sum(1 for s in best_similarities if s >= match_threshold)

    return ClaimConsistencyResult(
        claim=claim_text,
        best_sentence_similarities=best_similarities,
        consistency_score=consistency_score,
        support_count=support_count,
        sample_count=len(sampled_answers),
        match_threshold=match_threshold,
    )
