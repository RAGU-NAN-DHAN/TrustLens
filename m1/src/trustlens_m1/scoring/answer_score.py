"""Whole-answer trust score — aggregates per-claim TrustScoreResults into
one score for the complete generated answer.

``TrustScorer.predict()`` (see predict.py) scores one claim at a time. The
product surface (the source project plan's colour-coded answer view) shows
a single trust score for the *whole* answer, so this module provides the
missing aggregation step. It is pure post-processing over already-computed
per-claim results — it does not call Groq, an embedding model, or the
trust-scoring model itself, and it does not reinterpret or override any
individual claim's verdict.

Design notes
------------
- ``overall_trust_score`` defaults to the **weakest-link** (minimum)
  aggregation across claims, not the mean. Rationale: a reader of the final
  answer cannot tell which individual claim is untrustworthy, so the
  answer's overall trustworthiness is bounded by its least-trustworthy
  claim rather than diluted by averaging it against several trustworthy
  ones. ``"mean"`` and ``"weakest_link_mean_blend"`` are also available for
  callers who want a softer aggregate; the method used is always recorded
  on the result for explainability (one of the three novelty signals named
  in the source project plan).
- ``overall_label`` is rule-based, not derived from the numeric score: if
  any claim was predicted ``contradicted`` the whole answer is labelled
  ``contradicted`` regardless of how high the numeric score is, so a single
  false claim can never be hidden by an otherwise-strong average.
- ``weakest_claim`` identifies which specific claim drove the score down,
  so a UI can explain *why* the answer was flagged (again, explainability).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..dataset.schema import GOLD_LABELS
from ..errors import InvalidInputError
from .features import FusionFeatures
from .predict import TrustScorer, TrustScoreResult

AGGREGATION_METHODS = ("weakest_link", "mean", "weakest_link_mean_blend")
DEFAULT_AGGREGATION_METHOD = "weakest_link"


@dataclass(frozen=True)
class ClaimTrustScore:
    """One claim's trust result, as input to whole-answer aggregation.

    This is a thin, order-preserving wrapper around ``TrustScoreResult``
    that attaches an optional claim identifier/text so the aggregated
    result can point back at *which* claim mattered most.
    """

    predicted_label: str
    trust_score: float
    probabilities: dict[str, float]
    claim_id: str | None = None
    claim_text: str | None = None

    def __post_init__(self) -> None:
        if self.predicted_label not in GOLD_LABELS:
            raise InvalidInputError(
                f"predicted_label={self.predicted_label!r} must be one of {GOLD_LABELS}"
            )
        if not isinstance(self.trust_score, (int, float)) or isinstance(self.trust_score, bool):
            raise InvalidInputError("trust_score must be numeric")
        if not (0.0 <= float(self.trust_score) <= 1.0):
            raise InvalidInputError(f"trust_score must be in [0, 1], got {self.trust_score}")

    @staticmethod
    def from_trust_score_result(
        result: TrustScoreResult,
        *,
        claim_id: str | None = None,
        claim_text: str | None = None,
    ) -> ClaimTrustScore:
        return ClaimTrustScore(
            predicted_label=result.predicted_label,
            trust_score=result.trust_score,
            probabilities=dict(result.probabilities),
            claim_id=claim_id,
            claim_text=claim_text,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_label": self.predicted_label,
            "trust_score": self.trust_score,
            "probabilities": self.probabilities,
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
        }


@dataclass(frozen=True)
class AnswerTrustResult:
    """The data contract M2/M3/the UI consume for a whole generated answer.

    Attributes:
        overall_trust_score: 0-1, aggregated per ``aggregation_method``.
        overall_label: one of 'supported', 'partially_unverifiable',
            'unverifiable', 'contradicted' — see module docstring for the
            rule (not derived purely from the numeric score).
        n_claims, n_supported, n_contradicted, n_unverifiable: claim counts.
        weakest_claim: the ClaimTrustScore with the lowest trust_score
            (ties broken by first occurrence), for explainability.
        per_claim: all input ClaimTrustScore rows, in the order given.
        aggregation_method: which method produced overall_trust_score.
    """

    overall_trust_score: float
    overall_label: str
    n_claims: int
    n_supported: int
    n_contradicted: int
    n_unverifiable: int
    weakest_claim: ClaimTrustScore | None
    per_claim: list[ClaimTrustScore] = field(default_factory=list)
    aggregation_method: str = DEFAULT_AGGREGATION_METHOD

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_trust_score": self.overall_trust_score,
            "overall_label": self.overall_label,
            "n_claims": self.n_claims,
            "n_supported": self.n_supported,
            "n_contradicted": self.n_contradicted,
            "n_unverifiable": self.n_unverifiable,
            "weakest_claim": self.weakest_claim.to_dict() if self.weakest_claim else None,
            "per_claim": [c.to_dict() for c in self.per_claim],
            "aggregation_method": self.aggregation_method,
        }


def _overall_label(n_supported: int, n_contradicted: int, n_unverifiable: int) -> str:
    # Any contradicted claim overrides everything else: a single false
    # claim must not be hidden behind an otherwise-strong answer.
    if n_contradicted > 0:
        return "contradicted"
    if n_unverifiable > 0 and n_supported > 0:
        return "partially_unverifiable"
    if n_unverifiable > 0:
        return "unverifiable"
    return "supported"


def score_answer(
    claim_scores: list[ClaimTrustScore],
    *,
    method: str = DEFAULT_AGGREGATION_METHOD,
) -> AnswerTrustResult:
    """Aggregate per-claim trust scores into one whole-answer trust result.

    Args:
        claim_scores: one ``ClaimTrustScore`` per claim in the answer, in
            any order. Must be non-empty — an answer with zero claims
            (e.g. claim splitting failed, or produced nothing) has no
            meaningful trust score and is a caller-side error to aggregate.
        method: one of ``AGGREGATION_METHODS``. Defaults to
            ``"weakest_link"`` (the minimum trust_score across claims); see
            the module docstring for the rationale, and ``"mean"`` /
            ``"weakest_link_mean_blend"`` for softer alternatives.

    Returns:
        An ``AnswerTrustResult``.

    Raises:
        InvalidInputError: ``claim_scores`` is empty/not a list, or
            ``method`` is not a recognized aggregation method.
    """
    if not isinstance(claim_scores, list):
        raise InvalidInputError(
            f"claim_scores must be a list, got {type(claim_scores).__name__}"
        )
    if not claim_scores:
        raise InvalidInputError(
            "claim_scores must be non-empty — cannot compute a whole-answer "
            "trust score for an answer with zero claims"
        )
    if method not in AGGREGATION_METHODS:
        raise InvalidInputError(
            f"method={method!r} must be one of {AGGREGATION_METHODS}"
        )

    scores = [c.trust_score for c in claim_scores]
    mean_score = statistics.fmean(scores)
    min_score = min(scores)

    if method == "weakest_link":
        overall_trust_score = min_score
    elif method == "mean":
        overall_trust_score = mean_score
    else:  # weakest_link_mean_blend
        overall_trust_score = (min_score + mean_score) / 2.0

    n_supported = sum(1 for c in claim_scores if c.predicted_label == "supported")
    n_contradicted = sum(1 for c in claim_scores if c.predicted_label == "contradicted")
    n_unverifiable = sum(1 for c in claim_scores if c.predicted_label == "unverifiable")

    weakest_claim = min(claim_scores, key=lambda c: c.trust_score)

    return AnswerTrustResult(
        overall_trust_score=overall_trust_score,
        overall_label=_overall_label(n_supported, n_contradicted, n_unverifiable),
        n_claims=len(claim_scores),
        n_supported=n_supported,
        n_contradicted=n_contradicted,
        n_unverifiable=n_unverifiable,
        weakest_claim=weakest_claim,
        per_claim=list(claim_scores),
        aggregation_method=method,
    )


def score_answer_from_features(
    scorer: TrustScorer,
    claims: list[FusionFeatures],
    *,
    method: str = DEFAULT_AGGREGATION_METHOD,
) -> AnswerTrustResult:
    """Convenience one-call path: score every claim's FusionFeatures with
    ``scorer``, then aggregate into a whole-answer result.

    Equivalent to calling ``scorer.predict()`` once per claim and feeding
    the results into ``score_answer`` yourself, provided for callers (M2's
    FastAPI server, once it exists) that just want one function to call
    per request.

    Raises:
        InvalidInputError: ``claims`` is empty/not a list, or ``method`` is
            invalid (propagated from ``score_answer``).
    """
    if not isinstance(claims, list):
        raise InvalidInputError(f"claims must be a list, got {type(claims).__name__}")
    if not claims:
        raise InvalidInputError("claims must be non-empty")

    claim_scores = [
        ClaimTrustScore.from_trust_score_result(
            scorer.predict(features), claim_id=features.claim_id
        )
        for features in claims
    ]
    return score_answer(claim_scores, method=method)


def main(argv: list[str] | None = None) -> int:
    """CLI: aggregate a JSON file of per-claim trust results.

    Input JSON shape: a list of objects each with at least
    ``predicted_label``, ``trust_score``, ``probabilities`` (and optionally
    ``claim_id`` / ``claim_text``) — i.e. ``TrustScoreResult.to_dict()``
    output, one per claim, exactly as produced by
    ``scoring.predict.TrustScorer.predict().to_dict()``.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate per-claim trust scores (JSON list) into one whole-answer trust score."
    )
    parser.add_argument("claims_json", help="Path to a JSON file: a list of TrustScoreResult-shaped objects")
    parser.add_argument("--method", choices=AGGREGATION_METHODS, default=DEFAULT_AGGREGATION_METHOD)
    args = parser.parse_args(argv)

    path = Path(args.claims_json)
    if not path.exists():
        print(f"FAILED: file not found: {path}", file=sys.stderr)
        return 1
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise InvalidInputError("top-level JSON must be a list of per-claim results")
        claim_scores = [
            ClaimTrustScore(
                predicted_label=row["predicted_label"],
                trust_score=row["trust_score"],
                probabilities=row.get("probabilities", {}),
                claim_id=row.get("claim_id"),
                claim_text=row.get("claim_text"),
            )
            for row in raw
        ]
        result = score_answer(claim_scores, method=args.method)
    except (InvalidInputError, KeyError, json.JSONDecodeError) as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
