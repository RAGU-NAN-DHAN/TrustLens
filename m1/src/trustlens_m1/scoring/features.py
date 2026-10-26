"""Phase 3B — the fusion feature contract.

``FusionFeatures`` is the single-claim input to the trust-scoring model.
Three of its five fields come from other team members once the full
pipeline is wired together:

    self_consistency_score   <- M1 (self_consistency.py)
    evidence_relevance_score <- M2 (evidence ranking)
    verifier_agreement       <- M3 (dual verifier + disagreement rule)
    verifier_a_verdict       <- M3
    verifier_b_verdict       <- M3

M1 never computes verifier_agreement/verdicts itself and never overrides
M3's disagreement decision — see M1_INTEGRATION_CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..dataset.schema import VERIFIER_VERDICTS
from ..errors import InvalidInputError

FEATURE_SCHEMA_VERSION = "1.0.0"

FEATURE_COLUMNS = (
    "self_consistency_score",
    "evidence_relevance_score",
    "verifier_agreement",
    "verifier_a_verdict",
    "verifier_b_verdict",
)


@dataclass(frozen=True)
class FusionFeatures:
    """One claim's feature vector for trust scoring.

    Attributes:
        self_consistency_score: float in [0, 1].
        evidence_relevance_score: float in [0, 1].
        verifier_agreement: bool — did Verifier A and B agree?
        verifier_a_verdict: one of 'supported'/'contradicted'/'unverifiable'.
        verifier_b_verdict: one of 'supported'/'contradicted'/'unverifiable'.
        claim_id: optional identifier for traceability (not used as a
            model feature).
    """

    self_consistency_score: float
    evidence_relevance_score: float
    verifier_agreement: bool
    verifier_a_verdict: str
    verifier_b_verdict: str
    claim_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("self_consistency_score", "evidence_relevance_score"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidInputError(f"{name} must be numeric, got {type(value).__name__}")
            if not (0.0 <= float(value) <= 1.0):
                raise InvalidInputError(f"{name} must be in [0, 1], got {value}")
        if not isinstance(self.verifier_agreement, bool):
            raise InvalidInputError(
                f"verifier_agreement must be a bool, got {type(self.verifier_agreement).__name__}"
            )
        for name in ("verifier_a_verdict", "verifier_b_verdict"):
            value = getattr(self, name)
            if value not in VERIFIER_VERDICTS:
                raise InvalidInputError(
                    f"{name}={value!r} must be one of {VERIFIER_VERDICTS}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "self_consistency_score": self.self_consistency_score,
            "evidence_relevance_score": self.evidence_relevance_score,
            "verifier_agreement": self.verifier_agreement,
            "verifier_a_verdict": self.verifier_a_verdict,
            "verifier_b_verdict": self.verifier_b_verdict,
            "claim_id": self.claim_id,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> FusionFeatures:
        try:
            return FusionFeatures(
                self_consistency_score=float(data["self_consistency_score"]),
                evidence_relevance_score=float(data["evidence_relevance_score"]),
                verifier_agreement=bool(data["verifier_agreement"]),
                verifier_a_verdict=str(data["verifier_a_verdict"]),
                verifier_b_verdict=str(data["verifier_b_verdict"]),
                claim_id=data.get("claim_id"),
            )
        except KeyError as exc:
            raise InvalidInputError(f"FusionFeatures is missing required key: {exc}") from exc
