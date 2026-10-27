"""Shared TrustLens contract. Mirrored in frontend/src/types/trustlens.ts.

Change a field only after all three members agree (docs/contract_changes_m3.md
lists the fields M3 proposed on top of the team plan).
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Verdict = Literal["supported", "contradicted", "unverifiable"]
VerifierName = Literal["gemini", "groq_qwen"]

VERDICTS: tuple[str, ...] = ("supported", "contradicted", "unverifiable")


class Span(BaseModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class Claim(BaseModel):
    claim_id: str
    answer_id: str
    text: str
    span: Span
    claim_type: str = "factual"
    importance: str = "normal"


class Evidence(BaseModel):
    evidence_id: str
    claim_id: str
    source: str
    url: str
    passage: str
    retrieved_at: str | None = None
    keyword_score: float | None = None
    semantic_score: float | None = None
    combined_score: float | None = None
    # proposed by M3: the passage sentence that best matches the claim
    highlight_span: Span | None = None
    highlight_score: float | None = None


class VerificationResult(BaseModel):
    claim_id: str
    verifier: VerifierName
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    explanation: str
    evidence_ids: list[str] = []
    model_version: str


class ClaimDecision(BaseModel):
    claim_id: str
    final_verdict: Verdict
    verifier_agreement: bool
    verifier_results: list[VerificationResult]
    self_consistency_score: float | None = None
    evidence_relevance_score: float | None = None
    needs_human_review: bool
    correction_text: str | None = None
    # proposed by M3: result of the correction checker
    correction_verified: bool | None = None
    correction_support_score: float | None = None
    cache_hit: bool = False
    # the verifiers first disagreed, saw each other's reasoning once, and then agreed
    resolved_by_debate: bool = False
    first_verdicts: dict[str, Verdict] | None = None


class TrustScore(BaseModel):
    answer_id: str
    trust_score: float = Field(ge=0, le=1)
    method: str
    claim_breakdown: dict[str, int]
    disclaimer: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: str | None = None


class AnalyzeRequest(BaseModel):
    question: str | None = None
    answer_text: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> "AnalyzeRequest":
        if bool(self.question) == bool(self.answer_text):
            raise ValueError("send exactly one of 'question' or 'answer_text'")
        return self


class AnalysisResult(BaseModel):
    answer_id: str
    # proposed by M3: the UI needs the answer text and evidence to render
    question: str | None = None
    answer_text: str
    claims: list[Claim]
    evidence: list[Evidence] = []
    decisions: list[ClaimDecision]
    trust_score: TrustScore
    created_at: str | None = None
    # who wrote the answer and anything that happened on the way (key switches, fallbacks)
    answer_source: str | None = None
    # 'opinion' when the question has no single true answer; verdicts then cover the facts, not the opinion
    question_type: str = "factual"
    notes: list[str] = []


class HistoryItem(BaseModel):
    """Proposed by M3: one row of GET /api/v1/history."""

    answer_id: str
    question: str | None = None
    trust_score: float
    claim_count: int
    created_at: str | None = None
