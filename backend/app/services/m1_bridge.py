"""Calls into M1 (trustlens_m1): answer generation, self-consistency and the trust model.

Each piece degrades on its own: if M1's embedding model or trained trust model is not
available, the caller gets None and records that the signal was not computed.
"""

import asyncio
from pathlib import Path

from app.errors import TrustLensError

TRUST_MODEL = Path(__file__).resolve().parents[3] / "m1" / "data" / "trust_model.joblib"
SC_SAMPLES = 3


class GenerationError(TrustLensError):
    error_code = "GENERATION_FAILED"


def embeddings_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import trustlens_m1.embeddings  # noqa: F401
    except ImportError:
        return False
    return True


async def generate(question: str) -> tuple[str, str]:
    """(answer text, model id) from M1's Groq generator."""
    from trustlens_m1.errors import TrustLensM1Error
    from trustlens_m1.generator import generate_answer

    try:
        result = await generate_answer(question)
    except TrustLensM1Error as exc:
        raise GenerationError(f"answer generation failed: {type(exc).__name__}: {exc}") from exc
    return result.answer, result.model


async def self_consistency(question: str, claims: list[str]) -> list[float] | None:
    """Per-claim consistency across extra high-temperature samples, or None when M1's embeddings are not installed."""
    if not claims or not embeddings_available():
        return None
    from trustlens_m1.config import SelfConsistencyConfig
    from trustlens_m1.generator import generate_answer
    from trustlens_m1.self_consistency import score_claim_consistency

    sc = SelfConsistencyConfig.from_env() if hasattr(SelfConsistencyConfig, "from_env") else SelfConsistencyConfig()
    samples = []
    for _ in range(SC_SAMPLES):
        try:
            samples.append((await generate_answer(question, temperature=getattr(sc, "temperature", 0.9))).answer)
        except Exception:  # one failed sample should not sink the analysis
            continue
    if not samples:
        return None
    return [score_claim_consistency(c, samples).consistency_score for c in claims]


def run_async(coro):
    return asyncio.run(coro)


def trust_scores(rows: list[dict]) -> list[float] | None:
    """P(supported) per claim from M1's trained model, or None if a signal or the model is missing."""
    if not TRUST_MODEL.exists() or any(r["self_consistency"] is None or r["evidence_relevance"] is None for r in rows):
        return None
    try:
        from trustlens_m1.scoring.features import FusionFeatures
        from trustlens_m1.scoring.predict import TrustScorer
    except ImportError:
        return None
    scorer = TrustScorer(TRUST_MODEL)
    out = []
    for r in rows:
        features = FusionFeatures(
            self_consistency_score=float(r["self_consistency"]),
            evidence_relevance_score=float(r["evidence_relevance"]),
            verifier_agreement=bool(r["agreement"]),
            verifier_a_verdict=r["verdict_a"],
            verifier_b_verdict=r["verdict_b"],
            claim_id=r["claim_id"],
        )
        out.append(scorer.predict(features).trust_score)
    return out
