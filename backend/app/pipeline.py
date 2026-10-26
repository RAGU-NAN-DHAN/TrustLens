"""M2: the full TrustLens pipeline, wiring M1 (generation, self-consistency, trust model),
M2 (claims, evidence, ranking, cache, storage) and M3 (verifiers, disagreement, corrections, highlighting)."""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

from app import db
from app.errors import CorrectionError, TrustLensError, VerificationError
from app.models.schemas import VERDICTS, AnalysisResult
from app.services import m1_bridge
from app.services.claim_splitter import split_claims
from app.services.correction import check_correction, write_correction
from app.services.disagreement import resolve_disagreement
from app.services.evidence_ranking import rank_evidence
from app.services.evidence_retrieval import RetrievalError, retrieve_evidence
from app.services.highlighting import highlight_evidence
from app.services.semantic_cache import cache_lookup, cache_store, context_hash
from app.services.verification import VERIFIERS, verify_claim


log = logging.getLogger("trustlens.pipeline")
# free API tiers allow only a few requests a minute, so long answers are capped
MAX_CLAIMS = int(os.getenv("TRUSTLENS_MAX_CLAIMS", "6"))


def _judge(claim: dict, evidence: list[dict]) -> dict:
    results, failures = [], []
    for verifier in VERIFIERS:
        try:
            results.append(verify_claim(claim["text"], evidence, verifier, claim_id=claim["claim_id"]))
        except VerificationError as exc:
            failures.append(str(exc))
    if failures:
        # a verifier failed: never guess, send the claim to human review
        outcome = {"final_verdict": "unverifiable", "verifier_agreement": False, "needs_human_review": True}
    else:
        outcome = resolve_disagreement(results)
    decision = {
        "claim_id": claim["claim_id"],
        **outcome,
        "verifier_results": results,
        "self_consistency_score": None,
        "evidence_relevance_score": None,
        "correction_text": None,
        "correction_verified": None,
        "correction_support_score": None,
        "cache_hit": False,
    }
    if outcome["final_verdict"] == "contradicted":
        try:
            correction = write_correction(claim["text"], evidence)
        except CorrectionError:
            correction = None
        check = check_correction(correction, evidence)
        decision.update(correction_text=correction, correction_verified=check["verified"], correction_support_score=check["support_score"])
    return decision


def _trust(answer_id: str, decisions: list[dict], rows: list[dict]) -> dict:
    breakdown = {v: sum(d["final_verdict"] == v for d in decisions) for v in VERDICTS}
    n = len(decisions) or 1
    learned = m1_bridge.trust_scores(rows) if rows else None
    if learned is not None:
        return {
            "answer_id": answer_id,
            "trust_score": round(sum(learned) / len(learned), 3),
            "method": "m1_logreg_v1",
            "claim_breakdown": breakdown,
            "disclaimer": "Mean P(supported) from M1's logistic-regression model. The model is currently trained on simulated signals, so treat the number as indicative only; it is not a calibrated probability.",
        }
    score = (breakdown["supported"] + 0.5 * breakdown["unverifiable"]) / n
    return {
        "answer_id": answer_id,
        "trust_score": round(score, 2),
        "method": "placeholder_weighted",
        "claim_breakdown": breakdown,
        "disclaimer": "Simple score (supported = 1, unverifiable = 0.5, contradicted = 0) because self-consistency is not computed on this machine yet. Not a probability.",
    }


def analyze(question: str | None, answer_text: str | None) -> dict:
    answer_id = f"a_{uuid.uuid4().hex[:12]}"
    if answer_text:
        answer = answer_text.strip()
    else:
        answer, _ = asyncio.run(m1_bridge.generate(question or ""))

    claims = split_claims(answer, answer_id=answer_id)[:MAX_CLAIMS]
    log.warning("analysis %s: %d claims to check", answer_id, len(claims))
    consistency = asyncio.run(m1_bridge.self_consistency(question, [c["text"] for c in claims])) if question else None

    all_evidence, decisions, rows = [], [], []
    for i, claim in enumerate(claims):
        try:
            candidates = retrieve_evidence(claim["text"], claim_id=claim["claim_id"])
        except RetrievalError:
            candidates = []
        evidence = highlight_evidence(claim["text"], rank_evidence(claim["text"], candidates))
        relevance = max((e["combined_score"] for e in evidence), default=None)

        ctx = context_hash(evidence)
        cached = cache_lookup(claim["text"], ctx) if evidence else None
        if cached:
            decision = {**cached, "claim_id": claim["claim_id"], "cache_hit": True}
            for r in decision["verifier_results"]:
                r["claim_id"] = claim["claim_id"]
        else:
            decision = _judge(claim, evidence)
            if evidence and not decision["needs_human_review"]:
                cache_store(claim["text"], ctx, decision)

        decision["evidence_relevance_score"] = relevance
        decision["self_consistency_score"] = consistency[i] if consistency else None
        verdicts = {r["verifier"]: r["verdict"] for r in decision["verifier_results"]}
        if len(verdicts) == 2:
            rows.append(
                {
                    "claim_id": claim["claim_id"],
                    "self_consistency": decision["self_consistency_score"],
                    "evidence_relevance": relevance,
                    "agreement": decision["verifier_agreement"],
                    "verdict_a": verdicts["gemini"],
                    "verdict_b": verdicts["groq_qwen"],
                }
            )
        decisions.append(decision)
        all_evidence.extend(evidence)
        log.warning("analysis %s: claim %d/%d -> %s", answer_id, i + 1, len(claims), decision["final_verdict"])

    result = {
        "answer_id": answer_id,
        "question": question,
        "answer_text": answer,
        "claims": claims,
        "evidence": all_evidence,
        "decisions": decisions,
        "trust_score": _trust(answer_id, decisions, rows if len(rows) == len(decisions) else []),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    AnalysisResult.model_validate(result)
    db.save_analysis(result)
    return result


__all__ = ["analyze", "TrustLensError"]
