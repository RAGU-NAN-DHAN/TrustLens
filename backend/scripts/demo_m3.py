"""Run M3's part of the pipeline on the demo answer and write the UI's sample analysis.

    python scripts/demo_m3.py            # mock verifiers, no API keys needed
    python scripts/demo_m3.py --real     # real Gemini + Groq calls (keys in backend/.env)

Generation, claim splitting, retrieval and the trust score belong to M1/M2.
Until they are merged this script uses hand-written claims and evidence and a
placeholder trust score, and says so in the output.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.errors import CorrectionError, VerificationError  # noqa: E402
from app.models.schemas import AnalysisResult  # noqa: E402
from app.services.correction import check_correction, write_correction  # noqa: E402
from app.services.disagreement import resolve_disagreement  # noqa: E402
from app.services.highlighting import highlight_evidence  # noqa: E402
from app.services.verification import VERIFIERS, verify_claim  # noqa: E402

DEFAULT_OUT = ROOT.parent / "frontend" / "src" / "mocks" / "sampleAnalysis.json"


def judge(claim: dict, evidence: list[dict], mock: bool) -> dict:
    results, failures = [], []
    for verifier in VERIFIERS:
        try:
            results.append(verify_claim(claim["text"], evidence, verifier, mock=mock, claim_id=claim["claim_id"]))
        except VerificationError as exc:
            failures.append(str(exc))

    if failures:
        # a verifier failed: do not guess, send the claim to human review
        print(f"  {claim['claim_id']}: " + "; ".join(failures), file=sys.stderr)
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
            correction = write_correction(claim["text"], evidence, mock=mock)
        except CorrectionError as exc:
            print(f"  {claim['claim_id']}: {exc}", file=sys.stderr)
            correction = None
        check = check_correction(correction, evidence)
        decision.update(
            correction_text=correction,
            correction_verified=check["verified"],
            correction_support_score=check["support_score"],
        )
    return decision


def placeholder_trust(answer_id: str, decisions: list[dict]) -> dict:
    breakdown = {v: sum(d["final_verdict"] == v for d in decisions) for v in ("supported", "contradicted", "unverifiable")}
    n = len(decisions) or 1
    score = (breakdown["supported"] + 0.5 * breakdown["unverifiable"]) / n
    return {
        "answer_id": answer_id,
        "trust_score": round(score, 2),
        "method": "placeholder_until_m1",
        "claim_breakdown": breakdown,
        "disclaimer": "Placeholder score (supported = 1, unverifiable = 0.5, contradicted = 0) until M1's trained trust model is merged. Not a probability.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="call Gemini and Groq instead of the mock verifiers")
    parser.add_argument("--data", type=Path, default=ROOT / "scripts" / "demo_data.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    mock = not args.real

    data = json.loads(args.data.read_text(encoding="utf-8"))
    answer_text, answer_id = data["answer_text"], data["answer_id"]
    claims, all_evidence, decisions = [], [], []

    for c in data["claims"]:
        start = answer_text.find(c["text"])
        if start == -1:
            raise SystemExit(f"{c['claim_id']} text is not in answer_text")
        claim = {
            "claim_id": c["claim_id"],
            "answer_id": answer_id,
            "text": c["text"],
            "span": {"start": start, "end": start + len(c["text"])},
            "claim_type": "factual",
            "importance": "normal",
        }
        evidence = [
            {
                "evidence_id": f"e_{c['claim_id'][2:]}_{i + 1}",
                "claim_id": c["claim_id"],
                "source": ev["source"],
                "url": ev["url"],
                "passage": ev["passage"],
                "retrieved_at": None,
            }
            for i, ev in enumerate(c["evidence"])
        ]
        evidence = highlight_evidence(claim["text"], evidence)
        print(f"{claim['claim_id']}: judging ({'mock' if mock else 'real APIs'})")
        decisions.append(judge(claim, evidence, mock))
        claims.append(claim)
        all_evidence.extend(evidence)

    result = {
        "answer_id": answer_id,
        "question": data["question"],
        "answer_text": answer_text,
        "claims": claims,
        "evidence": all_evidence,
        "decisions": decisions,
        "trust_score": placeholder_trust(answer_id, decisions),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    AnalysisResult.model_validate(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for d in decisions:
        flag = "  needs review" if d["needs_human_review"] else ""
        print(f"  {d['claim_id']}: {d['final_verdict']}{flag}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
