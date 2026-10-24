"""M3: the single rule for combining the two verifiers. Nobody else re-implements it."""


def resolve_disagreement(results: list[dict]) -> dict:
    """Same verdict -> keep it. Different verdicts -> unverifiable and flag for human review."""
    if len(results) < 2:
        raise ValueError("resolve_disagreement needs results from both verifiers")
    verdicts = {r["verdict"] for r in results}
    if len(verdicts) == 1:
        return {"final_verdict": verdicts.pop(), "verifier_agreement": True, "needs_human_review": False}
    return {"final_verdict": "unverifiable", "verifier_agreement": False, "needs_human_review": True}
