"""M3: mark the evidence sentence that best matches each claim."""

from app.services._text import cosine, embed, split_sentences

HIGHLIGHT_MIN_SCORE = 0.15


def _best_sentence(claim_text: str, passage: str) -> tuple[int, int, str, float] | None:
    sentences = split_sentences(passage)
    if not sentences:
        return None
    vectors = embed([claim_text] + [s[2] for s in sentences])
    scores = [cosine(vectors[0], v) for v in vectors[1:]]
    i = max(range(len(scores)), key=scores.__getitem__)
    start, end, text = sentences[i]
    return start, end, text, scores[i]


def best_evidence_sentence(claim_text: str, evidence: list[dict]) -> tuple[str, str, float] | None:
    """(evidence_id, sentence, score) of the single best-matching sentence across all passages."""
    best: tuple[str, str, float] | None = None
    for ev in evidence:
        found = _best_sentence(claim_text, ev["passage"])
        if found and (best is None or found[3] > best[2]):
            best = (ev["evidence_id"], found[2], found[3])
    return best


def highlight_evidence(claim_text: str, evidence: list[dict]) -> list[dict]:
    """Copy of `evidence` with highlight_span / highlight_score filled in. Order is preserved."""
    out = []
    for ev in evidence:
        item = dict(ev)
        found = _best_sentence(claim_text, ev["passage"])
        if found is None:
            item["highlight_span"], item["highlight_score"] = None, None
        else:
            start, end, _, score = found
            item["highlight_score"] = round(score, 4)
            item["highlight_span"] = {"start": start, "end": end} if score >= HIGHLIGHT_MIN_SCORE else None
        out.append(item)
    return out
