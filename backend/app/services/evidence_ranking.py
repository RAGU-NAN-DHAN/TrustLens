"""M2: rank evidence by BM25 keyword overlap plus embedding similarity, keep the top passages."""

import math

from app.services._text import cosine, embed, tokens

K1 = 1.5
B = 0.75


def _bm25(query: list[str], docs: list[list[str]]) -> list[float]:
    n = len(docs)
    avg = sum(len(d) for d in docs) / n if n else 0
    scores = []
    for d in docs:
        score = 0.0
        for term in set(query):
            df = sum(1 for doc in docs if term in doc)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = d.count(term)
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * len(d) / (avg or 1)))
        scores.append(score)
    return scores


def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    return [0.0 if hi == lo else (v - lo) / (hi - lo) for v in values]


def rank_evidence(claim_text: str, evidence: list[dict], alpha: float = 0.6, beta: float = 0.4, top_k: int = 3) -> list[dict]:
    """combined = alpha * semantic + beta * keyword (both min-max normalised). Stable for ties."""
    if not evidence:
        return []
    keyword = _bm25(tokens(claim_text), [tokens(e["passage"]) for e in evidence])
    vectors = embed([claim_text] + [e["passage"] for e in evidence])
    semantic = [cosine(vectors[0], v) for v in vectors[1:]]
    kn, sn = _normalize(keyword), _normalize(semantic)
    scored = []
    for i, e in enumerate(evidence):
        item = dict(e)
        item["keyword_score"] = round(keyword[i], 4)
        item["semantic_score"] = round(semantic[i], 4)
        item["combined_score"] = round(alpha * sn[i] + beta * kn[i], 4)
        scored.append((item["combined_score"], -i, item))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item for _, _, item in scored[:top_k]]
