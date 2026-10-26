"""M2: reuse a verdict only for a near-identical claim checked against the same evidence (never similarity alone)."""

import hashlib
import json

from app.db import connect
from app.services._text import cosine, embed

SIMILARITY_THRESHOLD = 0.95


def context_hash(evidence: list[dict]) -> str:
    """Hash of the evidence actually shown to the verifiers."""
    parts = sorted(f"{e['url']}\n{e['passage']}" for e in evidence)
    return hashlib.sha256("\n\n".join(parts).encode("utf-8")).hexdigest()


def cache_lookup(claim_text: str, ctx_hash: str) -> dict | None:
    vector = embed([claim_text])[0]
    with connect() as conn:
        rows = conn.execute("SELECT embedding, decision FROM claim_cache WHERE context_hash = ?", (ctx_hash,)).fetchall()
    best, best_score = None, SIMILARITY_THRESHOLD
    for emb, decision in rows:
        score = cosine(vector, json.loads(emb))
        if score >= best_score:
            best, best_score = json.loads(decision), score
    return best


def cache_store(claim_text: str, ctx_hash: str, decision: dict) -> None:
    vector = embed([claim_text])[0]
    with connect() as conn:
        conn.execute(
            "INSERT INTO claim_cache (claim_text, context_hash, embedding, decision) VALUES (?, ?, ?, ?)",
            (claim_text, ctx_hash, json.dumps(vector), json.dumps(decision)),
        )
