"""M2: the full TrustLens pipeline, wiring M1 (generation, self-consistency, trust model),
M2 (claims, evidence, ranking, cache, storage) and M3 (verifiers, disagreement, corrections, highlighting)."""

import asyncio
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import httpx

from app import db
from app.config import get_settings
from app.services._llm import LLMCallError, call_gemini, call_groq, note, parse_json_object, take_notes
from app.errors import CorrectionError, TrustLensError, VerificationError
from app.models.schemas import VERDICTS, AnalysisResult
from app.services import m1_bridge
from app.services.claim_splitter import split_claims
from app.services.correction import check_correction, write_correction
from app.services.disagreement import resolve_disagreement
from app.services.evidence_ranking import rank_evidence
from app.services.evidence_retrieval import HEADERS as WIKI_HEADERS, RetrievalError, retrieve_evidence
from app.services.highlighting import highlight_evidence
from app.services.semantic_cache import cache_lookup, cache_store, context_hash
from app.services.verification import VERIFIERS, verify_claim, verify_claims_batch


log = logging.getLogger("trustlens.pipeline")
# free API tiers allow only a few requests a minute, so long answers are capped
MAX_CLAIMS = int(os.getenv("TRUSTLENS_MAX_CLAIMS", "6"))


def _judge(claim: dict, evidence: list[dict], batched: dict[str, dict] | None = None, dead: set[str] | None = None) -> dict:
    results, failures = [], []
    batched = batched or {}
    dead = dead or set()

    def run(verifier):
        if verifier in batched:
            return batched[verifier], None
        if verifier in dead:
            # its whole batch call failed (usually the free limit): retrying claim by claim would only wait longer
            return None, f"{verifier} verifier unavailable"
        try:
            return verify_claim(claim["text"], evidence, verifier, claim_id=claim["claim_id"]), None
        except VerificationError as exc:
            return None, str(exc)

    # both verifiers are independent, so ask them at the same time
    with ThreadPoolExecutor(max_workers=len(VERIFIERS)) as pool:
        for result, failure in pool.map(run, VERIFIERS):
            if failure:
                failures.append(failure)
            else:
                results.append(result)
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


_RECENT = re.compile(r"\b(la\w{0,3}st|current|currently|recent|recently|newest|now|today|this year|present)\b", re.I)
_FILLER = re.compile(r"\b(tell me about|who is|who was|who won|what is|what was|what were|result of|results of|score of|the|la\w{0,3}st|current|currently|recent|recently|newest|now|today|winner|holder)\b", re.I)
# "best laptop", "who deserved", "should I": no single true answer exists, only facts around it
_OPINION_Q = re.compile(
    r"\b(best|worst|greatest|better|favou?rite|most beautiful|deserved?|deserves|should i|should we|overrated|underrated|goat)\b", re.I
)
_OPINION_RULES = (
    "This question asks for an opinion. Start with one sentence saying plainly that there is no single correct answer "
    "and it depends on the person's needs or criteria. Then give 3 to 4 short factual sentences naming commonly "
    "cited options with checkable facts (specs, awards, results, rankings), never stated as the one right answer.\n"
    "No greetings, lists or markdown."
)

_RULES = (
    "Answer the question in 3 to 5 short, plain factual sentences (under 90 words).\n"
    "Rules: answer directly; no greetings, apologies, disclaimers, advice, lists or markdown; "
    "each sentence states one checkable fact with names, numbers and years where relevant."
)


def _wiki_title(topic: str) -> str:
    """Wikipedia's own spelling of a topic ('Ballon dor' -> "Ballon d'Or"), so year pages are found."""
    try:
        r = httpx.get("https://en.wikipedia.org/w/api.php", params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"},
                      headers=WIKI_HEADERS, timeout=10)
        titles = r.json()[1]
        return titles[0] if titles else topic
    except (httpx.HTTPError, ValueError, IndexError):
        return topic


def _suggest_titles(question: str) -> list[str]:
    """Up to 3 Wikipedia article titles likely to hold the answer ('csk head coach' -> 'Chennai Super Kings').
    Search engines don't expand abbreviations or nicknames; a quick model call does."""
    settings = get_settings()
    if settings.mock or not settings.groq_api_key:
        return []
    prompt = ("List up to 3 exact English Wikipedia article titles most likely to contain the answer to the question. "
              "Expand abbreviations and nicknames (e.g. CSK -> Chennai Super Kings). Do not answer the question.\n"
              'Reply with JSON only: {"titles": ["..."]}\n\nQuestion: ' + question)
    try:
        data = parse_json_object(call_groq(prompt, os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), settings.groq_api_key, 15))
    except LLMCallError:
        return []
    titles = data.get("titles")
    return [str(t).strip() for t in titles if str(t).strip()][:3] if isinstance(titles, list) else []


def _background(question: str) -> list[dict]:
    """Wikipedia notes for the question. For 'latest/current' questions also search the
    year-specific pages (e.g. '2025 Ballon d'Or'), which name the most recent result."""
    # also search Wikipedia's own title for the topic: "who won the fifa 2026" -> "2026 FIFA World Cup"
    topic = _wiki_title(re.sub(r"\s+", " ", _FILLER.sub(" ", question)).strip(" ?."))
    queries = [question] + ([topic] if topic and topic.lower() not in question.lower() else [])
    if _RECENT.search(question):
        year = datetime.now(timezone.utc).year
        queries = [f"{year} {topic}", f"{year - 1} {topic}", question]
    # and the article titles a model suggests, which fixes abbreviations and loose phrasing
    queries += [t for t in _suggest_titles(question) if t.lower() not in {q.lower() for q in queries}]
    def search(q: str) -> list[dict]:
        try:
            return retrieve_evidence(q, claim_id="q", k=10)
        except RetrievalError:
            return []

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        results = list(pool.map(search, queries))
    notes: list[dict] = []
    seen: set[str] = set()
    for found in results:
        for n in found:
            if n["passage"] not in seen:
                seen.add(n["passage"])
                notes.append(n)
    # keep the passages most relevant to the question itself (keywords + meaning)
    return rank_evidence(question, notes, top_k=8) if notes else []


_search_off_until = 0.0  # Google Search grounding refused (free tier): don't retry it on every question


def _generate(question: str, notes: list[dict]) -> tuple[str, str]:
    """(answer, who wrote it). Recent questions: Gemini with Google Search first.
    Otherwise, or if that fails: M1's Groq generator with today's date and Wikipedia notes."""
    today = datetime.now(timezone.utc).strftime("%d %B %Y")
    global _search_off_until
    if _RECENT.search(question) and not get_settings().mock and time.monotonic() > _search_off_until:
        settings = get_settings()
        try:
            text = call_gemini(f"Today is {today}. Use Google Search for up-to-date facts.\n{_RULES}\n\nQuestion: {question}",
                               settings.gemini_flash_model, settings.gemini_api_key, settings.http_timeout, search=True).strip()
            if text:
                return text, f"{settings.gemini_flash_model} + Google Search"
        except LLMCallError as exc:
            _search_off_until = time.monotonic() + 900
            note(f"Gemini Google Search unavailable ({exc}); answered from Wikipedia notes instead.")
    background = "\n".join(f"- {n['passage']}" for n in notes) or "- (no background notes found)"
    rules = _OPINION_RULES if _OPINION_Q.search(question) else _RULES
    prompt = (f"Today is {today}.\n{rules}\nThe background notes are current; your memory may be out of date. "
              "When the notes name the answer (a winner, a result, a date), use exactly what they say. "
              "If the notes show a change over time (someone replaced, took over, was appointed, left), the most recent one is the answer. "
              "Each note starts with [the page it comes from]; 'the team' or 'the franchise' in a note means that page's subject, so never move a fact from one team, person or event to another. "
              "Answer the question directly as a normal assistant: never mention notes, sources, background "
              "or 'provided information'.\n\n"
              f"Background notes:\n{background}\n\nQuestion: {question}")
    answer, model = asyncio.run(m1_bridge.generate(prompt))
    return answer, f"{model} (M1 generator) + Wikipedia notes"


def analyze(question: str | None, answer_text: str | None) -> dict:
    answer_id = f"a_{uuid.uuid4().hex[:12]}"
    take_notes()  # start this analysis with a clean list
    source = None
    # passages found for the question itself; they are also given to every claim as evidence,
    # so the verifiers see the same pages the answer was written from
    notes = _background(question) if question and not get_settings().mock else []
    if answer_text:
        answer = answer_text.strip()
    else:
        answer, source = _generate(question or "", notes)

    claims = split_claims(answer, answer_id=answer_id)[:MAX_CLAIMS]
    log.warning("analysis %s: %d claims to check", answer_id, len(claims))
    consistency = asyncio.run(m1_bridge.self_consistency(question, [c["text"] for c in claims])) if question else None

    def prepare(claim: dict) -> tuple[list[dict], str, dict | None]:
        try:
            candidates = retrieve_evidence(claim["text"], claim_id=claim["claim_id"])
        except RetrievalError:
            candidates = []
        cid = claim["claim_id"]
        shared = [{**n, "claim_id": cid, "evidence_id": f"e_{cid[2:]}_q{j + 1}"} for j, n in enumerate(notes)]
        known = {c["passage"] for c in candidates}
        candidates += [n for n in shared if n["passage"] not in known]
        evidence = highlight_evidence(claim["text"], rank_evidence(claim["text"], candidates, top_k=5))
        ctx = context_hash(evidence)
        return evidence, ctx, cache_lookup(claim["text"], ctx) if evidence else None

    # 1. evidence for every claim, searched at the same time
    with ThreadPoolExecutor(max_workers=max(1, len(claims))) as pool:
        prepared = list(pool.map(prepare, claims))

    # 2. each verifier judges ALL uncached claims in one call: 2 calls per answer instead of 2 per claim
    todo = [(c["claim_id"], c["text"], ev) for c, (ev, _, cached) in zip(claims, prepared) if not cached]

    dead: set[str] = set()

    def batch(verifier: str) -> dict[str, dict]:
        try:
            return verify_claims_batch(todo, verifier)
        except VerificationError as exc:
            log.warning("batch %s failed: %s", verifier, exc)
            dead.add(verifier)
            note(f"{'Gemini' if verifier == 'gemini' else 'Qwen'} verifier was unavailable ({exc}); affected claims are marked for review.")
            return {}

    batched: dict[str, dict[str, dict]] = {}
    if todo:
        with ThreadPoolExecutor(max_workers=len(VERIFIERS)) as pool:
            batched = dict(zip(VERIFIERS, pool.map(batch, VERIFIERS)))

    # 2b. debate: where both verifiers answered but disagree, each sees the other's reason once and re-judges.
    # Only those claims are sent, in one call per verifier, so agreement costs nothing extra.
    debated: dict[str, dict[str, str]] = {}
    if len(batched) == len(VERIFIERS):
        a, b = VERIFIERS
        split = [t for t in todo if t[0] in batched[a] and t[0] in batched[b] and batched[a][t[0]]["verdict"] != batched[b][t[0]]["verdict"]]
        if split:
            first = {cid: {a: batched[a][cid]["verdict"], b: batched[b][cid]["verdict"]} for cid, _, _ in split}

            def rebut(pair: tuple[str, str]) -> dict[str, dict]:
                me, other = pair
                try:
                    return verify_claims_batch(split, me, peer={cid: batched[other][cid] for cid, _, _ in split})
                except VerificationError:
                    return {}

            with ThreadPoolExecutor(max_workers=2) as pool:
                second = dict(zip((a, b), pool.map(rebut, [(a, b), (b, a)])))
            for cid, _, _ in split:
                if cid in second[a] and cid in second[b]:
                    batched[a][cid], batched[b][cid] = second[a][cid], second[b][cid]
                    debated[cid] = first[cid]

    # 3. disagreement rule and corrections per claim; only verdicts missing from the batch cost extra calls
    def finish(i: int) -> tuple[dict, list[dict], dict | None]:
        claim = claims[i]
        evidence, ctx, cached = prepared[i]
        relevance = max((e["combined_score"] for e in evidence), default=None)
        if cached:
            decision = {**cached, "claim_id": claim["claim_id"], "cache_hit": True}
            for r in decision["verifier_results"]:
                r["claim_id"] = claim["claim_id"]
        else:
            got = {v: batched[v][claim["claim_id"]] for v in batched if claim["claim_id"] in batched[v]}
            decision = _judge(claim, evidence, got, dead)
            if claim["claim_id"] in debated:
                decision["first_verdicts"] = debated[claim["claim_id"]]
                decision["resolved_by_debate"] = decision["verifier_agreement"]
            if evidence and not decision["needs_human_review"]:
                cache_store(claim["text"], ctx, decision)

        decision["evidence_relevance_score"] = relevance
        decision["self_consistency_score"] = consistency[i] if consistency else None
        verdicts = {r["verifier"]: r["verdict"] for r in decision["verifier_results"]}
        row = None
        if len(verdicts) == 2:
            row = {
                "claim_id": claim["claim_id"],
                "self_consistency": decision["self_consistency_score"],
                "evidence_relevance": relevance,
                "agreement": decision["verifier_agreement"],
                "verdict_a": verdicts["gemini"],
                "verdict_b": verdicts["groq_qwen"],
            }
        log.warning("analysis %s: claim %d/%d -> %s", answer_id, i + 1, len(claims), decision["final_verdict"])
        return decision, evidence, row

    with ThreadPoolExecutor(max_workers=max(1, len(claims))) as pool:
        done = list(pool.map(finish, range(len(claims))))
    all_evidence, decisions, rows = [], [], []
    for decision, evidence, row in done:
        decisions.append(decision)
        all_evidence.extend(evidence)
        if row:
            rows.append(row)

    result = {
        "answer_id": answer_id,
        "question": question,
        "answer_text": answer,
        "claims": claims,
        "evidence": all_evidence,
        "decisions": decisions,
        "trust_score": _trust(answer_id, decisions, rows if len(rows) == len(decisions) else []),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "answer_source": source,
        "notes": take_notes(),
        "question_type": "opinion" if question and _OPINION_Q.search(question) else "factual",
    }
    AnalysisResult.model_validate(result)
    db.save_analysis(result)
    return result


__all__ = ["analyze", "TrustLensError"]
