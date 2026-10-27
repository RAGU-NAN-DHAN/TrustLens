"""M3: two independent verifiers judge a claim against its evidence.

verifier="gemini"    -> Gemini (GEMINI_VERIFIER_MODEL, default gemini-2.5-pro)
verifier="groq_qwen" -> a non-Llama model on Groq (GROQ_VERIFIER_MODEL)
"""

from app.config import get_settings
from app.errors import VerificationError
from app.models.schemas import VERDICTS
from app.services._llm import LLMCallError, call_gemini, call_groq, parse_json_object
from app.services._text import numbers
from app.services.highlighting import best_evidence_sentence

VERIFIERS = ("gemini", "groq_qwen")

_PROMPT = """You are a strict fact-checking judge.
Decide whether the CLAIM is supported by the EVIDENCE passages.

Rules:
- Use ONLY the evidence text. Do not use your own knowledge, even if you are sure.
- "supported": the evidence explicitly states the claim, and every detail (names, numbers, dates) matches.
- "contradicted": the evidence explicitly states something incompatible with the claim.
- "unverifiable": the evidence neither confirms nor contradicts the claim.
- If unsure, answer "unverifiable". Never answer "supported" on a partial match.
- Judge what the claim plainly means, as a normal reader would; awkward wording alone is not a contradiction.
- "contradicted" needs evidence that states something different; evidence that is merely silent means "unverifiable".

Reply with JSON only, no other text:
{"verdict": "supported" | "contradicted" | "unverifiable",
 "confidence": <number from 0 to 1>,
 "explanation": "<one sentence citing the evidence id>",
 "evidence_ids": ["<ids of the passages you relied on>"]}

CLAIM:
<<CLAIM>>

EVIDENCE:
<<EVIDENCE>>"""


_BATCH_PROMPT = """You are a strict fact-checking judge.
For EACH claim below, decide whether it is supported by the evidence passages listed under it.

Rules:
- Judge every claim ONLY against its own evidence. Do not use your own knowledge, even if you are sure.
- "supported": the evidence explicitly states the claim, and every detail (names, numbers, dates) matches.
- "contradicted": the evidence explicitly states something incompatible with the claim.
- "unverifiable": the evidence neither confirms nor contradicts the claim.
- If unsure, answer "unverifiable". Never answer "supported" on a partial match.
- Judge what the claim plainly means, as a normal reader would; awkward wording alone is not a contradiction.
- "contradicted" needs evidence that states something different; evidence that is merely silent means "unverifiable".

Reply with JSON only, no other text, one entry per claim:
{"results": [{"claim_id": "<id>",
              "verdict": "supported" | "contradicted" | "unverifiable",
              "confidence": <number from 0 to 1>,
              "explanation": "<one sentence citing the evidence id>",
              "evidence_ids": ["<ids of the passages you relied on>"]}]}

<<CLAIMS>>"""


def format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "(no evidence passages were found)"
    return "\n\n".join(f"[{ev['evidence_id']}] ({ev.get('source', 'unknown')}) {ev['passage']}" for ev in evidence)


def build_prompt(claim_text: str, evidence: list[dict]) -> str:
    return _PROMPT.replace("<<CLAIM>>", claim_text).replace("<<EVIDENCE>>", format_evidence(evidence))


def verify_claim(
    claim_text: str,
    evidence: list[dict],
    verifier: str,
    mock: bool | None = None,
    claim_id: str | None = None,
) -> dict:
    """Return a VerificationResult dict. Raises VerificationError instead of ever guessing a verdict."""
    if verifier not in VERIFIERS:
        raise VerificationError(f"unknown verifier {verifier!r}; expected one of {VERIFIERS}")
    settings = get_settings()
    use_mock = settings.mock if mock is None else mock
    cid = claim_id if claim_id is not None else (evidence[0]["claim_id"] if evidence else "")

    if use_mock:
        verdict, confidence, explanation, ids = _mock_judgement(claim_text, evidence, strict=verifier == "groq_qwen")
        model = f"mock-{verifier}"
    else:
        model, raw = _call_real(verifier, build_prompt(claim_text, evidence), settings)
        verdict, confidence, explanation, ids = _parse_judgement(raw, evidence, verifier)

    return {
        "claim_id": cid,
        "verifier": verifier,
        "verdict": verdict,
        "confidence": confidence,
        "explanation": explanation,
        "evidence_ids": ids,
        "model_version": model,
    }


def verify_claims_batch(items: list[tuple[str, str, list[dict]]], verifier: str, peer: dict[str, dict] | None = None) -> dict[str, dict]:
    """Judge many claims in ONE model call (free tiers allow few calls a minute).
    items = (claim_id, claim_text, evidence). Returns results by claim_id; claims the
    model skipped or answered badly are left out so the caller can retry them one by one."""
    if verifier not in VERIFIERS:
        raise VerificationError(f"unknown verifier {verifier!r}; expected one of {VERIFIERS}")
    settings = get_settings()
    if settings.mock or not items:
        return {cid: verify_claim(text, ev, verifier, claim_id=cid) for cid, text, ev in items}

    def block(cid: str, text: str, ev: list[dict]) -> str:
        out = f"CLAIM {cid}:\n{text}\nEVIDENCE for {cid}:\n{format_evidence(ev)}"
        if peer and cid in peer:
            # debate round: show the other checker's verdict and reason once
            out += (f"\nANOTHER INDEPENDENT CHECKER said \"{peer[cid]['verdict']}\" because: {peer[cid]['explanation']}\n"
                    "Reconsider using ONLY the evidence. Change your verdict only if their reason is correct; otherwise keep it.")
        return out

    blocks = "\n\n".join(block(cid, text, ev) for cid, text, ev in items)
    model, raw = _call_real(verifier, _BATCH_PROMPT.replace("<<CLAIMS>>", blocks), settings)
    try:
        rows = parse_json_object(raw).get("results")
    except LLMCallError as exc:
        raise VerificationError(f"{verifier} verifier: {exc}") from exc
    if not isinstance(rows, list):
        raise VerificationError(f"{verifier} verifier returned no 'results' list")

    by_id = {cid: ev for cid, _, ev in items}
    out: dict[str, dict] = {}
    for row in rows:
        cid = str(row.get("claim_id", "")).strip() if isinstance(row, dict) else ""
        if cid not in by_id or cid in out:
            continue
        try:
            verdict, confidence, explanation, ids = _check_judgement(row, by_id[cid], verifier)
        except VerificationError:
            continue
        out[cid] = {
            "claim_id": cid,
            "verifier": verifier,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation,
            "evidence_ids": ids,
            "model_version": model,
        }
    return out


def _call_real(verifier: str, prompt: str, settings) -> tuple[str, str]:
    try:
        if verifier == "gemini":
            model = settings.gemini_verifier_model
            try:
                return model, call_gemini(prompt, model, settings.gemini_api_key, settings.http_timeout)
            except LLMCallError:
                # main Gemini model busy or out of free quota: use the lighter Gemini model, still an independent verifier
                if settings.gemini_flash_model in ("", model):
                    raise
                model = settings.gemini_flash_model
                return model, call_gemini(prompt, model, settings.gemini_api_key, settings.http_timeout)
        model = settings.groq_verifier_model
        if not model:
            raise VerificationError("GROQ_VERIFIER_MODEL is not set; pick a current non-Llama model on console.groq.com")
        if "llama" in model.lower():
            raise VerificationError("verifier 2 must not be a Llama model: Llama wrote the answer being checked")
        return model, call_groq(prompt, model, settings.groq_api_key, settings.http_timeout)
    except LLMCallError as exc:
        raise VerificationError(f"{verifier} verifier: {exc}") from exc


def _parse_judgement(raw: str, evidence: list[dict], verifier: str) -> tuple[str, float, str, list[str]]:
    try:
        data = parse_json_object(raw)
    except LLMCallError as exc:
        raise VerificationError(f"{verifier} verifier: {exc}") from exc
    return _check_judgement(data, evidence, verifier)


def _check_judgement(data: dict, evidence: list[dict], verifier: str) -> tuple[str, float, str, list[str]]:
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise VerificationError(f"{verifier} verifier returned an invalid verdict {data.get('verdict')!r}")
    try:
        confidence = float(data["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError(f"{verifier} verifier returned no usable confidence") from exc

    known = {ev["evidence_id"] for ev in evidence}
    ids = [i for i in data.get("evidence_ids") or [] if i in known]
    explanation = str(data.get("explanation", "")).strip() or "No explanation given."
    return verdict, round(min(max(confidence, 0.0), 1.0), 2), explanation, ids


def _mock_judgement(claim_text: str, evidence: list[dict], strict: bool) -> tuple[str, float, str, list[str]]:
    """Deterministic offline stand-in: word overlap plus a number check. Not a real verifier."""
    best = best_evidence_sentence(claim_text, evidence)
    if best is None:
        return "unverifiable", 0.5, "No evidence passages were provided.", []
    evidence_id, sentence, score = best
    threshold = 0.45 if strict else 0.30
    confidence = round(min(0.95, 0.5 + score / 2), 2)
    if score < threshold:
        return "unverifiable", confidence, f"No passage clearly addresses the claim (best match {evidence_id}).", []
    claim_nums, sentence_nums = numbers(claim_text), numbers(sentence)
    if claim_nums and sentence_nums and not claim_nums <= sentence_nums:
        missing = ", ".join(sorted(claim_nums - sentence_nums))
        return "contradicted", confidence, f"{evidence_id} gives different figures (claim says {missing}).", [evidence_id]
    return "supported", confidence, f"{evidence_id} states the same fact.", [evidence_id]
