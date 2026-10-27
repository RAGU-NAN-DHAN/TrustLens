"""M3: write a correction for a contradicted claim, then check it is backed by the evidence."""

from app.config import get_settings
from app.errors import CorrectionError
from app.services._llm import LLMCallError, call_gemini, parse_json_object
from app.services._text import cosine, embed, numbers, split_sentences
from app.services.highlighting import best_evidence_sentence
from app.services.verification import format_evidence

SUPPORT_THRESHOLD = 0.5
MOCK_MATCH_THRESHOLD = 0.3

_PROMPT = """A fact-checker found this CLAIM to be false.
Using ONLY the EVIDENCE passages, rewrite the CLAIM so it is true.
- Keep the claim's subject and form; change only what is wrong.
- The correction must be about the same subject as the claim (same team, person, event). If the evidence only has facts about something else, return an empty string.
  e.g. CLAIM "Argentina won the 2026 FIFA World Cup" -> "<the team the evidence names> won the 2026 FIFA World Cup."
- Do not add any fact, number or name that is not in the evidence.
- If the evidence does not state the true fact, return an empty string.

Reply with JSON only: {"correction": "<sentence or empty string>"}

CLAIM:
<<CLAIM>>

EVIDENCE:
<<EVIDENCE>>"""


def write_correction(claim_text: str, evidence: list[dict], mock: bool | None = None) -> str | None:
    if not evidence:
        return None
    settings = get_settings()
    if settings.mock if mock is None else mock:
        best = best_evidence_sentence(claim_text, evidence)
        return best[1] if best and best[2] >= MOCK_MATCH_THRESHOLD else None

    prompt = _PROMPT.replace("<<CLAIM>>", claim_text).replace("<<EVIDENCE>>", format_evidence(evidence))
    try:
        data = parse_json_object(
            call_gemini(prompt, settings.gemini_flash_model, settings.gemini_api_key, settings.http_timeout)
        )
    except LLMCallError as exc:
        raise CorrectionError(f"correction writer: {exc}") from exc
    text = str(data.get("correction", "")).strip()
    return text or None


def check_correction(correction_text: str | None, evidence: list[dict]) -> dict:
    """Verified only if the correction closely matches an evidence sentence and every number in it appears in the evidence."""
    if not correction_text or not evidence:
        return {"verified": False, "support_score": 0.0}

    sentences = [s[2] for ev in evidence for s in split_sentences(ev["passage"])]
    if not sentences:
        return {"verified": False, "support_score": 0.0}
    vectors = embed([correction_text] + sentences)
    score = max(cosine(vectors[0], v) for v in vectors[1:])

    evidence_numbers = set().union(*(numbers(ev["passage"]) for ev in evidence))
    numbers_backed = numbers(correction_text) <= evidence_numbers
    return {"verified": score >= SUPPORT_THRESHOLD and numbers_backed, "support_score": round(score, 4)}
