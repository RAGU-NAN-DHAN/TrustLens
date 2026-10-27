"""M2: split an answer into atomic factual claims, each tied to the span of the answer it came from."""

import re

from app.config import get_settings
from app.errors import TrustLensError
from app.services._llm import LLMCallError, call_gemini, parse_json_object
from app.services._text import cosine, embed, split_sentences


class ClaimSplitError(TrustLensError):
    error_code = "CLAIM_SPLIT_FAILED"


_PROMPT = """Split the ANSWER into atomic factual claims that can each be checked against an encyclopedia.

Rules:
- One fact per claim. Split "X was born in 1931 in Rameswaram" into two claims.
- Replace pronouns with the name they refer to, so every claim stands alone.
- Skip opinions, advice, greetings and questions.
- Keep the wording close to the answer. Do not add facts.
- Every claim must be a complete, unambiguous sentence that is true or false on its own,
  with the full subject and the full name of the thing (e.g. "the Formula One World Drivers' Championship").
- Never make a claim vaguer than the answer. A list like "won titles in 2008, 2014 and 2015" becomes
  "Lewis Hamilton won the Formula One World Drivers' Championship in 2008." and so on, never
  "Lewis Hamilton's championships were secured in 2008."
- At most 6 claims: keep the most important facts; fold minor details into the claim they belong to.

Reply with JSON only: {"claims": [{"text": "<claim>", "sentence": "<the exact sentence of the ANSWER it comes from>"}]}

ANSWER:
<<ANSWER>>"""

_OPINION = re.compile(r"\b(i think|i believe|in my opinion|you should|let me know|hope this helps)\b", re.I)


def _span_for(sentence: str, answer: str, sentences: list[tuple[int, int, str]]) -> tuple[int, int]:
    start = answer.find(sentence.strip()) if sentence else -1
    if start != -1:
        return start, start + len(sentence.strip())
    if not sentences:
        return 0, len(answer)
    vectors = embed([sentence] + [s[2] for s in sentences])
    best = max(range(len(sentences)), key=lambda i: cosine(vectors[0], vectors[i + 1]))
    return sentences[best][0], sentences[best][1]


def split_claims(answer_text: str, answer_id: str = "", mock: bool | None = None) -> list[dict]:
    """Return Claim dicts. Mock mode (no API key needed) uses one claim per sentence."""
    answer = answer_text.strip()
    if not answer:
        return []
    sentences = split_sentences(answer)
    settings = get_settings()
    use_mock = settings.mock if mock is None else mock

    pairs: list[tuple[str, str]]
    if use_mock:
        pairs = [(s[2], s[2]) for s in sentences if not _OPINION.search(s[2])]
    else:
        prompt = _PROMPT.replace("<<ANSWER>>", answer)
        try:
            data = parse_json_object(call_gemini(prompt, settings.gemini_flash_model, settings.gemini_api_key, settings.http_timeout))
            items = data.get("claims")
        except LLMCallError:
            items = None
        pairs = []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and str(item.get("text", "")).strip():
                pairs.append((str(item["text"]).strip(), str(item.get("sentence", "")).strip()))
        if not pairs:
            # Gemini busy or unusable reply: fall back to one claim per factual sentence rather than failing the check
            pairs = [(s[2], s[2]) for s in sentences if not _OPINION.search(s[2])]

    claims = []
    for i, (text, sentence) in enumerate(pairs, start=1):
        start, end = _span_for(sentence, answer, sentences)
        claims.append(
            {
                "claim_id": f"c_{i:04d}",
                "answer_id": answer_id,
                "text": text,
                "span": {"start": start, "end": end},
                "claim_type": "factual",
                "importance": "normal",
            }
        )
    return claims
