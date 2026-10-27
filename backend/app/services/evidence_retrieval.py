"""M2: fetch candidate evidence passages for a claim from Wikipedia, with Brave Search as an optional fallback."""

import os
import re
from datetime import datetime, timezone

import httpx

from app.config import get_settings
from app.errors import TrustLensError
from app.services._text import split_sentences, tokens

WIKI_API = "https://en.wikipedia.org/w/api.php"
BRAVE_API = "https://api.search.brave.com/res/v1/web/search"
HEADERS = {"User-Agent": "TrustLens/0.1 (https://github.com/RAGU-NAN-DHAN/TrustLens; college fact-checking project) python-httpx"}
PASSAGE_SENTENCES = 3
_OUTCOME = re.compile(r"(who won|winner|won|champion|result|final|beat|defeated)", re.I)


class RetrievalError(TrustLensError):
    error_code = "EVIDENCE_RETRIEVAL_FAILED"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _chunk(text: str) -> list[str]:
    sentences = [s[2] for s in split_sentences(text)]
    return [" ".join(sentences[i:i + PASSAGE_SENTENCES]) for i in range(0, len(sentences), PASSAGE_SENTENCES)]


def _wikipedia(query: str, pages: int, timeout: float) -> list[dict]:
    found = httpx.get(
        WIKI_API,
        params={"action": "query", "list": "search", "srsearch": query, "srlimit": pages, "format": "json"},
        headers=HEADERS,
        timeout=timeout,
    )
    found.raise_for_status()
    ids = [str(r["pageid"]) for r in found.json().get("query", {}).get("search", [])]
    if not ids:
        return []
    extracts = httpx.get(
        WIKI_API,
        params={"action": "query", "prop": "extracts|info", "inprop": "url", "explaintext": 1, "exintro": 1, "pageids": "|".join(ids), "format": "json"},
        headers=HEADERS,
        timeout=timeout,
    )
    extracts.raise_for_status()
    out = []
    pages_by_id = extracts.json().get("query", {}).get("pages", {})
    for pid in ids:
        page = pages_by_id.get(pid, {})
        for passage in _chunk(page.get("extract", "")):
            # label each passage with its page: "the franchise" means nothing once cut out of "Chennai Super Kings"
            out.append({"source": "wikipedia", "url": page.get("fullurl", f"https://en.wikipedia.org/?curid={pid}"), "passage": f"[{page.get('title', 'Wikipedia')}] {passage}"})
    # read the full text of the top two pages: search order varies, and the answer is often in the body
    for pid in ids[:2]:
        out.extend(_body_passages(query, pid, pages_by_id.get(pid, {}), timeout))
    return out


def _body_passages(query: str, pid: str, page: dict, timeout: float, keep: int = 6) -> list[dict]:
    """Details such as lists of years usually sit in the article body, not the intro.
    Fetch the top page's full text and keep the passages that share the most words with the query."""
    try:
        full = httpx.get(
            WIKI_API,
            params={"action": "query", "prop": "extracts", "explaintext": 1, "pageids": pid, "format": "json"},
            headers=HEADERS,
            timeout=timeout,
        )
        full.raise_for_status()
        text = full.json().get("query", {}).get("pages", {}).get(pid, {}).get("extract", "")
    except (httpx.HTTPError, ValueError):
        return []
    want = set(tokens(query))
    if _OUTCOME.search(query):
        # "who won ..." questions: the passage that names the result rarely repeats the question's words
        want |= {"won", "winner", "winners", "champion", "champions", "final", "defeated", "beat", "title"}
    if not want or not text:
        return []
    chunks = _chunk(text)
    scored = sorted(((len(want & set(tokens(c))), i, c) for i, c in enumerate(chunks)), reverse=True)
    url = page.get("fullurl", f"https://en.wikipedia.org/?curid={pid}")
    return [{"source": "wikipedia", "url": url, "passage": f"[{page.get('title', 'Wikipedia')}] {c}"} for n, _, c in scored[:keep] if n > 0]


def _brave(query: str, key: str, timeout: float) -> list[dict]:
    resp = httpx.get(
        BRAVE_API,
        params={"q": query, "count": 5},
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    results = resp.json().get("web", {}).get("results", [])
    return [{"source": "brave", "url": r.get("url", ""), "passage": r.get("description", "")} for r in results if r.get("description")]


def retrieve_evidence(claim_text: str, claim_id: str = "", k: int = 14, mock: bool | None = None) -> list[dict]:
    """Up to k unranked Evidence dicts. Raises RetrievalError only if every source fails."""
    settings = get_settings()
    if settings.mock if mock is None else mock:
        return []
    brave_key = os.getenv("BRAVE_API_KEY", "").strip()
    raw: list[dict] = []
    failures = []
    try:
        raw.extend(_wikipedia(claim_text, pages=3, timeout=settings.http_timeout))
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        failures.append(f"wikipedia: {type(exc).__name__}")
    if brave_key and len(raw) < 2:
        try:
            raw.extend(_brave(claim_text, brave_key, settings.http_timeout))
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            failures.append(f"brave: {type(exc).__name__}")
    if not raw and failures:
        raise RetrievalError("; ".join(failures))

    stamp = _now()
    return [
        {
            "evidence_id": f"e_{claim_id[2:] if claim_id.startswith('c_') else claim_id}_{i + 1}",
            "claim_id": claim_id,
            "source": r["source"],
            "url": r["url"],
            "passage": r["passage"],
            "retrieved_at": stamp,
        }
        for i, r in enumerate(raw[:k])
    ]
