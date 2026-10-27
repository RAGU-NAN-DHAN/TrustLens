"""Minimal HTTP clients for Gemini and Groq. Keys are sent in headers, never in URLs or logs."""

import json
import re
import threading
import time

import httpx

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_WAIT = 6.0  # free tiers allow only a few requests per minute; keep waits short so a check stays fast


def _post(url: str, headers: dict, body: dict, timeout: float) -> httpx.Response:
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code not in (429, 500, 502, 503, 504) or attempt == RATE_LIMIT_RETRIES:
            return resp
        time.sleep(RATE_LIMIT_WAIT * (attempt + 1))
    return resp


class LLMCallError(Exception):
    pass


_events: list[str] = []
_events_lock = threading.Lock()


def note(event: str) -> None:
    """Record something worth showing on the result (e.g. a key switch)."""
    with _events_lock:
        if event not in _events:
            _events.append(event)


def take_notes() -> list[str]:
    with _events_lock:
        out = list(_events)
        _events.clear()
    return out


def _post_failover(url: str, keys: list[str], body: dict, timeout: float, patient: bool = True) -> httpx.Response:
    """Try each key once; a key that is out of quota or busy hands over to the next immediately.
    Only the last key gets the slower retry loop."""
    for i, key in enumerate(keys[:-1]):
        resp = httpx.post(url, headers={"x-goog-api-key": key}, json=body, timeout=timeout)
        if resp.status_code not in (429, 500, 502, 503, 504):
            return resp
        note(f"Gemini key {i + 1} hit its limit (HTTP {resp.status_code}); switched to backup key {i + 2}.")
    if not patient:  # optional extras (Google Search) give up at once instead of waiting out the limit
        return httpx.post(url, headers={"x-goog-api-key": keys[-1]}, json=body, timeout=timeout)
    return _post(url, {"x-goog-api-key": keys[-1]}, body, timeout)


def call_gemini(prompt: str, model: str, api_key: str, timeout: float, search: bool = False) -> str:
    """api_key may be a comma-separated list of keys. search=True lets Gemini use Google Search
    (Gemini then answers in plain text, so JSON mode is off)."""
    keys = [k.strip() for k in api_key.split(",") if k.strip()]
    if not keys:
        raise LLMCallError("GEMINI_API_KEY is not set")
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}}
    if search:
        body["tools"] = [{"google_search": {}}]
    else:
        body["generationConfig"]["responseMimeType"] = "application/json"
    try:
        resp = _post_failover(GEMINI_URL.format(model=model), keys, body, timeout, patient=not search)
        resp.raise_for_status()
        parts = resp.json()["candidates"][0]["content"]["parts"]
    except httpx.HTTPStatusError as exc:
        raise LLMCallError(f"Gemini returned HTTP {exc.response.status_code}") from exc
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise LLMCallError(f"Gemini call failed: {type(exc).__name__}") from exc
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def call_groq(prompt: str, model: str, api_key: str, timeout: float) -> str:
    if not api_key:
        raise LLMCallError("GROQ_API_KEY is not set")
    body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    try:
        resp = _post(GROQ_URL, {"Authorization": f"Bearer {api_key}"}, body, timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""
    except httpx.HTTPStatusError as exc:
        raise LLMCallError(f"Groq returned HTTP {exc.response.status_code}") from exc
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
        raise LLMCallError(f"Groq call failed: {type(exc).__name__}") from exc


def parse_json_object(text: str) -> dict:
    """Pull the JSON object out of a model reply, ignoring <think> blocks and code fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMCallError("model reply contained no JSON object")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMCallError("model reply was not valid JSON") from exc
    if not isinstance(data, dict):
        raise LLMCallError("model reply was not a JSON object")
    return data
