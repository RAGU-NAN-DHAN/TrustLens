"""Minimal HTTP clients for Gemini and Groq. Keys are sent in headers, never in URLs or logs."""

import json
import re
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


def call_gemini(prompt: str, model: str, api_key: str, timeout: float) -> str:
    if not api_key:
        raise LLMCallError("GEMINI_API_KEY is not set")
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    try:
        resp = _post(GEMINI_URL.format(model=model), {"x-goog-api-key": api_key}, body, timeout)
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
