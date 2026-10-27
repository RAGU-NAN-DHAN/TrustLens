"""Key failover and 'latest' question handling, tested offline with fake HTTP responses."""

import httpx

from app import pipeline
from app.services import _llm


class FakeResp:
    def __init__(self, status, text="ok"):
        self.status_code = status
        self._text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)

    def json(self):
        return {"candidates": [{"content": {"parts": [{"text": self._text}]}}]}


def test_switches_to_backup_key_when_first_is_out_of_quota(monkeypatch):
    used = []

    def fake_post(url, headers, json, timeout):
        used.append(headers["x-goog-api-key"])
        return FakeResp(429 if headers["x-goog-api-key"] == "k1" else 200, '{"a": 1}')

    monkeypatch.setattr(_llm.httpx, "post", fake_post)
    _llm.take_notes()
    assert _llm.call_gemini("hi", "m", "k1,k2", 5) == '{"a": 1}'
    assert used == ["k1", "k2"]
    assert any("switched to backup key 2" in n for n in _llm.take_notes())


def test_single_key_does_not_note_a_switch(monkeypatch):
    monkeypatch.setattr(_llm.httpx, "post", lambda url, headers, json, timeout: FakeResp(200, "x"))
    _llm.take_notes()
    assert _llm.call_gemini("hi", "m", "only", 5) == "x"
    assert _llm.take_notes() == []


def test_search_mode_sends_google_search_tool(monkeypatch):
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(json)
        return FakeResp(200, "answer")

    monkeypatch.setattr(_llm.httpx, "post", fake_post)
    _llm.call_gemini("q", "m", "k", 5, search=True)
    assert seen["tools"] == [{"google_search": {}}]
    assert "responseMimeType" not in seen["generationConfig"]


def test_latest_questions_also_search_year_pages(monkeypatch):
    queries = []

    def fake_retrieve(q, claim_id="", k=8):
        queries.append(q)
        return [{"passage": f"note for {q}"}]

    monkeypatch.setattr(pipeline, "retrieve_evidence", fake_retrieve)
    monkeypatch.setattr(pipeline, "_suggest_titles", lambda q: [])
    monkeypatch.setattr(pipeline, "_wiki_title", lambda topic: "Ballon d'Or" if topic == "Ballon dor" else topic)
    pipeline._background("Tell me about the latest Ballon dor Winner")
    year = pipeline.datetime.now(pipeline.timezone.utc).year
    assert queries[0] == f"{year} Ballon d'Or"
    assert queries[1] == f"{year - 1} Ballon d'Or"


def test_normal_questions_search_once(monkeypatch):
    queries = []
    monkeypatch.setattr(pipeline, "retrieve_evidence", lambda q, claim_id="", k=8: queries.append(q) or [])
    monkeypatch.setattr(pipeline, "_suggest_titles", lambda q: [])
    pipeline._background("Who wrote Romeo and Juliet?")
    assert queries == ["Who wrote Romeo and Juliet?"]
