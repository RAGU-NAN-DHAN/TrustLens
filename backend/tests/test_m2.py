import pytest
from fastapi.testclient import TestClient

from app import db, pipeline
from app.models.schemas import AnalysisResult
from app.services import evidence_retrieval
from app.services.claim_splitter import split_claims
from app.services.evidence_ranking import rank_evidence
from app.services.semantic_cache import cache_lookup, cache_store, context_hash

ANSWER = "Kalam was born in 1935 in Rameswaram. He was the 11th President of India. I hope this helps."
EVIDENCE = [
    {"evidence_id": "e1", "claim_id": "c_0001", "source": "wikipedia", "url": "u1", "passage": "The Pamban bridge links the island to the mainland."},
    {"evidence_id": "e2", "claim_id": "c_0001", "source": "wikipedia", "url": "u2", "passage": "Kalam was born on 15 October 1931 in Rameswaram."},
]


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("TRUSTLENS_MOCK", "1")


def test_mock_split_one_claim_per_sentence_with_spans():
    claims = split_claims(ANSWER, answer_id="a1", mock=True)
    assert [c["text"] for c in claims] == ["Kalam was born in 1935 in Rameswaram.", "He was the 11th President of India."]
    for c in claims:
        assert ANSWER[c["span"]["start"]:c["span"]["end"]] == c["text"]


def test_ranking_puts_relevant_passage_first_and_scores_it():
    ranked = rank_evidence("Kalam was born in Rameswaram in 1931.", EVIDENCE)
    assert ranked[0]["evidence_id"] == "e2"
    assert ranked[0]["combined_score"] >= ranked[1]["combined_score"]
    assert {"keyword_score", "semantic_score"} <= ranked[0].keys()


def test_ranking_empty():
    assert rank_evidence("x", []) == []


def test_cache_needs_same_evidence_not_just_similar_text():
    ctx = context_hash(EVIDENCE)
    cache_store("Kalam was born in 1935 in Rameswaram.", ctx, {"final_verdict": "contradicted"})
    assert cache_lookup("Kalam was born in 1935 in Rameswaram.", ctx) == {"final_verdict": "contradicted"}
    assert cache_lookup("Kalam was born in 1935 in Rameswaram.", context_hash(EVIDENCE[:1])) is None
    assert cache_lookup("Paris is the capital of France.", ctx) is None


def test_retrieval_survives_one_source_failing(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "")

    def boom(*a, **k):
        raise evidence_retrieval.httpx.ConnectError("offline")

    monkeypatch.setattr(evidence_retrieval, "_wikipedia", boom)
    with pytest.raises(evidence_retrieval.RetrievalError):
        evidence_retrieval.retrieve_evidence("x", claim_id="c_0001", mock=False)


def test_pipeline_end_to_end_with_pasted_answer(monkeypatch):
    monkeypatch.setattr(pipeline, "retrieve_evidence", lambda text, claim_id="": [dict(e, claim_id=claim_id) for e in EVIDENCE])
    result = pipeline.analyze(None, ANSWER)
    AnalysisResult.model_validate(result)
    verdicts = [d["final_verdict"] for d in result["decisions"]]
    assert verdicts[0] == "contradicted"
    assert result["decisions"][0]["correction_verified"] is True
    assert db.get_analysis(result["answer_id"])["answer_text"] == ANSWER
    assert db.list_history()[0]["answer_id"] == result["answer_id"]


def test_api_routes(monkeypatch):
    from app.main import app

    monkeypatch.setattr(pipeline, "retrieve_evidence", lambda text, claim_id="": [])
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/api/v1/analyze", json={"answer_text": ANSWER})
    assert r.status_code == 200, r.text
    answer_id = r.json()["answer_id"]
    assert client.get(f"/api/v1/analysis/{answer_id}").status_code == 200
    assert client.get("/api/v1/analysis/nope").json()["error_code"] == "ANALYSIS_NOT_FOUND"
    assert client.get("/api/v1/history").json()[0]["answer_id"] == answer_id
    bad = client.post("/api/v1/analyze", json={})
    assert bad.status_code == 422 and bad.json()["error_code"] == "INVALID_REQUEST"
