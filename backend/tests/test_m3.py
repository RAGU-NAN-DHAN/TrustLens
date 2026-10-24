import json
from pathlib import Path

import pytest

from app.errors import VerificationError
from app.models.schemas import AnalysisResult
from app.services import verification
from app.services._text import split_sentences
from app.services.correction import check_correction, write_correction
from app.services.disagreement import resolve_disagreement
from app.services.highlighting import highlight_evidence
from app.services.verification import verify_claim

EVIDENCE = [
    {"evidence_id": "e1", "claim_id": "c1", "source": "wikipedia", "url": "u",
     "passage": "Rameswaram is a town in Tamil Nadu. Kalam was born on 15 October 1931 in Rameswaram."},
    {"evidence_id": "e2", "claim_id": "c1", "source": "wikipedia", "url": "u",
     "passage": "The Pamban bridge links the island to the mainland."},
]


def result(verdict):
    return {"verdict": verdict}


# disagreement rule

def test_agree_supported():
    assert resolve_disagreement([result("supported"), result("supported")]) == {
        "final_verdict": "supported", "verifier_agreement": True, "needs_human_review": False}


def test_agree_contradicted():
    assert resolve_disagreement([result("contradicted"), result("contradicted")])["final_verdict"] == "contradicted"


def test_both_unverifiable_is_agreement_without_review():
    out = resolve_disagreement([result("unverifiable"), result("unverifiable")])
    assert out["verifier_agreement"] and not out["needs_human_review"]


def test_disagree_goes_to_review():
    assert resolve_disagreement([result("supported"), result("contradicted")]) == {
        "final_verdict": "unverifiable", "verifier_agreement": False, "needs_human_review": True}


def test_single_result_is_rejected():
    with pytest.raises(ValueError):
        resolve_disagreement([result("supported")])


# verification

def test_mock_catches_wrong_year():
    r = verify_claim("Kalam was born on 15 October 1935 in Rameswaram.", EVIDENCE, "gemini", mock=True)
    assert r["verdict"] == "contradicted" and r["evidence_ids"] == ["e1"]
    assert r["claim_id"] == "c1" and r["model_version"] == "mock-gemini"


def test_mock_supports_matching_claim():
    r = verify_claim("Kalam was born on 15 October 1931 in Rameswaram.", EVIDENCE, "groq_qwen", mock=True)
    assert r["verdict"] == "supported"


def test_mock_without_evidence_is_unverifiable():
    assert verify_claim("Anything.", [], "gemini", mock=True)["verdict"] == "unverifiable"


def test_unknown_verifier_rejected():
    with pytest.raises(VerificationError):
        verify_claim("x", EVIDENCE, "chatgpt", mock=True)


def test_real_reply_is_parsed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    reply = '<think>hmm</think>```json\n{"verdict": "Contradicted", "confidence": 1.4, "explanation": "e1 says 1931.", "evidence_ids": ["e1", "e9"]}\n```'
    monkeypatch.setattr(verification, "call_gemini", lambda *a, **k: reply)
    r = verify_claim("born 1935", EVIDENCE, "gemini", mock=False)
    assert r["verdict"] == "contradicted" and r["confidence"] == 1.0 and r["evidence_ids"] == ["e1"]


def test_real_invalid_verdict_raises(monkeypatch):
    monkeypatch.setattr(verification, "call_gemini", lambda *a, **k: '{"verdict": "probably", "confidence": 0.7}')
    with pytest.raises(VerificationError):
        verify_claim("born 1935", EVIDENCE, "gemini", mock=False)


def test_groq_verifier_refuses_llama(monkeypatch):
    monkeypatch.setenv("GROQ_VERIFIER_MODEL", "llama-3.3-70b-versatile")
    with pytest.raises(VerificationError, match="Llama"):
        verify_claim("born 1935", EVIDENCE, "groq_qwen", mock=False)


def test_groq_verifier_needs_model(monkeypatch):
    monkeypatch.setenv("GROQ_VERIFIER_MODEL", "")
    with pytest.raises(VerificationError, match="GROQ_VERIFIER_MODEL"):
        verify_claim("born 1935", EVIDENCE, "groq_qwen", mock=False)


# highlighting

def test_sentence_split_keeps_initials():
    text = "A. P. J. Abdul Kalam was born in 1931. He died in 2015."
    assert [s[2] for s in split_sentences(text)] == ["A. P. J. Abdul Kalam was born in 1931.", "He died in 2015."]


def test_highlight_marks_matching_sentence_and_keeps_order():
    out = highlight_evidence("Kalam was born in Rameswaram in 1931.", EVIDENCE)
    assert [e["evidence_id"] for e in out] == ["e1", "e2"]
    span = out[0]["highlight_span"]
    assert EVIDENCE[0]["passage"][span["start"]:span["end"]] == "Kalam was born on 15 October 1931 in Rameswaram."
    assert out[1]["highlight_span"] is None
    assert "highlight_span" not in EVIDENCE[0]


def test_highlight_empty_evidence():
    assert highlight_evidence("x", []) == []


# correction

def test_correction_from_evidence_is_verified():
    text = write_correction("Kalam was born on 15 October 1935 in Rameswaram.", EVIDENCE, mock=True)
    assert text == "Kalam was born on 15 October 1931 in Rameswaram."
    assert check_correction(text, EVIDENCE)["verified"] is True


def test_correction_with_unbacked_number_is_rejected():
    check = check_correction("Kalam was born on 15 October 1929 in Rameswaram.", EVIDENCE)
    assert check["verified"] is False and check["support_score"] > 0.5


def test_unrelated_correction_is_rejected():
    assert check_correction("The Eiffel Tower is in Paris.", EVIDENCE)["verified"] is False


def test_empty_correction():
    assert check_correction(None, EVIDENCE) == {"verified": False, "support_score": 0.0}


# contract

def test_sample_analysis_matches_contract():
    sample = Path(__file__).resolve().parents[2] / "frontend" / "src" / "mocks" / "sampleAnalysis.json"
    AnalysisResult.model_validate(json.loads(sample.read_text(encoding="utf-8")))
