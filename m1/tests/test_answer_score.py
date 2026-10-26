from __future__ import annotations

import json

import pytest

from tests.test_scoring import _synthetic_rows, _write_dataset
from trustlens_m1.errors import InvalidInputError
from trustlens_m1.scoring.answer_score import (
    AGGREGATION_METHODS,
    ClaimTrustScore,
    score_answer,
    score_answer_from_features,
)
from trustlens_m1.scoring.answer_score import (
    main as answer_score_main,
)
from trustlens_m1.scoring.features import FusionFeatures
from trustlens_m1.scoring.predict import TrustScorer
from trustlens_m1.scoring.train import load_training_frame, save_model, train_model


def _claim(label, score, claim_id=None):
    probs = {"supported": 0.0, "contradicted": 0.0, "unverifiable": 0.0}
    probs[label] = score
    remainder = (1.0 - score) / 2
    for k in probs:
        if k != label:
            probs[k] = remainder
    return ClaimTrustScore(predicted_label=label, trust_score=score, probabilities=probs, claim_id=claim_id)


def test_claim_trust_score_validates_label():
    with pytest.raises(InvalidInputError):
        ClaimTrustScore(predicted_label="maybe", trust_score=0.5, probabilities={})


def test_claim_trust_score_validates_score_range():
    with pytest.raises(InvalidInputError):
        ClaimTrustScore(predicted_label="supported", trust_score=1.5, probabilities={})
    with pytest.raises(InvalidInputError):
        ClaimTrustScore(predicted_label="supported", trust_score=-0.1, probabilities={})


def test_score_answer_rejects_empty_list():
    with pytest.raises(InvalidInputError):
        score_answer([])


def test_score_answer_rejects_non_list():
    with pytest.raises(InvalidInputError):
        score_answer("not a list")  # type: ignore[arg-type]


def test_score_answer_rejects_unknown_method():
    with pytest.raises(InvalidInputError):
        score_answer([_claim("supported", 0.9)], method="bogus")


def test_all_supported_claims_yield_supported_label_and_high_score():
    claims = [_claim("supported", 0.9), _claim("supported", 0.95), _claim("supported", 0.85)]
    result = score_answer(claims)
    assert result.overall_label == "supported"
    assert result.n_claims == 3
    assert result.n_supported == 3
    assert result.n_contradicted == 0
    assert result.overall_trust_score == pytest.approx(0.85)  # weakest_link = min


def test_single_contradicted_claim_overrides_label_even_with_high_average():
    claims = [_claim("supported", 0.95), _claim("supported", 0.92), _claim("contradicted", 0.05)]
    result = score_answer(claims)
    assert result.overall_label == "contradicted"
    assert result.n_contradicted == 1
    assert result.overall_trust_score == pytest.approx(0.05)  # weakest-link catches it
    assert result.weakest_claim.predicted_label == "contradicted"


def test_mean_method_does_not_catch_single_bad_claim_as_harshly():
    claims = [_claim("supported", 0.95), _claim("supported", 0.92), _claim("contradicted", 0.05)]
    result = score_answer(claims, method="mean")
    # mean is higher than weakest_link for the same input - demonstrates why
    # weakest_link is the safer default.
    weakest = score_answer(claims, method="weakest_link")
    assert result.overall_trust_score > weakest.overall_trust_score
    # label is still "contradicted" regardless of aggregation method chosen,
    # because the label rule is independent of the numeric score.
    assert result.overall_label == "contradicted"


def test_blend_method_is_between_min_and_mean():
    claims = [_claim("supported", 0.9), _claim("supported", 0.5), _claim("supported", 0.1)]
    mean_r = score_answer(claims, method="mean")
    min_r = score_answer(claims, method="weakest_link")
    blend_r = score_answer(claims, method="weakest_link_mean_blend")
    assert min_r.overall_trust_score < blend_r.overall_trust_score < mean_r.overall_trust_score


def test_unverifiable_mixed_with_supported_is_partially_unverifiable():
    claims = [_claim("supported", 0.8), _claim("unverifiable", 0.4)]
    result = score_answer(claims)
    assert result.overall_label == "partially_unverifiable"


def test_all_unverifiable_is_unverifiable():
    claims = [_claim("unverifiable", 0.3), _claim("unverifiable", 0.4)]
    result = score_answer(claims)
    assert result.overall_label == "unverifiable"


def test_weakest_claim_identifies_lowest_scoring_claim():
    claims = [_claim("supported", 0.9, "c1"), _claim("supported", 0.3, "c2"), _claim("supported", 0.7, "c3")]
    result = score_answer(claims)
    assert result.weakest_claim.claim_id == "c2"


def test_per_claim_preserves_order_and_all_rows():
    claims = [_claim("supported", 0.9, "c1"), _claim("contradicted", 0.1, "c2")]
    result = score_answer(claims)
    assert [c.claim_id for c in result.per_claim] == ["c1", "c2"]


@pytest.mark.parametrize("method", AGGREGATION_METHODS)
def test_all_aggregation_methods_produce_valid_score(method):
    claims = [_claim("supported", 0.7), _claim("unverifiable", 0.5)]
    result = score_answer(claims, method=method)
    assert 0.0 <= result.overall_trust_score <= 1.0
    assert result.aggregation_method == method


def test_to_dict_roundtrips_through_json():
    claims = [_claim("supported", 0.8, "c1"), _claim("contradicted", 0.1, "c2")]
    result = score_answer(claims)
    payload = json.dumps(result.to_dict())
    loaded = json.loads(payload)
    assert loaded["overall_label"] == "contradicted"
    assert loaded["n_claims"] == 2
    assert loaded["weakest_claim"]["claim_id"] == "c2"


def test_score_answer_from_features_end_to_end(tmp_path):
    path = tmp_path / "synthetic_labelled.csv"
    _write_dataset(path, _synthetic_rows())
    frame = load_training_frame(path)
    pipeline, _ = train_model(frame)
    model_path = tmp_path / "model.joblib"
    save_model(pipeline, model_path)

    scorer = TrustScorer(model_path)
    claims = [
        FusionFeatures(
            self_consistency_score=0.9, evidence_relevance_score=0.85,
            verifier_agreement=True, verifier_a_verdict="supported",
            verifier_b_verdict="supported", claim_id="c1",
        ),
        FusionFeatures(
            self_consistency_score=0.3, evidence_relevance_score=0.2,
            verifier_agreement=False, verifier_a_verdict="unverifiable",
            verifier_b_verdict="supported", claim_id="c2",
        ),
    ]
    result = score_answer_from_features(scorer, claims)
    assert result.n_claims == 2
    assert {c.claim_id for c in result.per_claim} == {"c1", "c2"}
    assert 0.0 <= result.overall_trust_score <= 1.0


def test_score_answer_from_features_rejects_empty_list(tmp_path):
    path = tmp_path / "synthetic_labelled.csv"
    _write_dataset(path, _synthetic_rows())
    frame = load_training_frame(path)
    pipeline, _ = train_model(frame)
    model_path = tmp_path / "model.joblib"
    save_model(pipeline, model_path)
    scorer = TrustScorer(model_path)
    with pytest.raises(InvalidInputError):
        score_answer_from_features(scorer, [])


def test_cli_aggregates_json_file(tmp_path, capsys):
    claims_json = tmp_path / "claims.json"
    claims_json.write_text(json.dumps([
        {"predicted_label": "supported", "trust_score": 0.9, "probabilities": {"supported": 0.9, "contradicted": 0.05, "unverifiable": 0.05}, "claim_id": "c1"},
        {"predicted_label": "contradicted", "trust_score": 0.1, "probabilities": {"supported": 0.1, "contradicted": 0.8, "unverifiable": 0.1}, "claim_id": "c2"},
    ]), encoding="utf-8")
    rc = answer_score_main([str(claims_json)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["overall_label"] == "contradicted"
    assert out["n_claims"] == 2


def test_cli_missing_file_fails_cleanly(tmp_path, capsys):
    rc = answer_score_main([str(tmp_path / "nope.json")])
    assert rc == 1
    assert "FAILED" in capsys.readouterr().err
