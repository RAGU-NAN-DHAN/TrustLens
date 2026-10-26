"""Phase 5C — end-to-end integration example/test.

Demonstrates the full intended flow:

    Question
      -> M1 generates an answer and sample variants (generator + self_consistency)
      -> M1 computes self-consistency features
      -> M2 supplies evidence_relevance_score (mocked here)
      -> M3 supplies verifier verdicts and agreement (mocked here)
      -> M1 predicts trust probabilities from a loaded model

Uses a mocked Groq client, a fake embedding model, and a model trained on
synthetic fixtures in-process — no real Groq calls, no real weight
downloads, no pre-existing production model required.
"""

from __future__ import annotations

import pytest

from tests.conftest import FakeGroqClient
from tests.test_scoring import _synthetic_rows, _write_dataset
from trustlens_m1.config import SelfConsistencyConfig
from trustlens_m1.scoring.features import FusionFeatures
from trustlens_m1.scoring.predict import TrustScorer
from trustlens_m1.scoring.train import load_training_frame, save_model, train_model
from trustlens_m1.self_consistency import run_self_consistency


@pytest.mark.asyncio
async def test_end_to_end_question_to_trust_score(tmp_path, generator_config, fake_embedding_model):
    # --- M1: generate + self-consistency -----------------------------------
    answers = [
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower stands in Paris.",
        "Paris is home to the Eiffel Tower.",
        "You can find the Eiffel Tower in the city of Paris.",
    ]
    client = FakeGroqClient(answers=answers)
    sc_result = await run_self_consistency(
        "Where is the Eiffel Tower?",
        generator_config=generator_config,
        sc_config=SelfConsistencyConfig(sample_count=3, temperature=0.9, max_output_tokens=256),
        client=client,
        _embedding_model=fake_embedding_model,
    )
    assert sc_result.overall_consistency_score > 0.0

    # --- M2 (mocked): evidence relevance score -------------------------------
    evidence_relevance_score = 0.82  # would come from M2's ranking component

    # --- M3 (mocked): verifier verdicts + agreement ---------------------------
    verifier_a_verdict = "supported"
    verifier_b_verdict = "supported"
    verifier_agreement = verifier_a_verdict == verifier_b_verdict

    # --- M1: fuse features and predict trust score ----------------------------
    model_path = tmp_path / "integration_model.joblib"
    dataset_path = tmp_path / "integration_fixture.csv"
    _write_dataset(dataset_path, _synthetic_rows())
    frame = load_training_frame(dataset_path)
    pipeline, _ = train_model(frame)
    save_model(pipeline, model_path)

    scorer = TrustScorer(model_path)
    features = FusionFeatures(
        self_consistency_score=sc_result.overall_consistency_score,
        evidence_relevance_score=evidence_relevance_score,
        verifier_agreement=verifier_agreement,
        verifier_a_verdict=verifier_a_verdict,
        verifier_b_verdict=verifier_b_verdict,
    )
    trust_result = scorer.predict(features)

    assert trust_result.predicted_label in {"supported", "contradicted", "unverifiable"}
    assert 0.0 <= trust_result.trust_score <= 1.0
    # M1 must not overrule M3: it only reports probabilities, never a
    # "final verdict" that would supersede the disagreement rule's decision.
    assert not hasattr(trust_result, "final_verdict")
