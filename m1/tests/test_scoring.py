from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from trustlens_m1.dataset.schema import EXAMPLE_COLUMNS
from trustlens_m1.errors import InvalidInputError, ModelLoadError, ModelTrainingError
from trustlens_m1.scoring.features import FEATURE_SCHEMA_VERSION, FusionFeatures
from trustlens_m1.scoring.predict import TrustScorer
from trustlens_m1.scoring.train import load_training_frame, save_model, train_model


def test_fusion_features_valid():
    f = FusionFeatures(
        self_consistency_score=0.8,
        evidence_relevance_score=0.6,
        verifier_agreement=True,
        verifier_a_verdict="supported",
        verifier_b_verdict="supported",
    )
    assert f.to_dict()["self_consistency_score"] == 0.8


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(self_consistency_score=1.5),
        dict(self_consistency_score=-0.1),
        dict(evidence_relevance_score="high"),
        dict(verifier_agreement="yes"),
        dict(verifier_a_verdict="maybe"),
        dict(verifier_b_verdict="maybe"),
    ],
)
def test_fusion_features_rejects_invalid(kwargs):
    base = dict(
        self_consistency_score=0.5,
        evidence_relevance_score=0.5,
        verifier_agreement=True,
        verifier_a_verdict="supported",
        verifier_b_verdict="supported",
    )
    base.update(kwargs)
    with pytest.raises(InvalidInputError):
        FusionFeatures(**base)


def _synthetic_rows(n_per_class: int = 15) -> list[dict[str, str]]:
    random.seed(42)
    rows = []
    idx = 0
    configs = {
        "supported": dict(sc=(0.8, 0.95), ev=(0.7, 0.95), agree=True, va="supported", vb="supported"),
        "contradicted": dict(sc=(0.7, 0.9), ev=(0.6, 0.9), agree=True, va="contradicted", vb="contradicted"),
        "unverifiable": dict(sc=(0.2, 0.5), ev=(0.1, 0.4), agree=False, va="unverifiable", vb="supported"),
    }
    for label, cfg in configs.items():
        for _ in range(n_per_class):
            idx += 1
            rows.append(
                {
                    "example_id": f"ex_{idx:04d}",
                    "document_id": f"doc_{idx:04d}",
                    "source_title": f"Synthetic Source {idx}",
                    "source_url": f"https://example.test/synthetic/{idx}",
                    "claim_text": f"Synthetic claim {idx}",
                    "self_consistency_score": f"{random.uniform(*cfg['sc']):.3f}",
                    "evidence_relevance_score": f"{random.uniform(*cfg['ev']):.3f}",
                    "verifier_agreement": "true" if cfg["agree"] else "false",
                    "verifier_a_verdict": cfg["va"],
                    "verifier_a_model": "synthetic-verifier-a",
                    "verifier_b_verdict": cfg["vb"],
                    "verifier_b_model": "synthetic-verifier-b",
                    "gold_label": label,
                    "reviewed_by": "synthetic-fixture",
                    "review_status": "reviewed",
                    "notes": "SYNTHETIC TEST FIXTURE - not real annotation data",
                }
            )
    return rows


def _write_dataset(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(EXAMPLE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def test_train_model_on_synthetic_fixture(tmp_path):
    path = tmp_path / "synthetic_labelled.csv"
    _write_dataset(path, _synthetic_rows())

    frame = load_training_frame(path)
    pipeline, report = train_model(frame)

    assert report.n_train + report.n_test == len(frame)
    assert set(report.class_counts.keys()) == {"supported", "contradicted", "unverifiable"}
    assert "macro avg" in report.test_classification_report


def test_train_model_raises_on_single_class(tmp_path):
    rows = [r for r in _synthetic_rows() if r["gold_label"] == "supported"]
    path = tmp_path / "single_class.csv"
    _write_dataset(path, rows)
    frame = load_training_frame(path)
    with pytest.raises(ModelTrainingError):
        train_model(frame)


def test_train_model_raises_on_too_few_rows(tmp_path):
    rows = _synthetic_rows(n_per_class=2)[:5]
    path = tmp_path / "too_few.csv"
    _write_dataset(path, rows)
    frame = load_training_frame(path)
    with pytest.raises(ModelTrainingError):
        train_model(frame)


def test_load_training_frame_raises_when_no_usable_rows(tmp_path):
    rows = _synthetic_rows()
    for r in rows:
        r["review_status"] = "pending"
        r["gold_label"] = ""
    path = tmp_path / "all_pending.csv"
    _write_dataset(path, rows)
    with pytest.raises(ModelTrainingError):
        load_training_frame(path)


def test_save_and_load_and_predict_roundtrip(tmp_path):
    path = tmp_path / "synthetic_labelled.csv"
    _write_dataset(path, _synthetic_rows())
    frame = load_training_frame(path)
    pipeline, _ = train_model(frame)

    model_path = tmp_path / "model.joblib"
    save_model(pipeline, model_path)

    scorer = TrustScorer(model_path)
    features = FusionFeatures(
        self_consistency_score=0.9,
        evidence_relevance_score=0.85,
        verifier_agreement=True,
        verifier_a_verdict="supported",
        verifier_b_verdict="supported",
    )
    result = scorer.predict(features)
    assert result.predicted_label in {"supported", "contradicted", "unverifiable"}
    assert 0.0 <= result.trust_score <= 1.0
    assert set(result.probabilities.keys()) == {"supported", "contradicted", "unverifiable"}
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6
    assert result.model_version == FEATURE_SCHEMA_VERSION


def test_trust_scorer_missing_file_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        TrustScorer(tmp_path / "missing.joblib")


def test_trust_scorer_rejects_schema_mismatch(tmp_path):
    import joblib

    bad_payload = {
        "pipeline": object(),
        "feature_schema_version": "0.0.1-outdated",
        "feature_columns": ["self_consistency_score"],
        "gold_labels": ["supported", "contradicted", "unverifiable"],
        "metadata": {},
    }
    path = tmp_path / "bad_model.joblib"
    joblib.dump(bad_payload, path)
    with pytest.raises(ModelLoadError):
        TrustScorer(path)
