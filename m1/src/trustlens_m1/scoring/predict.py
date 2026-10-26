"""Phase 3B — trust-score inference.

Loads a persisted model (see train.py / save_model) and scores single
``FusionFeatures`` rows. ``trust_score`` is defined as the model's predicted
probability of the ``supported`` class: ``P(supported)``.

Usage:
    python -m trustlens_m1.scoring.predict <model.joblib> \\
        --self-consistency 0.8 --evidence-relevance 0.7 \\
        --verifier-agreement true --verifier-a supported --verifier-b supported
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ..dataset.schema import GOLD_LABELS
from ..errors import ModelLoadError
from ..logging_utils import get_logger, log_event
from .features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, FusionFeatures

logger = get_logger("scoring.predict")


@dataclass(frozen=True)
class TrustScoreResult:
    """The data contract M2/M3 consume for a single trust-score prediction.

    Attributes:
        predicted_label: argmax over the three class probabilities.
        trust_score: P(supported) — the model's probability for the
            'supported' class specifically (NOT max probability / NOT
            1 - P(contradicted)).
        probabilities: dict mapping each of GOLD_LABELS to its probability.
        model_version: the feature-schema version stamped at training time.
        feature_columns: the feature columns the loaded model expects.
    """

    predicted_label: str
    trust_score: float
    probabilities: dict[str, float]
    model_version: str
    feature_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_label": self.predicted_label,
            "trust_score": self.trust_score,
            "probabilities": self.probabilities,
            "model_version": self.model_version,
            "feature_columns": self.feature_columns,
        }


class TrustScorer:
    """Loads a trained model once and serves repeated predictions.

    M2/M3 should construct one ``TrustScorer`` per process (or share one)
    rather than reloading the model file per request.
    """

    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadError(f"Model file not found: {path}")
        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model from {path}: {type(exc).__name__}: {exc}"
            ) from exc

        required_keys = {"pipeline", "feature_schema_version", "feature_columns", "gold_labels"}
        if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
            raise ModelLoadError(
                f"Model file at {path} does not match the expected TrustLens "
                "model payload schema (missing keys)."
            )

        if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
            raise ModelLoadError(
                f"Model was trained with feature schema "
                f"{payload['feature_schema_version']!r}, but this code expects "
                f"{FEATURE_SCHEMA_VERSION!r}. Retrain the model or pin a matching "
                "version of trustlens_m1."
            )
        if list(payload["feature_columns"]) != list(FEATURE_COLUMNS):
            raise ModelLoadError(
                f"Model feature columns {payload['feature_columns']} do not match "
                f"the current feature contract {list(FEATURE_COLUMNS)}."
            )

        self._pipeline = payload["pipeline"]
        self._model_version = payload["feature_schema_version"]
        self._feature_columns = list(payload["feature_columns"])
        self._gold_labels = list(payload["gold_labels"])
        self._model_path = str(path)
        log_event(logger, logging.INFO, "model.loaded", path=str(path))

    def predict(self, features: FusionFeatures) -> TrustScoreResult:
        """Score a single claim's fusion features.

        Raises:
            trustlens_m1.errors.InvalidInputError: via FusionFeatures
                validation if constructed incorrectly upstream (not raised
                here directly since ``features`` is already validated).
        """
        row = pd.DataFrame([{k: getattr(features, k) for k in FEATURE_COLUMNS}])
        proba = self._pipeline.predict_proba(row)[0]
        classes = list(self._pipeline.named_steps["classifier"].classes_)
        prob_map = {cls: float(p) for cls, p in zip(classes, proba)}
        for label in GOLD_LABELS:
            prob_map.setdefault(label, 0.0)

        predicted_label = max(prob_map, key=lambda label: prob_map[label])
        trust_score = prob_map.get("supported", 0.0)

        log_event(
            logger,
            logging.INFO,
            "prediction.complete",
            claim_id=features.claim_id,
            predicted_label=predicted_label,
            trust_score=trust_score,
        )

        return TrustScoreResult(
            predicted_label=predicted_label,
            trust_score=trust_score,
            probabilities=prob_map,
            model_version=self._model_version,
            feature_columns=self._feature_columns,
        )


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict a TrustLens trust score for one claim.")
    parser.add_argument("model_path")
    parser.add_argument("--self-consistency", type=float, required=True)
    parser.add_argument("--evidence-relevance", type=float, required=True)
    parser.add_argument("--verifier-agreement", type=_parse_bool, required=True)
    parser.add_argument("--verifier-a", required=True, dest="verifier_a_verdict")
    parser.add_argument("--verifier-b", required=True, dest="verifier_b_verdict")
    args = parser.parse_args(argv)

    try:
        scorer = TrustScorer(args.model_path)
        features = FusionFeatures(
            self_consistency_score=args.self_consistency,
            evidence_relevance_score=args.evidence_relevance,
            verifier_agreement=args.verifier_agreement,
            verifier_a_verdict=args.verifier_a_verdict,
            verifier_b_verdict=args.verifier_b_verdict,
        )
        result = scorer.predict(features)
    except (ModelLoadError, Exception) as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"predicted_label: {result.predicted_label}")
    print(f"trust_score (P(supported)): {result.trust_score:.4f}")
    print(f"probabilities: {result.probabilities}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
