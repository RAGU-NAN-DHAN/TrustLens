"""Shared helpers for evaluation and ablation: a feature-subset-aware pipeline
builder and a single fixed train/test split used across all experiments so
that different feature-set variants remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42

ALL_NUMERIC = ["self_consistency_score", "evidence_relevance_score"]
ALL_CATEGORICAL = ["verifier_a_verdict", "verifier_b_verdict"]
ALL_BOOLEAN = ["verifier_agreement"]


@dataclass(frozen=True)
class FeatureSet:
    """A named subset of the full feature contract, used for ablation."""

    name: str
    description: str
    numeric: list[str]
    categorical: list[str]
    boolean: list[str]

    @property
    def columns(self) -> list[str]:
        return list(self.numeric) + list(self.categorical) + list(self.boolean)


def build_pipeline(feature_set: FeatureSet, *, random_state: int = RANDOM_SEED) -> Pipeline:
    """Build a LogisticRegression pipeline restricted to ``feature_set``.

    Mirrors ``scoring.train._build_pipeline`` but accepts an arbitrary
    feature subset so the ablation study can retrain fairly on each variant.
    If a feature subset drops all numeric/categorical/boolean columns in one
    group, that group's transformer is simply omitted.
    """
    transformers = []
    if feature_set.numeric:
        transformers.append(("numeric", StandardScaler(), feature_set.numeric))
    if feature_set.categorical:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), feature_set.categorical)
        )
    if feature_set.boolean:
        transformers.append(("boolean", "passthrough", feature_set.boolean))

    if not transformers:
        raise ValueError(f"Feature set {feature_set.name!r} has no columns at all")

    preprocessor = ColumnTransformer(transformers=transformers)
    model = LogisticRegression(
        # See scoring/train.py's _build_pipeline for why multi_class= is omitted.
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", model)])


def default_feature_sets() -> list[FeatureSet]:
    """The six documented ablation variants, in report order."""
    return [
        FeatureSet(
            name="full_model",
            description=(
                "self-consistency + evidence relevance + verifier agreement + "
                "both verifier verdict categories"
            ),
            numeric=ALL_NUMERIC,
            categorical=ALL_CATEGORICAL,
            boolean=ALL_BOOLEAN,
        ),
        FeatureSet(
            name="without_self_consistency",
            description="full model minus self_consistency_score",
            numeric=["evidence_relevance_score"],
            categorical=ALL_CATEGORICAL,
            boolean=ALL_BOOLEAN,
        ),
        FeatureSet(
            name="without_evidence_relevance",
            description="full model minus evidence_relevance_score",
            numeric=["self_consistency_score"],
            categorical=ALL_CATEGORICAL,
            boolean=ALL_BOOLEAN,
        ),
        FeatureSet(
            name="without_verifier_agreement",
            description="full model minus the verifier_agreement boolean",
            numeric=ALL_NUMERIC,
            categorical=ALL_CATEGORICAL,
            boolean=[],
        ),
        FeatureSet(
            name="without_verifier_verdicts",
            description="full model minus verifier_a_verdict and verifier_b_verdict",
            numeric=ALL_NUMERIC,
            categorical=[],
            boolean=ALL_BOOLEAN,
        ),
        FeatureSet(
            name="reduced_baseline",
            description=(
                "documented minimal baseline: self_consistency_score and "
                "evidence_relevance_score only, no verifier signals at all"
            ),
            numeric=ALL_NUMERIC,
            categorical=[],
            boolean=[],
        ),
    ]


def fixed_split(
    frame: pd.DataFrame, *, test_size: float = 0.2, random_state: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """One fixed, stratified-when-possible train/test split reused by every
    evaluation/ablation variant so results are comparable to each other.
    """
    from sklearn.model_selection import train_test_split

    y = frame["gold_label"]
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        frame,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test
