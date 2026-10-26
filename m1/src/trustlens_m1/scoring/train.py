"""Phase 3B — logistic-regression training pipeline for claim trust scoring.

``trust_score`` is defined as the model's predicted probability of the
``supported`` class: ``P(supported)``.

Usage:
    python -m trustlens_m1.scoring.train <labelled.csv> <output_model.joblib>
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..dataset.schema import GOLD_LABELS
from ..dataset.validate import load_usable_examples
from ..errors import ModelTrainingError
from ..logging_utils import get_logger, log_event
from .features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION

logger = get_logger("scoring.train")

NUMERIC_FEATURES = ["self_consistency_score", "evidence_relevance_score"]
CATEGORICAL_FEATURES = ["verifier_a_verdict", "verifier_b_verdict"]
BOOLEAN_FEATURES = ["verifier_agreement"]

RANDOM_SEED = 42


@dataclass
class TrainingReport:
    n_train: int
    n_test: int
    class_counts: dict[str, int]
    test_classification_report: dict[str, Any]
    test_confusion_matrix: list[list[int]]
    confusion_matrix_labels: list[str]
    cv_macro_f1_mean: float | None = None
    cv_macro_f1_std: float | None = None
    cv_folds: int | None = None
    warnings: list[str] = field(default_factory=list)


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
        ]
    )
    model = LogisticRegression(
        # Note: no explicit multi_class= here. Recent scikit-learn versions
        # removed the parameter (lbfgs now always fits a true multinomial
        # model for >2 classes); older versions default multi_class='auto',
        # which also selects multinomial for the lbfgs solver used here.
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", model)])


def rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        try:
            records.append(
                {
                    "self_consistency_score": float(row["self_consistency_score"]),
                    "evidence_relevance_score": float(row["evidence_relevance_score"]),
                    "verifier_agreement": str(row["verifier_agreement"]).strip().lower()
                    in ("true", "1", "yes"),
                    "verifier_a_verdict": row["verifier_a_verdict"],
                    "verifier_b_verdict": row["verifier_b_verdict"],
                    "gold_label": row["gold_label"],
                }
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ModelTrainingError(
                f"Row {row.get('example_id', '?')} has an invalid/missing feature: {exc}"
            ) from exc
    return pd.DataFrame.from_records(records)


def load_training_frame(csv_path: str | Path) -> pd.DataFrame:
    """Load only reviewed, labelled rows from ``csv_path`` as a feature frame.

    Raises:
        ModelTrainingError: the dataset has no usable rows, or a usable row
            is missing a required feature.
    """
    rows = load_usable_examples(csv_path)
    if not rows:
        raise ModelTrainingError(
            f"No usable (review_status=reviewed, labelled) rows found in {csv_path}. "
            "Complete manual annotation before training; see dataset/ANNOTATION_GUIDE.md."
        )
    return rows_to_frame(rows)


def train_model(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    cv_folds: int | None = 5,
) -> tuple[Pipeline, TrainingReport]:
    """Train and evaluate the logistic-regression trust-scoring model.

    Splitting strategy: a single stratified train/test split (default 80/20)
    for the headline held-out metrics, PLUS stratified k-fold cross-validation
    (default 5 folds) on the training portion for a more robust macro-F1
    estimate when the dataset is large enough to support it. All
    preprocessing (scaling, one-hot encoding) is fit only on each split's
    training data via the sklearn ``Pipeline``, so there is no leakage from
    the held-out portion.

    Args:
        frame: Output of ``load_training_frame`` (or an equivalent DataFrame
            with columns in FEATURE_COLUMNS + 'gold_label').
        test_size: Fraction of rows held out for the test split.
        random_state: Seed used for the split and for LogisticRegression.
        cv_folds: Requested number of stratified CV folds; silently reduced
            (and reported in ``warnings``) if the smallest class doesn't
            have enough examples, and skipped entirely (``cv_macro_f1_mean
            = None``) if even 2 folds aren't supportable.

    Raises:
        ModelTrainingError: frame is empty, has too few rows for even a
            minimal split, or every row belongs to a single class.
    """
    if frame.empty:
        raise ModelTrainingError("Training frame is empty; nothing to train on.")

    missing_cols = (set(FEATURE_COLUMNS) | {"gold_label"}) - set(frame.columns)
    if missing_cols:
        raise ModelTrainingError(f"Training frame is missing columns: {sorted(missing_cols)}")

    class_counts = frame["gold_label"].value_counts().to_dict()
    warnings: list[str] = []

    if len(class_counts) < 2:
        raise ModelTrainingError(
            f"Only one gold_label class present ({class_counts}); need at least 2 "
            "classes to train a classifier. Collect more diverse annotations."
        )

    n_rows = len(frame)
    min_class_count = min(class_counts.values())

    if n_rows < 10:
        raise ModelTrainingError(
            f"Only {n_rows} usable labelled rows; need at least 10 for a minimally "
            "meaningful train/test split. Collect more reviewed data."
        )
    if min_class_count < 2:
        warnings.append(
            f"At least one class has only {min_class_count} example(s); stratified "
            "splitting may be unreliable for that class."
        )

    X = frame[list(FEATURE_COLUMNS)]
    y = frame["gold_label"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if min_class_count >= 2 else None,
        )
    except ValueError as exc:
        raise ModelTrainingError(f"Train/test split failed: {exc}") from exc

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    label_order = sorted(set(y_train) | set(y_test))
    report_dict = classification_report(
        y_test, y_pred, labels=label_order, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=label_order).tolist()

    cv_mean: float | None = None
    cv_std: float | None = None
    actual_folds: int | None = None
    train_class_counts = y_train.value_counts()
    max_possible_folds = int(train_class_counts.min()) if len(train_class_counts) else 0

    if cv_folds and max_possible_folds >= 2:
        actual_folds = min(cv_folds, max_possible_folds)
        if actual_folds < cv_folds:
            warnings.append(
                f"Reduced CV folds from {cv_folds} to {actual_folds} because the "
                f"smallest training class has only {max_possible_folds} examples."
            )
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)
        fold_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            fold_pipeline = _build_pipeline()
            fold_pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            fold_pred = fold_pipeline.predict(X_train.iloc[val_idx])
            fold_report = classification_report(
                y_train.iloc[val_idx],
                fold_pred,
                labels=label_order,
                output_dict=True,
                zero_division=0,
            )
            fold_scores.append(fold_report["macro avg"]["f1-score"])
        cv_mean = float(np.mean(fold_scores))
        cv_std = float(np.std(fold_scores))
    else:
        warnings.append(
            "Cross-validation skipped: not enough examples in the smallest training "
            "class to support even 2 stratified folds."
        )

    if n_rows < 50:
        warnings.append(
            f"Only {n_rows} usable labelled rows (< {50} document minimum worth of "
            "claims); treat held-out and CV metrics as preliminary, not conclusive."
        )

    report = TrainingReport(
        n_train=len(X_train),
        n_test=len(X_test),
        class_counts={str(k): int(v) for k, v in class_counts.items()},
        test_classification_report=report_dict,
        test_confusion_matrix=cm,
        confusion_matrix_labels=label_order,
        cv_macro_f1_mean=cv_mean,
        cv_macro_f1_std=cv_std,
        cv_folds=actual_folds,
        warnings=warnings,
    )

    log_event(
        logger,
        logging.INFO,
        "training.complete",
        n_train=report.n_train,
        n_test=report.n_test,
        macro_f1=report_dict.get("macro avg", {}).get("f1-score"),
        cv_macro_f1_mean=cv_mean,
        warning_count=len(warnings),
    )

    return pipeline, report


def save_model(pipeline: Pipeline, path: str | Path, *, extra_metadata: dict | None = None) -> None:
    """Persist the trained pipeline with a feature-schema version stamp."""
    payload = {
        "pipeline": pipeline,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "gold_labels": list(GOLD_LABELS),
        "metadata": extra_metadata or {},
    }
    joblib.dump(payload, path)
    log_event(logger, logging.INFO, "model.saved", path=str(path))


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(
            "Usage: python -m trustlens_m1.scoring.train <labelled.csv> <output_model.joblib>",
            file=sys.stderr,
        )
        return 2
    csv_path, model_path = argv
    try:
        frame = load_training_frame(csv_path)
        pipeline, report = train_model(frame)
    except ModelTrainingError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    save_model(
        pipeline,
        model_path,
        extra_metadata={
            "source_csv": str(csv_path),
            "n_train": report.n_train,
            "n_test": report.n_test,
        },
    )
    print(f"Trained on {report.n_train} rows, held out {report.n_test} rows.")
    print(f"Class counts: {report.class_counts}")
    print(f"Macro-F1 (held-out): {report.test_classification_report['macro avg']['f1-score']:.3f}")
    if report.cv_macro_f1_mean is not None:
        print(
            f"CV macro-F1 ({report.cv_folds}-fold): "
            f"{report.cv_macro_f1_mean:.3f} +/- {report.cv_macro_f1_std:.3f}"
        )
    for w in report.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"Model saved to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
