"""Phase 4A — classification metrics for the full-feature trust-scoring model.

Usage:
    python -m trustlens_m1.evaluation.metrics <labelled.csv> <reports_dir>

Writes reports/evaluation_summary.json, reports/per_class_metrics.csv,
reports/confusion_matrix.csv (paths adapted to whatever ``reports_dir`` is
passed). Uses stratified cross-validation when class counts permit, and
reports a limitation instead of a misleading score when they don't.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from ..dataset.schema import GOLD_LABELS
from ..errors import ModelTrainingError
from ..logging_utils import get_logger, log_event
from .common import RANDOM_SEED, build_pipeline, default_feature_sets, fixed_split

logger = get_logger("evaluation.metrics")

MIN_ROWS_FOR_EVAL = 10
MIN_FOLD_CLASS_COUNT = 2


@dataclass
class EvaluationSummary:
    n_examples: int
    class_counts: dict[str, int]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    confusion_matrix_labels: list[str]
    evaluation_method: str
    random_seed: int
    cv_folds: int | None
    cv_macro_f1_mean: float | None
    cv_macro_f1_std: float | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "class_counts": self.class_counts,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "confusion_matrix_labels": self.confusion_matrix_labels,
            "evaluation_method": self.evaluation_method,
            "random_seed": self.random_seed,
            "cv_folds": self.cv_folds,
            "cv_macro_f1_mean": self.cv_macro_f1_mean,
            "cv_macro_f1_std": self.cv_macro_f1_std,
            "warnings": self.warnings,
        }


def evaluate_full_model(
    frame: pd.DataFrame, *, random_state: int = RANDOM_SEED
) -> EvaluationSummary:
    """Evaluate the full-feature model with a held-out split plus CV when possible.

    Raises:
        ModelTrainingError: fewer than ``MIN_ROWS_FOR_EVAL`` rows, or only one
            gold_label class present.
    """
    if len(frame) < MIN_ROWS_FOR_EVAL:
        raise ModelTrainingError(
            f"Only {len(frame)} usable labelled rows; need at least "
            f"{MIN_ROWS_FOR_EVAL} to evaluate. Collect more reviewed data."
        )
    class_counts = frame["gold_label"].value_counts().to_dict()
    if len(class_counts) < 2:
        raise ModelTrainingError(
            f"Only one gold_label class present ({class_counts}); cannot evaluate "
            "a classifier with a single class."
        )

    warnings: list[str] = []
    feature_set = default_feature_sets()[0]  # full_model

    X_train, X_test, y_train, y_test = fixed_split(frame, random_state=random_state)

    pipeline = build_pipeline(feature_set, random_state=random_state)
    pipeline.fit(X_train[feature_set.columns], y_train)
    y_pred = pipeline.predict(X_test[feature_set.columns])

    label_order = list(GOLD_LABELS)
    present_labels = sorted(set(y_train) | set(y_test))
    for label in present_labels:
        if label not in label_order:
            label_order.append(label)

    report = classification_report(
        y_test, y_pred, labels=label_order, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=label_order).tolist()
    accuracy = float(accuracy_score(y_test, y_pred))

    per_class = {
        label: {
            "precision": report[label]["precision"],
            "recall": report[label]["recall"],
            "f1": report[label]["f1-score"],
            "support": report[label]["support"],
        }
        for label in label_order
        if label in report
    }
    for label in label_order:
        if label not in per_class:
            per_class[label] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
            warnings.append(f"No examples of class {label!r} in the held-out test split.")

    min_class_count = min(class_counts.values())
    cv_mean: float | None = None
    cv_std: float | None = None
    cv_folds: int | None = None
    evaluation_method = "single stratified 80/20 train/test split"

    if min_class_count >= MIN_FOLD_CLASS_COUNT:
        requested_folds = 5
        cv_folds = min(requested_folds, min_class_count)
        if cv_folds < requested_folds:
            warnings.append(
                f"Reduced CV folds from {requested_folds} to {cv_folds} because the "
                f"smallest class has only {min_class_count} examples."
            )
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        y_all = frame["gold_label"]
        X_all = frame[feature_set.columns]
        fold_f1s = []
        for train_idx, val_idx in skf.split(X_all, y_all):
            fold_pipeline = build_pipeline(feature_set, random_state=random_state)
            fold_pipeline.fit(X_all.iloc[train_idx], y_all.iloc[train_idx])
            fold_pred = fold_pipeline.predict(X_all.iloc[val_idx])
            fold_report = classification_report(
                y_all.iloc[val_idx],
                fold_pred,
                labels=label_order,
                output_dict=True,
                zero_division=0,
            )
            fold_f1s.append(fold_report["macro avg"]["f1-score"])
        cv_mean = float(np.mean(fold_f1s))
        cv_std = float(np.std(fold_f1s))
        evaluation_method += f" + {cv_folds}-fold stratified cross-validation"
    else:
        warnings.append(
            f"Stratified cross-validation skipped: smallest class has only "
            f"{min_class_count} example(s), need >= {MIN_FOLD_CLASS_COUNT}."
        )

    if len(frame) < 50:
        warnings.append(
            f"Dataset has only {len(frame)} usable labelled examples (the project "
            "targets >= 50 reviewed documents); treat these metrics as preliminary."
        )

    summary = EvaluationSummary(
        n_examples=len(frame),
        class_counts={str(k): int(v) for k, v in class_counts.items()},
        accuracy=accuracy,
        macro_f1=report["macro avg"]["f1-score"],
        weighted_f1=report["weighted avg"]["f1-score"],
        per_class=per_class,
        confusion_matrix=cm,
        confusion_matrix_labels=label_order,
        evaluation_method=evaluation_method,
        random_seed=random_state,
        cv_folds=cv_folds,
        cv_macro_f1_mean=cv_mean,
        cv_macro_f1_std=cv_std,
        warnings=warnings,
    )

    log_event(
        logger,
        logging.INFO,
        "evaluation.complete",
        n_examples=summary.n_examples,
        accuracy=summary.accuracy,
        macro_f1=summary.macro_f1,
        cv_macro_f1_mean=summary.cv_macro_f1_mean,
        warning_count=len(summary.warnings),
    )
    return summary


def write_reports(summary: EvaluationSummary, reports_dir: str | Path) -> None:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "evaluation_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )

    with (out / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1", "support"])
        for label, metrics in summary.per_class.items():
            writer.writerow(
                [label, metrics["precision"], metrics["recall"], metrics["f1"], metrics["support"]]
            )

    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + summary.confusion_matrix_labels)
        for label, row in zip(summary.confusion_matrix_labels, summary.confusion_matrix):
            writer.writerow([label] + row)


def main(argv: list[str] | None = None) -> int:
    from ..dataset.validate import load_usable_examples
    from ..scoring.train import rows_to_frame

    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(
            "Usage: python -m trustlens_m1.evaluation.metrics <labelled.csv> <reports_dir>",
            file=sys.stderr,
        )
        return 2
    csv_path, reports_dir = argv
    rows = load_usable_examples(csv_path)
    if not rows:
        print(
            f"No usable (reviewed + labelled) rows in {csv_path}. Nothing to evaluate. "
            "Complete annotation first; see dataset/ANNOTATION_GUIDE.md.",
            file=sys.stderr,
        )
        return 1
    frame = rows_to_frame(rows)
    try:
        summary = evaluate_full_model(frame)
    except ModelTrainingError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    write_reports(summary, reports_dir)
    print(f"accuracy={summary.accuracy:.3f} macro_f1={summary.macro_f1:.3f} "
          f"weighted_f1={summary.weighted_f1:.3f}")
    print(f"evaluation_method: {summary.evaluation_method}")
    for w in summary.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    print(f"Reports written to {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
