"""Phase 4B — ablation study over the trust-scoring feature set.

Compares the full feature set against five documented variants (see
``common.default_feature_sets``). Every variant is retrained from scratch on
the SAME fixed train/test split (``common.fixed_split``, seeded) so results
are comparable to each other, and each variant's preprocessing (scaling,
one-hot encoding) is fit only on that split's training portion — no leakage
from the held-out rows.

Usage:
    python -m trustlens_m1.evaluation.ablation <labelled.csv> <reports_dir>

Writes reports/ablation_results.csv.
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from ..dataset.schema import GOLD_LABELS
from ..errors import ModelTrainingError
from ..logging_utils import get_logger, log_event
from .common import RANDOM_SEED, build_pipeline, default_feature_sets, fixed_split
from .metrics import MIN_ROWS_FOR_EVAL

logger = get_logger("evaluation.ablation")


@dataclass(frozen=True)
class AblationVariantResult:
    name: str
    description: str
    feature_columns: list[str]
    accuracy: float
    macro_f1: float
    weighted_f1: float
    n_train: int
    n_test: int


def run_ablation(
    frame: pd.DataFrame, *, random_state: int = RANDOM_SEED
) -> list[AblationVariantResult]:
    """Retrain and evaluate every documented feature-set variant.

    Raises:
        ModelTrainingError: dataset too small or single-class, same
            conditions as ``evaluation.metrics.evaluate_full_model``.
    """
    if len(frame) < MIN_ROWS_FOR_EVAL:
        raise ModelTrainingError(
            f"Only {len(frame)} usable labelled rows; need at least "
            f"{MIN_ROWS_FOR_EVAL} to run the ablation study."
        )
    if frame["gold_label"].nunique() < 2:
        raise ModelTrainingError(
            "Only one gold_label class present; cannot run ablation on a single class."
        )

    X_train, X_test, y_train, y_test = fixed_split(frame, random_state=random_state)
    label_order = list(GOLD_LABELS)
    present = sorted(set(y_train) | set(y_test))
    for label in present:
        if label not in label_order:
            label_order.append(label)

    results: list[AblationVariantResult] = []
    for feature_set in default_feature_sets():
        pipeline = build_pipeline(feature_set, random_state=random_state)
        pipeline.fit(X_train[feature_set.columns], y_train)
        y_pred = pipeline.predict(X_test[feature_set.columns])

        report = classification_report(
            y_test, y_pred, labels=label_order, output_dict=True, zero_division=0
        )
        results.append(
            AblationVariantResult(
                name=feature_set.name,
                description=feature_set.description,
                feature_columns=feature_set.columns,
                accuracy=float(accuracy_score(y_test, y_pred)),
                macro_f1=report["macro avg"]["f1-score"],
                weighted_f1=report["weighted avg"]["f1-score"],
                n_train=len(X_train),
                n_test=len(X_test),
            )
        )
        log_event(
            logger,
            logging.INFO,
            "ablation.variant_complete",
            variant=feature_set.name,
            accuracy=results[-1].accuracy,
            macro_f1=results[-1].macro_f1,
        )

    return results


def write_ablation_report(results: list[AblationVariantResult], reports_dir: str | Path) -> None:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ablation_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "variant",
                "description",
                "feature_columns",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "n_train",
                "n_test",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.name,
                    r.description,
                    "|".join(r.feature_columns),
                    r.accuracy,
                    r.macro_f1,
                    r.weighted_f1,
                    r.n_train,
                    r.n_test,
                ]
            )


def main(argv: list[str] | None = None) -> int:
    from ..dataset.validate import load_usable_examples
    from ..scoring.train import rows_to_frame

    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(
            "Usage: python -m trustlens_m1.evaluation.ablation <labelled.csv> <reports_dir>",
            file=sys.stderr,
        )
        return 2
    csv_path, reports_dir = argv
    rows = load_usable_examples(csv_path)
    if not rows:
        print(f"No usable rows in {csv_path}. Nothing to ablate.", file=sys.stderr)
        return 1
    frame = rows_to_frame(rows)
    try:
        results = run_ablation(frame)
    except ModelTrainingError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    write_ablation_report(results, reports_dir)
    for r in results:
        print(f"{r.name:28s} acc={r.accuracy:.3f} macro_f1={r.macro_f1:.3f} "
              f"weighted_f1={r.weighted_f1:.3f}")
    if len(frame) < 50:
        print(
            f"WARNING: only {len(frame)} usable examples; ablation differences may not "
            "be statistically meaningful at this sample size.",
            file=sys.stderr,
        )
    print(f"Report written to {reports_dir}/ablation_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
