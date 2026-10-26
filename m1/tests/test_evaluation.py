from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tests.test_scoring import _synthetic_rows, _write_dataset
from trustlens_m1.errors import ModelTrainingError
from trustlens_m1.evaluation.ablation import run_ablation, write_ablation_report
from trustlens_m1.evaluation.metrics import evaluate_full_model, write_reports
from trustlens_m1.scoring.train import rows_to_frame


def _frame(tmp_path: Path, n_per_class: int = 15):
    path = tmp_path / "synthetic_labelled.csv"
    _write_dataset(path, _synthetic_rows(n_per_class))
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if r["review_status"] == "reviewed"]
    return rows_to_frame(rows)


def test_evaluate_full_model_reports_expected_fields(tmp_path):
    frame = _frame(tmp_path)
    summary = evaluate_full_model(frame)
    assert 0.0 <= summary.accuracy <= 1.0
    assert 0.0 <= summary.macro_f1 <= 1.0
    assert 0.0 <= summary.weighted_f1 <= 1.0
    assert set(summary.per_class.keys()) == {"supported", "contradicted", "unverifiable"}
    assert summary.confusion_matrix_labels
    assert summary.random_seed == 42
    assert "WARNING" not in summary.evaluation_method  # sanity, not an assertion on content


def test_evaluate_full_model_uses_cv_when_possible(tmp_path):
    frame = _frame(tmp_path, n_per_class=15)
    summary = evaluate_full_model(frame)
    assert summary.cv_folds is not None
    assert summary.cv_macro_f1_mean is not None


def test_evaluate_full_model_warns_on_small_dataset(tmp_path):
    frame = _frame(tmp_path, n_per_class=4)  # 12 rows total, still >= MIN_ROWS_FOR_EVAL
    summary = evaluate_full_model(frame)
    assert any("preliminary" in w for w in summary.warnings)


def test_evaluate_full_model_raises_on_too_few_rows(tmp_path):
    frame = _frame(tmp_path, n_per_class=1)[:3]
    with pytest.raises(ModelTrainingError):
        evaluate_full_model(frame)


def test_write_reports_creates_files(tmp_path):
    frame = _frame(tmp_path)
    summary = evaluate_full_model(frame)
    reports_dir = tmp_path / "reports"
    write_reports(summary, reports_dir)
    assert (reports_dir / "evaluation_summary.json").exists()
    assert (reports_dir / "per_class_metrics.csv").exists()
    assert (reports_dir / "confusion_matrix.csv").exists()


def test_run_ablation_produces_six_variants(tmp_path):
    frame = _frame(tmp_path)
    results = run_ablation(frame)
    names = {r.name for r in results}
    assert names == {
        "full_model",
        "without_self_consistency",
        "without_evidence_relevance",
        "without_verifier_agreement",
        "without_verifier_verdicts",
        "reduced_baseline",
    }
    for r in results:
        assert 0.0 <= r.accuracy <= 1.0
        assert r.n_train + r.n_test == len(frame)


def test_run_ablation_raises_on_single_class(tmp_path):
    rows = [r for r in _synthetic_rows() if r["gold_label"] == "supported"]
    path = tmp_path / "single.csv"
    _write_dataset(path, rows)
    frame = rows_to_frame(rows)
    with pytest.raises(ModelTrainingError):
        run_ablation(frame)


def test_write_ablation_report_creates_csv(tmp_path):
    frame = _frame(tmp_path)
    results = run_ablation(frame)
    reports_dir = tmp_path / "reports"
    write_ablation_report(results, reports_dir)
    out_file = reports_dir / "ablation_results.csv"
    assert out_file.exists()
    with out_file.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
