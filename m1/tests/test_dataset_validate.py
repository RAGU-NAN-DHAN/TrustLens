from __future__ import annotations

import csv
from pathlib import Path

import pytest

from trustlens_m1.dataset.schema import EXAMPLE_COLUMNS
from trustlens_m1.dataset.validate import load_usable_examples, validate_dataset
from trustlens_m1.errors import DatasetError


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(EXAMPLE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _base_row(**overrides: str) -> dict[str, str]:
    row = {
        "example_id": "ex_001_1",
        "document_id": "doc_001",
        "source_title": "Example Source",
        "source_url": "https://example.com/a",
        "claim_text": "Water boils at 100C at sea level.",
        "self_consistency_score": "0.9",
        "evidence_relevance_score": "0.8",
        "verifier_agreement": "true",
        "verifier_a_verdict": "supported",
        "verifier_a_model": "gemini-2.5-pro",
        "verifier_b_verdict": "supported",
        "verifier_b_model": "groq-qwen",
        "gold_label": "supported",
        "reviewed_by": "tester",
        "review_status": "reviewed",
        "notes": "",
    }
    row.update(overrides)
    return row


def test_validate_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError):
        validate_dataset(tmp_path / "nope.csv")


def test_validate_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["example_id"])
        writer.writeheader()
        writer.writerow({"example_id": "x"})
    with pytest.raises(DatasetError):
        validate_dataset(path)


def test_validate_clean_dataset_has_no_hard_errors(tmp_path):
    path = tmp_path / "clean.csv"
    _write_csv(path, [_base_row()])
    report = validate_dataset(path)
    assert report.is_valid
    assert report.usable_rows == 1
    assert report.distinct_documents_usable == 1


def test_validate_rejects_duplicate_example_id(tmp_path):
    path = tmp_path / "dupe.csv"
    rows = [_base_row(), _base_row(example_id="ex_001_1", document_id="doc_002")]
    _write_csv(path, rows)
    report = validate_dataset(path)
    assert not report.is_valid


def test_validate_rejects_inconsistent_document_metadata(tmp_path):
    path = tmp_path / "inconsistent.csv"
    rows = [
        _base_row(example_id="ex_1", document_id="doc_x", source_title="Title A"),
        _base_row(example_id="ex_2", document_id="doc_x", source_title="Title B"),
    ]
    _write_csv(path, rows)
    report = validate_dataset(path)
    assert not report.is_valid


def test_validate_rejects_invalid_gold_label(tmp_path):
    path = tmp_path / "bad_label.csv"
    _write_csv(path, [_base_row(gold_label="definitely_true")])
    report = validate_dataset(path)
    assert not report.is_valid


def test_validate_rejects_score_out_of_range(tmp_path):
    path = tmp_path / "bad_score.csv"
    _write_csv(path, [_base_row(self_consistency_score="1.5")])
    report = validate_dataset(path)
    assert not report.is_valid


def test_unreviewed_rows_excluded_from_usable(tmp_path):
    path = tmp_path / "pending.csv"
    rows = [_base_row(review_status="pending", gold_label="")]
    _write_csv(path, rows)
    report = validate_dataset(path)
    assert report.usable_rows == 0
    examples = load_usable_examples(path)
    assert examples == []


def test_load_usable_examples_filters_correctly(tmp_path):
    path = tmp_path / "mixed.csv"
    rows = [
        _base_row(example_id="ex_1", document_id="doc_1", review_status="reviewed"),
        _base_row(example_id="ex_2", document_id="doc_2", review_status="pending", gold_label=""),
        _base_row(example_id="ex_3", document_id="doc_3", review_status="rejected"),
    ]
    _write_csv(path, rows)
    examples = load_usable_examples(path)
    assert len(examples) == 1
    assert examples[0]["example_id"] == "ex_1"


def test_template_csv_has_fifty_blank_rows():
    template_path = Path(__file__).resolve().parents[1] / "src" / "trustlens_m1" / "dataset" / "template.csv"
    with template_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) >= 50
    document_ids = {r["document_id"] for r in rows}
    assert len(document_ids) >= 50
    for row in rows:
        assert row["review_status"] == "pending"
        assert row["gold_label"] == ""
