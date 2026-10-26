"""Phase 3A — dataset validation.

Validates a claim-level CSV against ``schema.py`` before it is used for
training or evaluation. Run as a script:

    python -m trustlens_m1.dataset.validate path/to/dataset.csv

Or import ``validate_dataset`` / ``load_usable_examples`` directly.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import DatasetError
from .schema import (
    EXAMPLE_COLUMNS,
    GOLD_LABELS,
    MIN_REVIEWED_DOCUMENTS,
    REVIEW_STATUSES,
    VERIFIER_VERDICTS,
)


@dataclass
class ValidationIssue:
    row_number: int  # 1-indexed, header is row 0
    example_id: str
    message: str

    def __str__(self) -> str:
        return f"row {self.row_number} ({self.example_id or '?'}): {self.message}"


@dataclass
class ValidationReport:
    total_rows: int
    usable_rows: int
    distinct_documents_total: int
    distinct_documents_usable: int
    issues: list[ValidationIssue] = field(default_factory=list)
    gold_label_counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.__dict__.get("_hard_error_count", 0) == 0

    @property
    def meets_document_minimum(self) -> bool:
        return self.distinct_documents_usable >= MIN_REVIEWED_DOCUMENTS

    def summary(self) -> str:
        lines = [
            f"Total rows: {self.total_rows}",
            f"Usable rows (review_status=reviewed, non-empty gold_label): {self.usable_rows}",
            f"Distinct documents (all rows): {self.distinct_documents_total}",
            f"Distinct documents (usable rows): {self.distinct_documents_usable} "
            f"(minimum required: {MIN_REVIEWED_DOCUMENTS})",
            f"Gold label distribution (usable rows): {self.gold_label_counts}",
            f"Issues found: {len(self.issues)}",
        ]
        for issue in self.issues:
            lines.append(f"  - {issue}")
        return "\n".join(lines)


def _parse_float_or_none(value: str | None) -> tuple[bool, float | None]:
    if value is None or value.strip() == "":
        return True, None
    try:
        return True, float(value)
    except ValueError:
        return False, None


def _parse_bool_or_none(value: str | None) -> tuple[bool, bool | None]:
    if value is None or value.strip() == "":
        return True, None
    lowered = value.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True, True
    if lowered in ("false", "0", "no"):
        return True, False
    return False, None


def validate_dataset(csv_path: str | Path) -> ValidationReport:
    """Validate a claim-level CSV and return a structured report.

    Does not raise for row-level problems (collected as ``issues`` instead)
    so a single malformed row doesn't hide every other problem; raises
    ``DatasetError`` only for structural failures (missing file, missing
    required columns).
    """
    path = Path(csv_path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise DatasetError(f"Dataset file has no header row: {path}")
        missing_cols = set(EXAMPLE_COLUMNS) - set(reader.fieldnames)
        if missing_cols:
            raise DatasetError(
                f"Dataset is missing required columns: {sorted(missing_cols)}"
            )
        rows = list(reader)

    issues: list[ValidationIssue] = []
    doc_metadata: dict[str, tuple[str, str]] = {}  # document_id -> (title, url)
    seen_example_ids: set[str] = set()
    all_document_ids: set[str] = set()
    usable_document_ids: set[str] = set()
    usable_rows = 0
    gold_label_counts: dict[str, int] = {label: 0 for label in GOLD_LABELS}

    for idx, row in enumerate(rows, start=1):
        example_id = (row.get("example_id") or "").strip()
        document_id = (row.get("document_id") or "").strip()

        if not example_id:
            issues.append(ValidationIssue(idx, example_id, "example_id is required"))
        elif example_id in seen_example_ids:
            issues.append(ValidationIssue(idx, example_id, "duplicate example_id"))
        else:
            seen_example_ids.add(example_id)

        if not document_id:
            issues.append(ValidationIssue(idx, example_id, "document_id is required"))
        else:
            all_document_ids.add(document_id)
            title = (row.get("source_title") or "").strip()
            url = (row.get("source_url") or "").strip()
            if document_id in doc_metadata:
                prev_title, prev_url = doc_metadata[document_id]
                if title and prev_title and title != prev_title:
                    issues.append(
                        ValidationIssue(
                            idx,
                            example_id,
                            f"source_title for document_id={document_id} is "
                            f"inconsistent with an earlier row",
                        )
                    )
                if url and prev_url and url != prev_url:
                    issues.append(
                        ValidationIssue(
                            idx,
                            example_id,
                            f"source_url for document_id={document_id} is "
                            f"inconsistent with an earlier row",
                        )
                    )
            else:
                doc_metadata[document_id] = (title, url)

        review_status = (row.get("review_status") or "").strip()
        if review_status and review_status not in REVIEW_STATUSES:
            issues.append(
                ValidationIssue(
                    idx, example_id, f"review_status={review_status!r} is not valid"
                )
            )

        gold_label = (row.get("gold_label") or "").strip()
        if gold_label and gold_label not in GOLD_LABELS:
            issues.append(
                ValidationIssue(idx, example_id, f"gold_label={gold_label!r} is not valid")
            )

        for col in ("verifier_a_verdict", "verifier_b_verdict"):
            v = (row.get(col) or "").strip()
            if v and v not in VERIFIER_VERDICTS:
                issues.append(ValidationIssue(idx, example_id, f"{col}={v!r} is not valid"))

        for col in ("self_consistency_score", "evidence_relevance_score"):
            ok, parsed = _parse_float_or_none(row.get(col))
            if not ok:
                issues.append(
                    ValidationIssue(idx, example_id, f"{col}={row.get(col)!r} is not numeric")
                )
            elif parsed is not None and not (0.0 <= parsed <= 1.0):
                issues.append(
                    ValidationIssue(idx, example_id, f"{col}={parsed} is outside [0, 1]")
                )

        ok, _ = _parse_bool_or_none(row.get("verifier_agreement"))
        if not ok:
            issues.append(
                ValidationIssue(
                    idx,
                    example_id,
                    f"verifier_agreement={row.get('verifier_agreement')!r} is not a boolean",
                )
            )

        is_usable = (
            review_status == "reviewed"
            and gold_label in GOLD_LABELS
            and bool(example_id)
            and bool(document_id)
        )
        if is_usable:
            usable_rows += 1
            usable_document_ids.add(document_id)
            gold_label_counts[gold_label] += 1

    hard_errors = [
        i
        for i in issues
        if "inconsistent" in i.message
        or "duplicate" in i.message
        or "is required" in i.message
        or "is not valid" in i.message
        or "not numeric" in i.message
        or "outside [0, 1]" in i.message
        or "not a boolean" in i.message
    ]

    report = ValidationReport(
        total_rows=len(rows),
        usable_rows=usable_rows,
        distinct_documents_total=len(all_document_ids),
        distinct_documents_usable=len(usable_document_ids),
        issues=issues,
        gold_label_counts=gold_label_counts,
    )
    # Attach hard-error flag via attribute rather than property hack above.
    report.__dict__["_hard_error_count"] = len(hard_errors)
    return report


def load_usable_examples(csv_path: str | Path) -> list[dict[str, Any]]:
    """Return only rows that pass validation and are ready for training.

    A row is "usable" when ``review_status == 'reviewed'`` and
    ``gold_label`` is one of the three valid classes. Unreviewed template
    rows and rejected rows are silently excluded (not an error).

    Raises:
        DatasetError: if the file has structural problems (missing file or
            columns) or contains hard validation errors (bad types,
            inconsistent document metadata, duplicate ids).
    """
    report = validate_dataset(csv_path)
    if report.__dict__.get("_hard_error_count", 0) > 0:
        raise DatasetError(
            "Dataset failed validation with hard errors; run "
            "`python -m trustlens_m1.dataset.validate <path>` for details."
        )

    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    usable = [
        row
        for row in rows
        if (row.get("review_status") or "").strip() == "reviewed"
        and (row.get("gold_label") or "").strip() in GOLD_LABELS
    ]
    return usable


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("Usage: python -m trustlens_m1.dataset.validate <path-to-csv>", file=sys.stderr)
        return 2
    try:
        report = validate_dataset(argv[0])
    except DatasetError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(report.summary())
    hard_errors = report.__dict__.get("_hard_error_count", 0)
    if hard_errors:
        print(f"\nFAILED: {hard_errors} hard validation error(s).", file=sys.stderr)
        return 1
    if not report.meets_document_minimum:
        print(
            f"\nWARNING: only {report.distinct_documents_usable} usable reviewed "
            f"documents; {MIN_REVIEWED_DOCUMENTS} required for the data-collection "
            "milestone. Training is still possible but the dataset is incomplete.",
            file=sys.stderr,
        )
    print("\nOK: dataset passed structural validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
