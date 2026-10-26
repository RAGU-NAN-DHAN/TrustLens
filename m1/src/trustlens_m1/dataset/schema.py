"""Phase 3A — labelled dataset schema.

Defines the columns, types, and allowed values for TrustLens's manually
reviewed claim-level training data. Document metadata (source) is tracked
separately from per-claim training rows so a single document can contribute
multiple claims without inflating the "distinct reviewed sources" count.

IMPORTANT (documented disagreement between project documents): the original
project plan PDF describes the second verifier as "a Llama-based verifier
(via Groq)", while a separately agreed team split specifies Groq: Qwen for
Verifier 2. This schema does NOT hardcode either assumption — it stores
whatever model produced the verdict under ``verifier_a_model`` /
``verifier_b_model`` (free text, supplied by M3) and keeps the verdict
itself in ``verifier_a_verdict`` / ``verifier_b_verdict``. M2/M3/the team
must agree on the actual Verifier 2 model before the first real annotation
pass; this is flagged again in M1_INTEGRATION_CONTRACT.md.
"""

from __future__ import annotations

GOLD_LABELS = ("supported", "contradicted", "unverifiable")
"""Ground-truth claim classes.

- supported: reliable evidence supports the claim.
- contradicted: reliable evidence contradicts the claim.
- unverifiable: available evidence is insufficient for a justified verdict.

``unverifiable`` is a distinct class, not an alias for ``contradicted``.
"""

VERIFIER_VERDICTS = ("supported", "contradicted", "unverifiable")
"""Allowed values for verifier_a_verdict / verifier_b_verdict (same label set)."""

REVIEW_STATUSES = ("pending", "in_review", "reviewed", "rejected")

# Document-level metadata (one row per distinct source document).
DOCUMENT_COLUMNS = (
    "document_id",
    "source_title",
    "source_url",
    "reviewed_by",
    "review_status",
    "notes",
)

# Claim-level training examples (many rows may share a document_id).
EXAMPLE_COLUMNS = (
    "example_id",
    "document_id",
    "source_title",
    "source_url",
    "claim_text",
    "self_consistency_score",
    "evidence_relevance_score",
    "verifier_agreement",
    "verifier_a_verdict",
    "verifier_a_model",
    "verifier_b_verdict",
    "verifier_b_model",
    "gold_label",
    "reviewed_by",
    "review_status",
    "notes",
)
"""Columns of the flat, single-CSV training template.

Document metadata (source_title/source_url) is duplicated onto each claim
row for a simple single-file workflow, but ``validate.py`` checks that all
rows sharing a ``document_id`` agree on ``source_title``/``source_url`` so
the data never silently drifts, and counts *distinct* ``document_id``
values (not rows) toward the 50-document milestone.
"""

SCORE_COLUMNS = ("self_consistency_score", "evidence_relevance_score")
BOOL_COLUMNS = ("verifier_agreement",)

MIN_REVIEWED_DOCUMENTS = 50
