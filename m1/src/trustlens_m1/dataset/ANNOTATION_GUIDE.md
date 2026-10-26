# TrustLens Claim Annotation Guide (M1 dataset)

This guide is for whoever fills in `template.csv` (or a copy of it) with
real, manually reviewed data. **Do not invent URLs, reviewer names, or
labels.** Rows you have not actually reviewed must stay `review_status =
pending` and are excluded from training by `validate.py` / `train.py`.

## 1. Unit of work

- **Document**: one source (an article, API doc page, Wikipedia entry,
  etc.) that a generated answer's claims are checked against. Identified by
  `document_id`. The project plan requires **at least 50 distinct reviewed
  source documents**.
- **Claim / example**: one atomic factual statement extracted from a
  generated answer, checked against one document. Identified by
  `example_id`. A single document may support multiple claims/examples —
  add more rows with the same `document_id` and a new `example_id`
  (e.g. `ex_003_1`, `ex_003_2`).

## 2. Columns

| Column | Required for training? | Notes |
|---|---|---|
| `example_id` | yes | unique per row, e.g. `ex_003_2` |
| `document_id` | yes | shared across claims from the same source |
| `source_title` | yes | must be identical for every row sharing a `document_id` |
| `source_url` | yes | must be identical for every row sharing a `document_id`; real URL only |
| `claim_text` | yes | the atomic claim being judged |
| `self_consistency_score` | yes | float in [0, 1]; from `self_consistency.py`, or computed during the manual review session using the same method |
| `evidence_relevance_score` | yes | float in [0, 1]; supplied by M2's evidence ranker (BM25 + embedding fusion) |
| `verifier_agreement` | yes | `true`/`false` — did Verifier A and Verifier B agree on a verdict? supplied by M3 |
| `verifier_a_verdict` | yes | one of `supported` / `contradicted` / `unverifiable`; supplied by M3 |
| `verifier_a_model` | recommended | e.g. `gemini-2.5-pro` |
| `verifier_b_verdict` | yes | one of `supported` / `contradicted` / `unverifiable`; supplied by M3 |
| `verifier_b_model` | recommended | **not yet finalized by the team** — the PDF says "Llama via Groq", the team split says "Groq: Qwen"; record whatever was actually used, e.g. `groq-qwen` or `groq-llama-3.3-70b`, and flag the mismatch to the team if you see both in use across rows |
| `gold_label` | yes | the human reviewer's ground truth — see definitions below |
| `reviewed_by` | yes | real reviewer name/initials, never fabricated |
| `review_status` | yes | `pending` (not yet reviewed — excluded from training), `in_review`, `reviewed` (complete, included if gold_label is set), `rejected` (reviewed but discarded, e.g. duplicate or malformed — excluded) |
| `notes` | optional | free text, e.g. disagreement notes |

## 3. Gold label definitions

Apply exactly one of these three labels. Do not use `contradicted` as a
catch-all for "not supported" — if the evidence simply doesn't address the
claim, that is `unverifiable`, not `contradicted`.

- **`supported`** — At least one reliable, independent source directly and
  specifically supports the claim as stated (not just a related topic).
- **`contradicted`** — At least one reliable, independent source directly
  and specifically contradicts the claim as stated.
- **`unverifiable`** — No reliable source was found that is specific enough
  to confirm or deny the claim, even though the claim may be plausible.
  This includes claims about future/undated events, subjective claims, and
  claims where only low-quality or circular sources were found.

## 4. Reviewer workflow

1. Set `review_status = in_review` while you work on a row.
2. Fill in every required column truthfully from what you actually
   observed — never estimate a score you did not compute, never guess a
   verdict you did not read.
3. Set `gold_label` only once you are confident in the classification above.
4. Set `review_status = reviewed` when the row is complete and correct, or
   `rejected` if the row should be discarded (do not delete rejected rows —
   keeping them preserves an audit trail of what was considered).
5. Run `python -m trustlens_m1.dataset.validate <path-to-csv>` before
   treating the file as ready for training.

## 5. Common mistakes to avoid

- Filling in `gold_label` on a row still marked `pending` (the row will be
  silently excluded by the validator until `review_status` is corrected).
- Giving two rows with the same `document_id` different `source_title` /
  `source_url` values (the validator rejects this as inconsistent metadata).
- Counting the same document twice under two different `document_id`
  values — reuse the existing `document_id` for a source you've already
  logged.
- Mixing synthetic/test rows into this file. Keep synthetic fixtures only
  under `tests/fixtures/`, clearly separate from real annotation data.
