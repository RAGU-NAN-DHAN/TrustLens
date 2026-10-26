# TrustLens — M1 Module

Groq-based answer generation, self-consistency sampling, a shared local
embedding service, and logistic-regression claim trust scoring for the
TrustLens AI answer-verification project.

This is M1's independently testable module. It is deliberately decoupled
from M2 (FastAPI server, evidence retrieval/ranking, claim splitting,
semantic caching) and M3 (independent verifiers, disagreement rule,
correction generation/checking, React UI) — see
[`M1_INTEGRATION_CONTRACT.md`](./M1_INTEGRATION_CONTRACT.md) for exactly
what M1 exposes and what M2/M3 must supply.

## What's here

```
src/trustlens_m1/
  generator.py          Phase 1: Groq answer generation (default model: openai/gpt-oss-120b)
  embeddings.py          Phase 2A: shared local sentence-transformers service
  self_consistency.py    Phase 2B: multi-sample consistency scoring
  dataset/                Phase 3A: labelled-dataset schema, template, validator
  scoring/                Phase 3B: FusionFeatures + logistic regression train/predict
  evaluation/              Phase 4: classification metrics + ablation study
tests/                    Unit + integration tests (mocked/fake, no real API calls)
M1_INTEGRATION_CONTRACT.md  Full API contract for M2/M3
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
cp .env.example .env   # fill in a real GROQ_API_KEY for real generation
```

## Run the tests

```bash
pytest -q
```

No test makes a real Groq API call or downloads real embedding weights —
everything is exercised against injected fakes (see `tests/conftest.py`).

## Try it end to end

A real, 50-document labelled dataset now ships at `data/labelled.csv` (see
`build_dataset.py` and `M1_INTEGRATION_CONTRACT.md` §10 for its full
provenance — claims/labels/sources are real, the verifier/score columns are
documented simulated placeholders pending M2/M3), with a trained model at
`data/trust_model.joblib` and evaluation/ablation reports already written
to `reports/`. To reproduce or rebuild on your own data:

```bash
# 1. Validate a labelled dataset (see dataset/ANNOTATION_GUIDE.md to build one)
python -m trustlens_m1.dataset.validate data/labelled.csv

# 2. Train the trust-scoring model
python -m trustlens_m1.scoring.train data/labelled.csv data/trust_model.joblib

# 3. Score one claim
python -m trustlens_m1.scoring.predict data/trust_model.joblib \
  --self-consistency 0.8 --evidence-relevance 0.7 \
  --verifier-agreement true --verifier-a supported --verifier-b supported

# 4. Score a whole answer (aggregates multiple claims into one trust score)
python -m trustlens_m1.scoring.answer_score path/to/claims.json

# 5. Evaluate + run the ablation study
python -m trustlens_m1.evaluation.metrics data/labelled.csv reports/
python -m trustlens_m1.evaluation.ablation data/labelled.csv reports/
```

Results on the real dataset (50 rows, seed 42, held-out test = 10 rows):
accuracy 0.800, macro-F1 0.806. Full ablation table in
`M1_INTEGRATION_CONTRACT.md` §11 and `reports/ablation_results.csv`.

## Known open issues

The project plan PDF describes Verifier 2 as Llama-based (via Groq); the
team's agreed work split instead names Groq: Qwen for Verifier 2. M1's
dataset schema does not hardcode either (`verifier_b_model` is free text) —
the team needs to finalize this before real annotation begins. See
`src/trustlens_m1/dataset/schema.py` and `ANNOTATION_GUIDE.md`.

`data/labelled.csv`'s `verifier_agreement`/`verifier_a_verdict`/
`verifier_b_verdict`/`self_consistency_score`/`evidence_relevance_score`
columns are simulated placeholders, not output from a real M2/M3 pipeline
run (M2's evidence ranker and M3's two verifiers aren't built yet in this
repo). Re-run `scoring.train`/`evaluation.metrics`/`evaluation.ablation`
once real M2/M3 signals are available — current results (README above,
`M1_INTEGRATION_CONTRACT.md` §11) describe the simulated data, not
real-world performance.

The real-Groq-key test for the new default generator model
(`openai/gpt-oss-120b`) has not been run — no API key was available in the
build environment. See `M1_INTEGRATION_CONTRACT.md` §12 for the exact
command to run it yourself.
