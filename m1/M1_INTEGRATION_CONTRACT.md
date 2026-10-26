# M1 Integration Contract

Owner: M1 (generation, self-consistency, shared embeddings, trust scoring).
Consumers: M2 (claim splitting, evidence retrieval/ranking, semantic
caching, FastAPI server, SQLite/history routes) and M3 (independent
verifiers, disagreement rule, correction generation + checking, evidence
highlighting, React UI).

This document is the single source of truth for how M2 and M3 import and
call M1's code. If code and this document ever disagree, treat it as a bug
and fix whichever one is wrong — don't silently follow the code.

## 1. Install

```bash
cd trustlens-m1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # or requirements.txt for runtime-only
pip install -e .
cp .env.example .env   # then fill in a real GROQ_API_KEY
```

Import path for the whole team: `trustlens_m1` (installed as an editable
package from `src/trustlens_m1`). M2/M3 should add this repo as a path or
git dependency rather than copying files.

## 2. Required environment variables

| Variable | Required | Default | Used by |
|---|---|---|---|
| `GROQ_API_KEY` | yes, for any real generation | — | `generator.py` |
| `GROQ_MODEL` | no | `openai/gpt-oss-120b` (Groq no longer serves Llama 3.3 70B; never set this to a Qwen model — Qwen is M3's second verifier) | `generator.py` |
| `GROQ_TIMEOUT_SECONDS` | no | `30` | `generator.py` |
| `GROQ_MAX_OUTPUT_TOKENS` | no | `1024` | `generator.py` |
| `GROQ_MAX_RETRIES` | no | `2` (max allowed: 5) | `generator.py` |
| `TRUSTLENS_SC_SAMPLE_COUNT` | no | `3` (must be 3-5) | `self_consistency.py` |
| `TRUSTLENS_SC_TEMPERATURE` | no | `0.9` | `self_consistency.py` |
| `TRUSTLENS_SC_MAX_OUTPUT_TOKENS` | no | `512` | `self_consistency.py` |
| `TRUSTLENS_EMBEDDING_MODEL` | no | `sentence-transformers/all-MiniLM-L6-v2` | `embeddings.py` |
| `TRUSTLENS_EMBEDDING_DEVICE` | no | `auto` (cpu/cuda) | `embeddings.py` |
| `TRUSTLENS_EMBEDDING_BATCH_SIZE` | no | `32` | `embeddings.py` |

See `.env.example` for a safe placeholder file. Never commit a real
`GROQ_API_KEY`.

First-time setup note: the embedding model's weights download once on first
use (`sentence-transformers` fetches and caches them locally); after that,
embedding inference is fully local/offline.

## 3. Public functions and data contracts

### 3.1 Generation — `trustlens_m1.generator`

```python
async def generate_answer(
    question: str,
    *,
    config: GeneratorConfig | None = None,
    client: SupportsChatCompletion | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> GenerationResult
```

`GenerationResult` fields: `question, answer, model, temperature,
created_at, duration_seconds, prompt_tokens, completion_tokens,
total_tokens, request_id, attempt`. Has `.to_dict()`.

Example:
```python
from trustlens_m1.generator import generate_answer
result = await generate_answer("What is the boiling point of water at sea level?")
print(result.answer, result.model)
```

Raises: `InvalidInputError` (bad question), `trustlens_m1.config.ConfigError`
(missing/invalid env config), `AuthenticationError`, `RateLimitError`,
`GenerationTimeoutError`, `NetworkError`, `UnexpectedResponseError` — all
subclasses of `trustlens_m1.errors.GenerationError` /
`TrustLensM1Error`. Catch the base class if you just need "generation
failed."

### 3.2 Shared embeddings — `trustlens_m1.embeddings`

```python
def embed_texts(texts: list[str], *, config: EmbeddingConfig | None = None) -> np.ndarray
def cosine_similarity(text_a: str, text_b: str, *, config: EmbeddingConfig | None = None) -> float
def cosine_similarity_matrix(texts_a: list[str], texts_b: list[str], *, config=None) -> np.ndarray
```

This is the ONE embedding implementation for the whole team. M2's evidence
ranker and M3's evidence highlighting should both call into this module
rather than instantiating their own `SentenceTransformer`. The model loads
lazily on first call and is cached in-process; vectors are L2-normalized
float32, so dot product == cosine similarity.

Raises: `InvalidInputError` (empty/invalid text), `EmbeddingError` (model
load/inference failure).

### 3.3 Self-consistency — `trustlens_m1.self_consistency`

```python
async def run_self_consistency(
    question: str,
    *,
    generator_config: GeneratorConfig | None = None,
    sc_config: SelfConsistencyConfig | None = None,
    client: SupportsChatCompletion | None = None,
    claims: list[str] | None = None,
) -> SelfConsistencyResult

def score_claim_consistency(
    claim: str,
    sampled_answers: list[str],
    *,
    match_threshold: float = 0.6,
) -> ClaimConsistencyResult
```

`SelfConsistencyResult` fields: `question, canonical_answer,
sampled_answers, overall_consistency_score (0-1), sample_count, temperature,
duration_seconds, per_claim_results (list[ClaimConsistencyResult]), errors`.

`ClaimConsistencyResult` fields: `claim, best_sentence_similarities,
consistency_score (0-1), support_count, sample_count, match_threshold`.

**Important**: `overall_consistency_score` / `consistency_score` measure
agreement across independent samples, NOT factual correctness. A
consistently wrong answer still scores high. M2/M3 should not treat this as
a verification signal on its own — it is one input feature to the
downstream trust model (see 3.5).

Once M2's claim splitter exists, pass its output as `claims=` to get
per-claim scores back; until then, `score_claim_consistency` can be called
directly on sentence-level text.

### 3.4 Dataset schema — `trustlens_m1.dataset`

- `schema.py`: column names, `GOLD_LABELS = ("supported", "contradicted",
  "unverifiable")`, `VERIFIER_VERDICTS` (same three values),
  `MIN_REVIEWED_DOCUMENTS = 50`.
- `template.csv`: blank 50-document starter template.
- `ANNOTATION_GUIDE.md`: how to fill it in by hand.
- `validate.py`: `validate_dataset(path) -> ValidationReport`,
  `load_usable_examples(path) -> list[dict]` (only reviewed + labelled
  rows). CLI: `python -m trustlens_m1.dataset.validate <csv>`.

**Open issue the team must resolve**: the original project-plan PDF
describes Verifier 2 as Llama-based (via Groq); the team split agreed
separately specifies Groq: Qwen for Verifier 2. The dataset schema does not
hardcode either — it has `verifier_b_model` as free text — but M3 must
confirm which model is actually used before the first real annotation pass,
and should keep `verifier_b_model` consistent across the dataset once
decided.

### 3.5 Trust scoring — `trustlens_m1.scoring`

```python
# scoring/features.py
@dataclass(frozen=True)
class FusionFeatures:
    self_consistency_score: float      # 0-1, from self_consistency.py (M1)
    evidence_relevance_score: float    # 0-1, from M2's ranking component
    verifier_agreement: bool           # from M3's disagreement rule
    verifier_a_verdict: str            # supported/contradicted/unverifiable, from M3
    verifier_b_verdict: str            # supported/contradicted/unverifiable, from M3
    claim_id: str | None = None
```

```python
# scoring/train.py
def load_training_frame(csv_path) -> pandas.DataFrame
def train_model(frame, *, test_size=0.2, random_state=42, cv_folds=5) -> (Pipeline, TrainingReport)
def save_model(pipeline, path, *, extra_metadata=None) -> None
```
CLI: `python -m trustlens_m1.scoring.train <labelled.csv> <model.joblib>`

```python
# scoring/predict.py
class TrustScorer:
    def __init__(self, model_path: str | Path): ...
    def predict(self, features: FusionFeatures) -> TrustScoreResult
```
CLI: `python -m trustlens_m1.scoring.predict <model.joblib> --self-consistency 0.8 --evidence-relevance 0.7 --verifier-agreement true --verifier-a supported --verifier-b supported`

`TrustScoreResult` fields: `predicted_label, trust_score, probabilities
(dict of all 3 classes), model_version, feature_columns`.

**`trust_score` is defined as `P(supported)`** — the model's probability
for the `supported` class specifically, not the max class probability and
not `1 - P(contradicted)`.

Raises: `ModelLoadError` (missing file / schema version mismatch / shape
mismatch), `ModelTrainingError` (bad/insufficient training data),
`InvalidInputError` (via `FusionFeatures` construction).

#### Whole-answer trust score — `trustlens_m1.scoring.answer_score`

`TrustScorer.predict()` scores one claim at a time. Once a generated answer
has been split into claims (M2) and each claim scored, aggregate them into
one answer-level result:

```python
def score_answer(
    claim_scores: list[ClaimTrustScore],
    *, method: str = "weakest_link",  # or "mean" / "weakest_link_mean_blend"
) -> AnswerTrustResult

def score_answer_from_features(
    scorer: TrustScorer, claims: list[FusionFeatures], *, method: str = "weakest_link",
) -> AnswerTrustResult  # one-call convenience: predicts every claim, then aggregates
```

`AnswerTrustResult` fields: `overall_trust_score (0-1), overall_label,
n_claims, n_supported, n_contradicted, n_unverifiable, weakest_claim
(ClaimTrustScore | None), per_claim (list[ClaimTrustScore]),
aggregation_method`. Has `.to_dict()`.

**`overall_trust_score` defaults to the weakest-link (minimum) aggregation**
across claims, not the mean — a reader of the final answer can't tell which
claim is bad, so the whole answer's trust is bounded by its single worst
claim rather than diluted by averaging. `weakest_claim` names which claim
drove the score, for UI explainability. **`overall_label` is rule-based, not
derived from the score**: any `contradicted` claim makes the whole answer
`contradicted` regardless of the numeric aggregate; otherwise any
`unverifiable` claim makes it `partially_unverifiable` (if some claims are
`supported`) or `unverifiable` (if all are); otherwise `supported`.

CLI: `python -m trustlens_m1.scoring.answer_score <claims.json> [--method weakest_link|mean|weakest_link_mean_blend]`
— `claims.json` is a JSON list of `TrustScoreResult.to_dict()`-shaped
objects (optionally with `claim_id`/`claim_text`).

Raises: `InvalidInputError` (empty/non-list input, invalid `method`, or an
invalid `ClaimTrustScore`).

### 3.6 Evaluation — `trustlens_m1.evaluation`

```python
# evaluation/metrics.py
def evaluate_full_model(frame, *, random_state=42) -> EvaluationSummary
def write_reports(summary, reports_dir) -> None   # evaluation_summary.json, per_class_metrics.csv, confusion_matrix.csv

# evaluation/ablation.py
def run_ablation(frame, *, random_state=42) -> list[AblationVariantResult]
def write_ablation_report(results, reports_dir) -> None   # ablation_results.csv
```
CLIs:
```bash
python -m trustlens_m1.evaluation.metrics <labelled.csv> <reports_dir>
python -m trustlens_m1.evaluation.ablation <labelled.csv> <reports_dir>
```

## 4. Team boundaries (binding)

- **M2** supplies `evidence_relevance_score` from its ranking component,
  owns claim splitting, evidence retrieval, semantic caching, the FastAPI
  server, and SQLite/history routes. M1 does not implement a second server.
- **M3** supplies `verifier_agreement`, `verifier_a_verdict`,
  `verifier_b_verdict`, and owns the independent verifiers, the
  **disagreement rule**, **correction generation and checking**, evidence
  highlighting, and the React UI.
- **M1** computes `self_consistency_score` and the ML `trust_score`. M1
  does not implement claim splitting, evidence retrieval/ranking, the
  disagreement rule, or correction generation/checking, and does not build
  a FastAPI server or frontend.
- If Verifier A and Verifier B disagree, **M1 must not overrule M3's
  disagreement rule or its decision to require human review**. M1 only
  returns class probabilities (`TrustScoreResult`); it never emits a "final
  verdict" field that could be read as superseding M3's workflow decision.
- M1 must not claim that a correction M3 generates is factually verified —
  correction checking is entirely M3's responsibility; M1's trust score is
  about claims from the *original* answer, scored from whatever features
  are supplied to it.

## 5. Example payloads

**`GenerationResult.to_dict()`**
```json
{
  "question": "What is the capital of France?",
  "answer": "The capital of France is Paris.",
  "model": "openai/gpt-oss-120b",
  "temperature": 0.0,
  "created_at": 1730000000.0,
  "duration_seconds": 0.8,
  "prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
  "request_id": "req_abc123",
  "attempt": 1
}
```

**`FusionFeatures` -> `TrustScoreResult.to_dict()`**
```json
// input
{
  "self_consistency_score": 0.84,
  "evidence_relevance_score": 0.77,
  "verifier_agreement": true,
  "verifier_a_verdict": "supported",
  "verifier_b_verdict": "supported"
}
// output
{
  "predicted_label": "supported",
  "trust_score": 0.91,
  "probabilities": {"supported": 0.91, "contradicted": 0.04, "unverifiable": 0.05},
  "model_version": "1.0.0",
  "feature_columns": ["self_consistency_score", "evidence_relevance_score",
                       "verifier_agreement", "verifier_a_verdict", "verifier_b_verdict"]
}
```

## 6. Commands

```bash
# Install
pip install -r requirements-dev.txt && pip install -e .

# Run the full test suite (no real Groq calls, no real model downloads)
pytest -q

# Dataset workflow
python -m trustlens_m1.dataset.validate data/labelled.csv
python -m trustlens_m1.scoring.train data/labelled.csv models/trust_model.joblib
python -m trustlens_m1.scoring.predict models/trust_model.joblib \
  --self-consistency 0.8 --evidence-relevance 0.7 \
  --verifier-agreement true --verifier-a supported --verifier-b supported

# Evaluation / ablation (once real labelled data exists)
python -m trustlens_m1.evaluation.metrics data/labelled.csv reports/
python -m trustlens_m1.evaluation.ablation data/labelled.csv reports/
```

## 7. Error/failure behavior summary

- Empty/whitespace input anywhere -> `InvalidInputError` immediately, no
  API call made.
- Missing `GROQ_API_KEY` -> `ConfigError` immediately, no API call made.
- Groq auth failure -> `AuthenticationError`, never retried.
- Groq rate limit / timeout / network error -> retried up to
  `GROQ_MAX_RETRIES` (default 2, hard cap 5) times with capped backoff, then
  raised.
- All self-consistency samples failing -> `GenerationError` (never returns
  a score computed from zero samples).
- Training/evaluating on too few rows or a single class -> `ModelTrainingError`
  with an explicit message, never a silently misleading score.
- Loading a model saved under a different feature schema -> `ModelLoadError`.
- No trained model available -> M1 raises `ModelLoadError`; M1 never
  silently substitutes a heuristic score in its place.

## 8. Initialization / trained-artifact requirements

- `generate_answer` / `run_self_consistency` need `GROQ_API_KEY` set; no
  local artifacts required.
- `embed_texts` / `cosine_similarity` download the configured
  sentence-transformers model on first use (needs network once), then run
  fully offline.
- `TrustScorer` needs a `.joblib` file produced by `scoring.train.save_model`
  (or the `scoring.train` CLI) — it is not usable until at least 10 labelled
  examples across 2+ classes have been reviewed and a model trained; the
  project's data-collection milestone targets 50 reviewed documents.

## 10. The real 50-document labelled dataset (`data/labelled.csv`)

Built this phase to satisfy the project's data-collection milestone. Full
provenance and generation logic lives in `build_dataset.py` (included in
this zip at the repo root) — read its module docstring before treating this
as a production gold-standard set. Summary:

- **Real**: `claim_text`, `gold_label`, `source_title`, `source_url` for all
  50 rows/50 distinct documents. Each claim is a well-established,
  stable, textbook/encyclopedia-level fact (geography, chemistry, physics,
  history, basic anatomy) or a claim hand-constructed to directly
  contradict one, checked against a real, resolvable source URL. Two rows
  (Paris=capital of France; gold's symbol=Au) were spot-verified live this
  session via WebFetch against the cited Wikipedia pages.
- **Honestly disclosed as AI-conducted, not human-reviewed**: `reviewed_by`
  on every row says so explicitly (`"automated-research-assistant (Claude,
  Sonnet 5) -- AI desk research, not a human annotation pass; recommend
  human spot-check before production use"`). `review_status="reviewed"` is
  accurate (the claim/label/source triple really was checked) but a human
  spot-check pass is recommended before treating this as a production
  gold-standard set, per `ANNOTATION_GUIDE.md`'s own review workflow.
- **Simulated placeholder signals**: `self_consistency_score`,
  `evidence_relevance_score`, `verifier_agreement`, `verifier_a_verdict`,
  `verifier_b_verdict` are NOT from a real pipeline run — this M1 codebase
  does not include M2's evidence ranker or M3's two verifiers (those are
  the other workstreams' deliverables, not yet built), so there is nothing
  real to source these columns from yet. Every row's `notes` column says so
  explicitly. They are generated with a fixed random seed (42) using a
  documented, deliberately noisy rule — gaussian-jittered scores with real
  class overlap, and two *independent*, imperfect simulated verifiers
  (74%/68% accuracy, not simply copying `gold_label` or each other) — so
  the dataset is informative for training/ablation today (see below) and
  can be swapped for real M2/M3 output later without any schema change.
  `verifier_a_model="gemini-2.5-pro"`, `verifier_b_model="groq-qwen"` are
  placeholder labels for the same reason (see the Verifier 2 model
  ambiguity already flagged in §3.4/README's "Known open issue").

## 11. Training, evaluation, and ablation results on the real dataset

Run via `python -m trustlens_m1.scoring.train data/labelled.csv
data/trust_model.joblib`, then `evaluation.metrics` and
`evaluation.ablation` (same CLIs as §6), both against `data/labelled.csv`
(50 rows, 18 supported / 16 contradicted / 16 unverifiable, stratified
80/20 split, seed 42). These numbers reflect the simulated verifier/score
signals above, **not** real-world verifier accuracy — re-run both commands
once M2/M3's real signals replace the simulated ones.

- **Held-out test (10 rows)**: accuracy 0.800, macro-F1 0.806, weighted-F1
  0.800.
- **5-fold stratified CV**: macro-F1 0.776 mean (± wide variance expected —
  only 50 rows).
- **Ablation** (`reports/ablation_results.csv`, each variant retrained from
  scratch on the identical fixed split):

  | variant | accuracy | macro-F1 |
  |---|---|---|
  | full_model (all features) | 0.800 | 0.806 |
  | without_self_consistency | 0.700 | 0.714 |
  | without_evidence_relevance | 0.600 | 0.589 |
  | without_verifier_agreement | 0.900 | 0.896 |
  | without_verifier_verdicts | 0.400 | 0.356 |
  | reduced_baseline (scores only, no verifier signals) | 0.400 | 0.356 |

  Reading: the two verifier verdicts carry the most signal (removing both
  collapses performance to the `reduced_baseline` floor); evidence
  relevance is the next most valuable single feature; self-consistency adds
  a smaller but real amount; the boolean `verifier_agreement` flag is
  redundant with the two verdict columns it's derived from (dropping it
  scored *higher* on this particular 10-row split — within noise for a
  dataset this small, not evidence the feature is harmful).

## 9. Final QA status (Phase 5D, updated this phase)

- `pytest -q` → **92 passed**, 0 failed (see `tests/`; was 70 passed at the
  end of Phase 5D proper — the 22 new tests are from the generator-model
  config change and the new whole-answer trust score module, §3.5).
- `mypy --ignore-missing-imports src/trustlens_m1` → same pre-existing
  "Library stubs not installed" / "Cannot find implementation" errors as
  before for `pandas`/`numpy`/`sklearn`/`joblib` (PyPI access is blocked in
  this sandboxed build environment, so their stub packages can't be
  installed here) — no *new* type errors introduced this phase. Run
  `mypy --install-types` in an environment with PyPI access to confirm 0
  errors with stubs present.
- `ruff check --fix src tests` → all auto-fixable issues fixed (mostly
  import ordering / `Optional[X]` → `X | None` style); 13 pre-existing
  style-only findings remain in test files untouched by this phase (not
  correctness issues — see ruff's own output for the full list).
- CLI run on the real `data/labelled.csv` (50 rows/50 documents, see §10):
  `dataset.validate` → OK, 0 issues, meets the 50-document minimum;
  `scoring.train` → trained, held-out macro-F1 0.806 (see §11, not
  synthetically perfect — the simulated signals have realistic noise);
  `scoring.predict` → produces valid `TrustScoreResult`s; `evaluation.metrics`
  and `evaluation.ablation` → wrote all report files successfully (§11).
- Real-Groq-key test: **blocked** — no `GROQ_API_KEY` is present in this
  build/execution environment, and none was supplied. See §12 for exactly
  how to run it yourself once you have a key.

## 12. Real-Groq-key test — how to run it (not run in this environment)

This phase's code changes (new default model `openai/gpt-oss-120b`) have
only been exercised against the test suite's fake Groq client (`FakeGroqClient`
in `tests/conftest.py`), never a real Groq API call — the sandboxed
environment this was built in has no network path to the Groq API and no
`GROQ_API_KEY` was available or supplied. To actually test it:

```bash
pip install -r requirements-dev.txt && pip install -e .
export GROQ_API_KEY=gsk_your_real_key_here
python - <<'PY'
import asyncio
from trustlens_m1.generator import generate_answer

async def main():
    result = await generate_answer("What is the capital of France?")
    print(result.answer)
    print("model:", result.model)

asyncio.run(main())
PY
```

If this succeeds, it confirms `openai/gpt-oss-120b` is in fact being served
by Groq's API under your account and the generator's request/response
handling (retries, error mapping, token accounting) works against the real
API, not just the fake client. If Groq returns a "model not found" or
similar error, that means Groq's available model lineup has changed again
since this phase — set `GROQ_MODEL` to whatever Groq currently serves (see
`.env.example`), confirming it is **not** a Qwen model (reserved for M3's
second verifier).
- No secrets are present in the repository; `.env.example` contains only
  placeholder values; `.gitignore` excludes `.env`, `*.joblib`, and
  `reports/` (the real `data/labelled.csv`, the trained
  `data/trust_model.joblib`, and `reports/` generated this phase are
  included in the delivered zip regardless, since `.gitignore` only
  matters if this becomes a git repo).
