# TrustLens

AI fact-checker: Llama writes an answer, the answer is split into claims, evidence is retrieved and ranked, two independent verifiers judge each claim, and every false claim gets a checked correction.

Full plan and work split: `TrustLens_Context.md` (in the team chat). Contract changes M3 proposed: `docs/contract_changes_m3.md`.

## Status

| Part | Owner | Status |
|---|---|---|
| Verifiers (Gemini + Groq), disagreement rule, correction + checker, evidence highlighting, UI | M3 | Built and tested with real Gemini 3.5 Flash + Groq qwen3.8-27b |
| Generation, self-consistency, embeddings, trust model, evaluation | M1 | In `m1/` (92 tests). Trust model is trained on simulated signals until real pipeline runs replace them |
| Claim splitting, retrieval, ranking, cache, server, database | M2 | Built: FastAPI server, Gemini claim splitting, Wikipedia (+ optional Brave) retrieval, BM25 + embedding ranking, semantic cache, SQLite |

## Run it

Backend (M3 modules and tests):

```bash
cd backend
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/demo_m3.py          # mock verifiers, no keys
.venv/Scripts/python scripts/demo_m3.py --real   # real Gemini + Groq (needs backend/.env)
```

Run the API server (all three parts wired together):

```bash
cd backend
.venv/Scripts/python -m pip install -e ../m1 --no-deps
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

`frontend/.env.development.local` with `VITE_USE_MOCK=false` points the dev site at it. Free API tiers are slow, so each answer is capped at `TRUSTLENS_MAX_CLAIMS` (default 6) claims.

`demo_m3.py` writes `frontend/src/mocks/sampleAnalysis.json`, which the UI shows in demo mode.

Frontend:

```bash
cd frontend
npm install
npm test
npm run dev        # http://localhost:5180
```

Animation test page (dev only, not deployed):

```bash
cd frontend
npm run anim       # http://localhost:5190, one button per animation
```

## Deploy on Render

`render.yaml` defines two services:

- **trustlens**: the website (static, built from `frontend/`). The animation test page is never published.
- **trustlens-api**: the backend running the full pipeline.

In Render: Blueprints → the TrustLens blueprint → **Manual sync**. Render asks for the values marked `sync: false`:

- `GEMINI_API_KEY`, `GROQ_API_KEY` (and `BRAVE_API_KEY`, or leave it empty) on **trustlens-api**
- `VITE_API_BASE_URL` on **trustlens**: the backend's URL, e.g. `https://trustlens-api-xxxx.onrender.com`

Keys live only in Render's environment settings, never in git. On the free plan the backend sleeps when idle (first request takes about a minute to wake) and SQLite history resets on each deploy.

## API keys

Copy `backend/.env.example` to `backend/.env` and fill in:

- `GEMINI_API_KEY`: aistudio.google.com → Get API key (free tier). A Gemini app subscription does not include API access.
- `GROQ_API_KEY`: console.groq.com → API Keys (free tier).
- `GROQ_VERIFIER_MODEL`: a current **non-Llama** model from console.groq.com/docs/models.

Keys stay in `.env` only, never in git or chat.
