# TrustLens

AI fact-checker: Llama writes an answer, the answer is split into claims, evidence is retrieved and ranked, two independent verifiers judge each claim, and every false claim gets a checked correction.

Full plan and work split: `TrustLens_Context.md` (in the team chat). Contract changes M3 proposed: `docs/contract_changes_m3.md`.

## Status

| Part | Owner | Status |
|---|---|---|
| Verifiers (Gemini + Groq), disagreement rule, correction + checker, evidence highlighting, UI | M3 | Built and tested with real Gemini 3.5 Flash + Groq qwen3.8-27b |
| Generation, self-consistency, embeddings, trust model, evaluation | M1 | Built separately (trustlens-m1, 70 tests), not merged yet |
| Claim splitting, retrieval, ranking, cache, server, database | M2 | Not started (hand-written demo claims and evidence in use) |

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

`render.yaml` deploys the website as a static site from `frontend/`. Only `frontend/index.html` is built, so the animation test page is never published. Until M2's backend is online the site runs in demo mode with the saved real-API run.

## API keys

Copy `backend/.env.example` to `backend/.env` and fill in:

- `GEMINI_API_KEY`: aistudio.google.com → Get API key (free tier). A Gemini app subscription does not include API access.
- `GROQ_API_KEY`: console.groq.com → API Keys (free tier).
- `GROQ_VERIFIER_MODEL`: a current **non-Llama** model from console.groq.com/docs/models.

Keys stay in `.env` only, never in git or chat.
