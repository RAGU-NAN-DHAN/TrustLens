# TrustLens

AI fact-checker: Llama writes an answer, the answer is split into claims, evidence is retrieved and ranked, two independent verifiers judge each claim, and every false claim gets a checked correction.

Full plan and work split: `TrustLens_Context.md` (in the team chat). Contract changes M3 proposed: `docs/contract_changes_m3.md`.

## Status

| Part | Owner | Status |
|---|---|---|
| Verifiers (Gemini + Groq), disagreement rule, correction + checker, evidence highlighting, UI | M3 | Built. Runs on mock data; real API mode written but not yet run with keys |
| Generation, self-consistency, embeddings, trust model, evaluation | M1 | Not started (placeholders in use) |
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

## API keys

Copy `backend/.env.example` to `backend/.env` and fill in:

- `GEMINI_API_KEY`: aistudio.google.com → Get API key (free tier). A Gemini app subscription does not include API access.
- `GROQ_API_KEY`: console.groq.com → API Keys (free tier).
- `GROQ_VERIFIER_MODEL`: a current **non-Llama** model from console.groq.com/docs/models.

Keys stay in `.env` only, never in git or chat.
