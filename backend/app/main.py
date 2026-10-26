"""M2: FastAPI server. The UI only ever talks to these routes."""

import os

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import db
from app.errors import TrustLensError
from app.models.schemas import AnalysisResult, AnalyzeRequest, HistoryItem
from app.pipeline import analyze

app = FastAPI(title="TrustLens API", version="0.1.0")

origins = [o.strip() for o in os.getenv("TRUSTLENS_CORS_ORIGINS", "http://localhost:5180,http://127.0.0.1:5180").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error_code": code, "message": message, "detail": None})


@app.exception_handler(TrustLensError)
async def trustlens_error(_: Request, exc: TrustLensError) -> JSONResponse:
    return _error(502, exc.error_code, str(exc))


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    return _error(422, "INVALID_REQUEST", str(first.get("msg", "invalid request")))


@app.exception_handler(Exception)
async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    return _error(500, "INTERNAL_ERROR", f"unexpected server error: {type(exc).__name__}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/analyze", response_model=AnalysisResult)
async def analyze_route(body: AnalyzeRequest) -> dict:
    return await run_in_threadpool(analyze, body.question, body.answer_text)


@app.get("/api/v1/analysis/{answer_id}", response_model=AnalysisResult)
def get_analysis(answer_id: str):
    found = db.get_analysis(answer_id)
    if found is None:
        return _error(404, "ANALYSIS_NOT_FOUND", f"No analysis with id {answer_id}.")
    return found


@app.get("/api/v1/history", response_model=list[HistoryItem])
def history() -> list[dict]:
    return db.list_history()
