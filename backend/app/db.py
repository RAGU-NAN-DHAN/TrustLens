"""M2: SQLite storage for analyses (history) and the semantic cache."""

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.errors import TrustLensError


class PersistenceError(TrustLensError):
    error_code = "PERSISTENCE_FAILED"


def _path() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./trustlens.db")
    path = url.removeprefix("sqlite:///")
    return str(Path(__file__).resolve().parent.parent / path) if not os.path.isabs(path) else path


@contextmanager
def connect():
    conn = sqlite3.connect(_path())
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS analyses (answer_id TEXT PRIMARY KEY, question TEXT, created_at TEXT, trust REAL, claim_count INTEGER, body TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS claim_cache (id INTEGER PRIMARY KEY AUTOINCREMENT, claim_text TEXT, context_hash TEXT, embedding TEXT, decision TEXT)"
        )
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        raise PersistenceError(f"database error: {type(exc).__name__}") from exc
    finally:
        conn.close()


def save_analysis(result: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO analyses VALUES (?, ?, ?, ?, ?, ?)",
            (
                result["answer_id"],
                result.get("question"),
                result.get("created_at"),
                result["trust_score"]["trust_score"],
                len(result["claims"]),
                json.dumps(result),
            ),
        )


def get_analysis(answer_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT body FROM analyses WHERE answer_id = ?", (answer_id,)).fetchone()
    return json.loads(row[0]) if row else None


def list_history(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT answer_id, question, trust, claim_count, created_at FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"answer_id": r[0], "question": r[1], "trust_score": r[2], "claim_count": r[3], "created_at": r[4]} for r in rows]
