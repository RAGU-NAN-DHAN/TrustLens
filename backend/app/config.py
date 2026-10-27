import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    groq_api_key: str
    gemini_verifier_model: str
    gemini_flash_model: str
    groq_verifier_model: str
    mock: bool
    http_timeout: float


def get_settings() -> Settings:
    # read at call time so tests and scripts can change the environment
    return Settings(
        # GEMINI_API_KEYS="k1,k2,..." gives backup keys, tried in order when one hits its limit
        gemini_api_key=",".join(k for k in [os.getenv("GEMINI_API_KEY", "").strip()] + os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        gemini_verifier_model=os.getenv("GEMINI_VERIFIER_MODEL", "gemini-3.5-flash").strip(),
        gemini_flash_model=os.getenv("GEMINI_FLASH_MODEL", "gemini-3.5-flash-lite").strip(),
        groq_verifier_model=os.getenv("GROQ_VERIFIER_MODEL", "").strip(),
        mock=_flag("TRUSTLENS_MOCK", "1"),
        http_timeout=float(os.getenv("TRUSTLENS_HTTP_TIMEOUT", "25")),
    )
