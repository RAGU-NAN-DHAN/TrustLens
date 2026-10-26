"""Structured logging helpers shared across M1 modules.

All logs are emitted as single-line JSON so they are easy to grep and ship
to a log aggregator later. Nothing in this module ever logs a secret value;
callers must pass already-redacted data.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("trustlens_m1")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger under the ``trustlens_m1`` namespace."""
    _configure_root_logger()
    return logging.getLogger(f"trustlens_m1.{name}")


_REDACT_KEYS = {"api_key", "authorization", "token", "secret", "groq_api_key"}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: ("***REDACTED***" if k.lower() in _REDACT_KEYS else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit one structured, credential-safe log line.

    Example: ``log_event(logger, logging.INFO, "generation.start", model=model)``
    """
    payload = {
        "event": event,
        "ts": time.time(),
        **_redact(fields),
    }
    logger.log(level, json.dumps(payload, default=str))
