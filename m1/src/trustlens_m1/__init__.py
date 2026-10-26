"""TrustLens M1 module: generation, self-consistency, embeddings, and trust scoring.

This package is owned by M1 and is designed to be imported independently by
M2 (FastAPI server, evidence pipeline) and M3 (verifiers, disagreement rule,
correction checking, UI) without pulling in a server or frontend.

See M1_INTEGRATION_CONTRACT.md at the repository root for the full contract.
"""

__version__ = "0.1.0"
