"""Bootstrap real deployment dependencies and start the HTTP server."""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv


def _run_mode() -> str:
    explicit = os.environ.get("APP_RUN_MODE", "").strip().casefold()
    if explicit:
        return explicit
    if (
        os.environ.get("LLM_API_KEY", "").strip()
        and os.environ.get("LLM_MODEL", "").strip()
    ):
        return "real"
    return "demo"


def main() -> None:
    load_dotenv()
    if _run_mode() == "real":
        from api.bootstrap import ensure_real_index

        ensure_real_index()
    uvicorn.run(
        "api.deployment:app",
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    main()
