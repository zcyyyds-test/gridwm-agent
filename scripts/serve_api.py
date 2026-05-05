"""Localhost launcher for the gridwm-agent world-model HTTP API.

Boots the V2 model into a FastAPI app and serves it on
``http://127.0.0.1:8000`` by default. Localhost only, no auth, no TLS — do
not bind to a routable address without an upstream reverse proxy that
handles auth and rate-limiting.
"""
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to outputs/<run_id> (sets GRIDWM_RUN_DIR env var).")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Bind address. Keep on 127.0.0.1 unless you have an "
                             "auth proxy in front.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.run_dir:
        os.environ["GRIDWM_RUN_DIR"] = args.run_dir

    uvicorn.run(
        "wmagent.serve.api:app",
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
