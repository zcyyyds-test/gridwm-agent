"""Localhost launcher for the wm-agent world-model HTTP API.

Boots the V2 latent graph world model into a FastAPI app and serves it on
``http://127.0.0.1:8000`` by default. Localhost only, no auth.
"""
from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to outputs/run_<id> (sets GRIDWM_RUN_DIR env var).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.run_dir:
        os.environ["GRIDWM_RUN_DIR"] = args.run_dir

    uvicorn.run(
        "wmagent.serve.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
