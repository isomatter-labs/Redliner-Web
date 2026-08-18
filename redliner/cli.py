"""Command line entry point."""

from __future__ import annotations

import argparse

from .ui.app import main


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="redliner", description="Per-pixel document comparison server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="interface to bind (default: all)")
    parser.add_argument("--port", type=int, default=8080, help="port to listen on")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = parser.parse_args()
    main(host=args.host, port=args.port, reload=args.reload)
