"""Loads .env into the process environment if python-dotenv is available
and a .env file exists. Call this once, early, in any entry point that
reads API keys from the environment (test scripts, api.py, a future CLI
runner) -- it's a no-op if there's no .env file or the package isn't
installed, so it's always safe to call."""

from __future__ import annotations


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed -- fall back to real env vars only