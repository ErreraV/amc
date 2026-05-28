"""Small environment helpers to centralize dotenv loading and typed getters.

This module is optional — it will silently skip loading python-dotenv if
it's not installed. The code in the repo will call `ensure_loaded()` once
and then use `get()`/`getint()` to read settings.
"""
from pathlib import Path
import os
from typing import Any, Callable, Optional


def ensure_loaded() -> None:
    """If python-dotenv is available, load any .env files from cwd/parents."""
    try:
        from dotenv import load_dotenv

        # allow dotenv to discover the .env file in the project root
        load_dotenv(override=False)
    except Exception:
        # python-dotenv not installed or failed — that's fine, we rely on
        # environment inherited from the shell / Makefile.
        return


def _cast(value: str, caster: Optional[Callable[[str], Any]]):
    if caster is None:
        return value
    try:
        return caster(value)
    except Exception:
        return None


def get(key: str, default: Optional[str] = None, cast: Optional[Callable[[str], Any]] = None) -> Any:
    v = os.getenv(key, default)
    if v is None:
        return default
    casted = _cast(v, cast)
    return casted if cast is not None else v


def getint(key: str, default: int) -> int:
    return int(get(key, default, cast=int) or default)


def getbool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")
