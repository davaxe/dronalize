"""Lazy-import helpers for optional dependencies."""

from __future__ import annotations

from typing import NoReturn


def raise_missing_optional_dependency(
    error: ModuleNotFoundError, *, feature: str, extra: str
) -> NoReturn:
    """Raise a friendlier import error for optional storage features."""
    msg = (
        f"{feature} requires optional dependencies that are not installed. "
        f"Install them with `pip install dronalize[{extra}]`."
    )
    raise ModuleNotFoundError(msg) from error
