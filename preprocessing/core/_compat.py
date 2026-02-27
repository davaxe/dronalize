"""Lazy-import helpers for optional dependencies."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def require_optional(module_name: str, *, extra: str | None = None) -> ModuleType:
    """Import and return an optional dependency, raising a clear error if missing.

    Args:
        module_name: fully-qualified module name (e.g. `"torch"`).
        extra: pip extras name to suggest in the error message.
            If `None`, the suggestion defaults to
            `pip install <module_name>`.

    Returns:
        The imported module.

    Raises:
        ImportError: if the module is not installed.

    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        install_target = f"dronalize[{extra}]" if extra else module_name
        msg = (
            f"{module_name!r} is required for this functionality. "
            f"Install it with: pip install {install_target}"
        )
        raise ImportError(msg) from exc
