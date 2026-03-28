"""Lazy-import helpers for optional dependencies."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, NoReturn

from dronalize.core.errors import MissingOptionalDependencyError

if TYPE_CHECKING:
    from types import ModuleType


def require_optional(module_name: str, *, extra: str | None = None) -> ModuleType:
    """Import and return an optional dependency.

    Parameters
    ----------
    module_name : str
        Fully-qualified module name (e.g. `"torch"`).
    extra : str, optional
        pip extras name to suggest in the error message. If `None`, the
        suggestion defaults to `pip install <module_name>`.

    Returns
    -------
    ModuleType
        The imported module.

    Raises
    ------
    MissingOptionalDependencyError
        If the module is not installed.

    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        install_target = f"dronalize[{extra}]" if extra else module_name
        msg = (
            f"{module_name!r} is required for this functionality. "
            f"Install {install_target} to use it."
        )
        raise MissingOptionalDependencyError(
            msg,
            dependencies=(module_name,),
            install_target=install_target,
        ) from exc


def raise_missing_optional_dependency(
    error: ModuleNotFoundError,
    *,
    feature: str,
    extra: str,
) -> NoReturn:
    """Raise a friendlier import error for optional storage features."""
    msg = (
        f"{feature} requires optional dependencies that are not installed. "
        f"Install them with `pip install dronalize[{extra}]`."
    )
    raise ModuleNotFoundError(msg) from error
