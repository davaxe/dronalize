"""Shared helpers for lightweight lazy package exports."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def resolve_lazy_export(
    module_globals: dict[str, object],
    exports: Mapping[str, tuple[str, str]],
    *,
    module_name: str,
    name: str,
) -> object:
    """Resolve one lazily exported symbol and cache it in module globals."""
    if name not in exports:
        msg = f"module '{module_name}' has no attribute '{name}'"
        raise AttributeError(msg)

    target_module, export_name = exports[name]
    value = getattr(importlib.import_module(target_module), export_name)
    module_globals[name] = value
    return value


def lazy_dir(module_globals: Mapping[str, object], exported_names: list[str]) -> list[str]:
    """Return a sorted directory listing including lazy exports."""
    return sorted(set(module_globals) | set(exported_names))
