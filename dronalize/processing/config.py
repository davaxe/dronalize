from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

import tomllib

from dronalize.core.datatypes.loader_config import LoaderConfig

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from pathlib import Path


class ConfigDict(TypedDict, total=False):
    """TypedDict representing the structure of the configuration overrides."""

    input_len: int
    output_len: int
    sample_time: float
    resampling: ResamplingDict
    filtering: FilteringDict
    window: WindowDict


class ResamplingDict(TypedDict, total=False):
    """TypedDict for resampling configuration."""

    up: int
    down: int
    method: str


class FilteringDict(TypedDict, total=False):
    """TypedDict for filtering configuration."""

    min_agents: int
    require_all_valid: bool
    require_frames: Collection[int]
    filter_agent_category: Collection[int | str]
    filter_slow_agents: float
    min_samples_per_agent: int


class WindowDict(TypedDict, total=False):
    """TypedDict for windowing configuration."""

    step_size: int
    window_size: int


def load_overrides(config_path: Path) -> dict[str, ConfigDict]:
    """Load toml config overrides from the given path."""
    with config_path.open("rb") as f:
        return cast("dict[str, ConfigDict]", tomllib.load(f))


def resolve_config(default: LoaderConfig, overrides: ConfigDict) -> LoaderConfig:
    """Merge the default config with the overrides.

    Parameters
    ----------
    default : LoaderConfig
        The default configuration to use as a base.
    overrides : ConfigDict
        The configuration overrides loaded from the toml file. This will be
        deeply merged with the default config.
    """
    merged_data = _deep_merge(default.model_dump(), overrides)
    return LoaderConfig.model_validate(merged_data)


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in overrides.items():
        # If both the base value and the override value are dicts, merge them recursively
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        # Otherwise, the override completely replaces the base value (or creates a new key)
        else:
            merged[key] = value
    return merged
