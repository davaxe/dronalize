from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

import tomllib
from pydantic import BaseModel, Field

from dronalize.core.datatypes.loader_config import LoaderConfig
from dronalize.core.datatypes.map_config import MapConfig

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from pathlib import Path


class Config(BaseModel):
    """Pydantic model representing the structure of the configuration overrides."""

    loader: LoaderConfig
    map: MapConfig = Field(default_factory=MapConfig.default)


class ConfigDict(TypedDict, total=False):
    """TypedDict representing the structure of the configuration overrides.

    Each dataset section in the TOML config file corresponds to a `ConfigDict`, which
    may contain overrides for the loader and map configurations. The keys in this
    dictionary should match the fields of the `Config` model, and the values should
    match the corresponding `LoaderConfigDict` and `MapConfigDict` structures.

    """

    loader: LoaderConfigDict
    map: MapConfigDict


class LoaderConfigDict(TypedDict, total=False):
    """TypedDict representing the structure of the configuration overrides."""

    input_len: int
    output_len: int
    sample_time: float
    resampling: ResamplingDict
    filtering: FilteringDict
    window: WindowDict
    extra_kwargs: dict[str, Any]


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


class MapConfigDict(TypedDict, total=False):
    """TypedDict for map configuration."""

    min_distance: float
    interp_distance: float
    include_map: bool


def load_config(config_path: Path) -> dict[str, ConfigDict]:
    """Load and parse the configuration overrides from the given path."""
    with config_path.open("rb") as f:
        raw_data = tomllib.load(f)
    return cast("dict[str, ConfigDict]", raw_data)


def resolve_loader_config(default: LoaderConfig, overrides: LoaderConfigDict) -> LoaderConfig:
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


def resolve_map_config(default: MapConfig, overrides: MapConfigDict) -> MapConfig:
    """Merge the default map config with the overrides."""
    merged_data = _deep_merge(default.model_dump(), overrides)
    return MapConfig.model_validate(merged_data)


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


if __name__ == "__main__":
    from pathlib import Path

    from rich import print as rprint

    default = Config(
        loader=LoaderConfig(input_len=10, output_len=20, sample_time=0.1),
        map=MapConfig(min_distance=1, interp_distance=2, include_map=True),
    )

    config_path = Path("config.toml")
    config = load_config(config_path)
    a43_config = config["a43"]
    rprint(a43_config)
    rprint(default.loader)
    resolved_loader_config = resolve_loader_config(default.loader, a43_config.get("loader", {}))
    rprint(resolved_loader_config)
    rprint(default.map)
    resolved_map_config = resolve_map_config(default.map, a43_config.get("map", {}))
    rprint(resolved_map_config)
