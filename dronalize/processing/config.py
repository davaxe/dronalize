from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any, TypedDict

import tomllib
from pydantic import BaseModel, Field, RootModel

from dronalize.core.datatypes.loader_config import LoaderConfig
from dronalize.core.datatypes.map_config import MapConfig

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from pathlib import Path


class ExecutionConfig(BaseModel):
    """Pydantic model for execution configuration defaults."""

    parallel: bool = False
    workers: int = Field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    chunksize: int | None = None


class ExecutionConfigDict(TypedDict, total=False):
    """TypedDict for execution configuration overrides."""

    parallel: bool
    workers: int
    chunksize: int | None


class LoaderConfigDict(TypedDict, total=False):
    """TypedDict representing the structure of the loader configuration overrides."""

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


class ConfigDict(TypedDict, total=False):
    """TypedDict representing the structure of the configuration overrides.

    Each dataset section in the TOML config file corresponds to a `ConfigDict`,
    which may contain overrides for the loader and map configurations. The keys
    in this dictionary should match the fields of the `Config` model, and the
    values should match the corresponding `LoaderConfigDict` and `MapConfigDict`
    structures.
    """

    loader: LoaderConfigDict
    map: MapConfigDict
    execution: ExecutionConfigDict


class ConfigFile(RootModel[dict[str, ConfigDict]]):
    """Validated view of the dataset override file."""


class Config(BaseModel):
    """Pydantic model representing the structure of the configuration overrides."""

    loader: LoaderConfig
    map: MapConfig = Field(default_factory=MapConfig.default)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


def load_config(config_path: Path) -> dict[str, ConfigDict]:
    """Load, validate, and parse the configuration overrides from the given path."""
    with config_path.open("rb") as f:
        raw_data = tomllib.load(f)
    return ConfigFile.model_validate(raw_data).root


def resolve_config(default: Config | ConfigDict, overrides: Config | ConfigDict) -> Config:
    """Merge the default config with the overrides.

    Parameters
    ----------
    default : Config | ConfigDict
        The default configuration to use as a base.
    overrides : Config | ConfigDict
        The configuration overrides. This will be deeply merged with the
        default config, taking priority.
    """
    base_data = default.model_dump() if isinstance(default, Config) else default
    override_data = overrides.model_dump() if isinstance(overrides, Config) else overrides
    merged_data = _deep_merge(base_data, override_data)
    return Config.model_validate(merged_data)


def resolve_loader_config(
    default: LoaderConfig | LoaderConfigDict, overrides: LoaderConfig | LoaderConfigDict
) -> LoaderConfig:
    """Merge the default loader config with the overrides."""
    base_data = default.model_dump() if isinstance(default, LoaderConfig) else default
    override_data = overrides.model_dump() if isinstance(overrides, LoaderConfig) else overrides
    merged_data = _deep_merge(base_data, override_data)
    return LoaderConfig.model_validate(merged_data)


def resolve_map_config(
    default: MapConfig | MapConfigDict, overrides: MapConfig | MapConfigDict
) -> MapConfig:
    """Merge the default map config with the overrides."""
    base_data = default.model_dump() if isinstance(default, MapConfig) else default
    override_data = overrides.model_dump() if isinstance(overrides, MapConfig) else overrides
    merged_data = _deep_merge(base_data, override_data)
    return MapConfig.model_validate(merged_data)


def resolve_execution_config(
    default: ExecutionConfig | ExecutionConfigDict, overrides: ExecutionConfig | ExecutionConfigDict
) -> ExecutionConfig:
    """Merge the default execution config with the overrides."""
    base_data = default.model_dump() if isinstance(default, ExecutionConfig) else default
    override_data = overrides.model_dump() if isinstance(overrides, ExecutionConfig) else overrides
    merged_data = _deep_merge(base_data, override_data)
    return ExecutionConfig.model_validate(merged_data)


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
