from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any, TypeVar

import tomllib
from pydantic import BaseModel, Field, RootModel

from dronalize.config.map import MapConfig

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from dronalize.config.loader import LoaderConfig

# Define a TypeVar bound to BaseModel for generic type hinting
TModel = TypeVar("TModel", bound=BaseModel)


class ExecutionConfig(BaseModel):
    """Pydantic model for execution configuration defaults."""

    parallel: bool = False
    workers: int = Field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    chunksize: int | None = None


class Config(BaseModel):
    """Pydantic model representing the structure of the configuration overrides."""

    loader: LoaderConfig
    map: MapConfig = Field(default_factory=MapConfig.default)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


class ConfigFile(RootModel[dict[str, dict[str, Any]]]):
    """Validated view of the dataset override file."""


def load_config(config_path: Path) -> dict[str, dict[str, Any]]:
    """Load, validate, and parse the configuration overrides from the given path."""
    with config_path.open("rb") as f:
        raw_data = tomllib.load(f)
    return ConfigFile.model_validate(raw_data).root


def resolve_config(
    model_cls: type[TModel], default: TModel | dict[str, Any], overrides: TModel | dict[str, Any]
) -> TModel:
    """Merge a default config model/dict with overrides and validate into the target model.

    Parameters
    ----------
    model_cls : type[TModel]
        The Pydantic model class to validate the final merged data against.
    default : TModel | dict[str, Any]
        The default configuration to use as a base.
    overrides : TModel | dict[str, Any]
        The configuration overrides. Deeply merged with the default config, taking priority.
    """
    base_data = default.model_dump() if isinstance(default, BaseModel) else default
    override_data = overrides.model_dump() if isinstance(overrides, BaseModel) else overrides

    merged_data = _deep_merge(base_data, override_data)
    return model_cls.model_validate(merged_data)


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
