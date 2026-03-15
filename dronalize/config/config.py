from __future__ import annotations

import multiprocessing as mp
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar

from pydantic import BaseModel, Field, RootModel

from dronalize.config.loader import LoaderConfig  # noqa: TC001 (needed for pydantic)
from dronalize.config.map import MapConfig

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

if TYPE_CHECKING:
    from pathlib import Path


# Define a TypeVar bound to BaseModel for generic type hinting
TModel = TypeVar("TModel", bound=BaseModel)
ConfigSection = dict[str, object]


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


class ConfigFile(RootModel[dict[str, ConfigSection]]):
    """Validated view of the dataset override file."""


def load_config(config_path: Path) -> dict[str, ConfigSection]:
    """Load, validate, and parse the configuration overrides from the given path."""
    with config_path.open("rb") as f:
        config_file = ConfigFile.model_validate(tomllib.load(f)).root
    return _apply_global_section(config_file)


def resolve_config(
    model_cls: type[TModel],
    default: TModel | ConfigSection,
    overrides: TModel | ConfigSection,
) -> TModel:
    """Merge a default config model/dict with overrides and validate into the target model.

    Parameters
    ----------
    model_cls : type[TModel]
        The Pydantic model class to validate the final merged data against.
    default : TModel | ConfigSection
        The default configuration to use as a base.
    overrides : TModel | ConfigSection
        The configuration overrides. Deeply merged with the default config, taking priority.
    """
    merged_data = _deep_merge(_config_data(default), _config_data(overrides))
    return model_cls.model_validate(merged_data)


def _config_data(value: TModel | ConfigSection) -> ConfigSection:
    """Return plain mapping data from either a config model or a config section."""
    return value.model_dump() if isinstance(value, BaseModel) else value


def _apply_global_section(config_file: dict[str, ConfigSection]) -> dict[str, ConfigSection]:
    """Merge the optional `global` section into each dataset-specific section."""
    global_section: ConfigSection = config_file.get("global", {})
    return {
        dataset_name: _deep_merge(global_section, section)
        for dataset_name, section in config_file.items()
        if dataset_name != "global"
    }


def _is_config_section(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(k, str) for k in value)


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> ConfigSection:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in overrides.items():
        base_value = merged.get(key)
        if _is_config_section(value) and _is_config_section(base_value):
            merged[key] = _deep_merge(dict(base_value), dict(value))
        else:
            merged[key] = value
    return merged
