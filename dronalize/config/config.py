from __future__ import annotations

import multiprocessing as mp
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar

from pydantic import BaseModel, Field, RootModel

from dronalize.config.map import MapConfig

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config.loader import LoaderConfig

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
        raw_data = tomllib.load(f)
    config_file = ConfigFile.model_validate(raw_data).root
    global_section: ConfigSection = config_file.get("global", {})
    return {k: _deep_merge(global_section, v) for k, v in config_file.items() if k != "global"}


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
    base_data = default.model_dump() if isinstance(default, BaseModel) else default
    override_data = overrides.model_dump() if isinstance(overrides, BaseModel) else overrides

    merged_data = _deep_merge(base_data, override_data)
    return model_cls.model_validate(merged_data)


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


if __name__ == "__main__":
    from pathlib import Path

    path = Path("config.toml")
    config = load_config(path)
