from __future__ import annotations

import multiprocessing as mp
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import BaseModel, Field, RootModel

from dronalize.config.loader import LoaderConfig  # noqa: TC001
from dronalize.config.map import MapConfig
from dronalize.config.writer import WriterConfig

if TYPE_CHECKING:
    from pathlib import Path

TModel = TypeVar("TModel", bound=BaseModel)
ConfigOverride = dict[str, object]


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
    writer: WriterConfig = Field(default_factory=WriterConfig)


class _ConfigOverrideFile(RootModel[dict[str, ConfigOverride]]):
    """Validated raw view of a TOML override file."""


@dataclass(slots=True, frozen=True)
class ConfigOverrides:
    """Validated per-dataset config-file overrides with globals already applied."""

    datasets: dict[str, ConfigOverride] = field(default_factory=dict)

    def for_dataset(self, dataset_name: str) -> ConfigOverride:
        """Return a copy of the overrides for one dataset name."""
        return dict(self.datasets.get(dataset_name, {}))


def load_config_overrides(config_path: Path) -> ConfigOverrides:
    """Load and validate dataset override sections from a TOML config file."""
    with config_path.open("rb") as handle:
        config_file = _ConfigOverrideFile.model_validate(tomllib.load(handle)).root
    return ConfigOverrides(datasets=_apply_global_section(config_file))


def resolve_runtime_config(
    *,
    default: Config,
    overrides: ConfigOverride | None = None,
) -> Config:
    """Merge dataset overrides into a default runtime config and validate the result."""
    return _resolve_model(
        Config,
        default=default,
        overrides={} if overrides is None else overrides,
    )


def _resolve_model(
    model_cls: type[TModel],
    *,
    default: TModel | ConfigOverride,
    overrides: TModel | ConfigOverride,
) -> TModel:
    """Merge defaults with overrides and validate into the target model.

    Parameters
    ----------
    model_cls : type[TModel]
        The Pydantic model class to validate the final merged data against.
    default : TModel | ConfigOverride
        The default configuration to use as a base.
    overrides : TModel | ConfigOverride
        The configuration overrides. Deeply merged with the default config, taking priority.
    """
    merged_data = _deep_merge(_config_data(default), _config_data(overrides))
    return model_cls.model_validate(merged_data)


def _config_data(value: BaseModel | ConfigOverride) -> ConfigOverride:
    """Return plain mapping data from either a config model or a config section."""
    return value.model_dump() if isinstance(value, BaseModel) else value


def _apply_global_section(config_file: dict[str, ConfigOverride]) -> dict[str, ConfigOverride]:
    """Merge the optional `global` section into each dataset-specific section."""
    global_section: ConfigOverride = config_file.get("global", {})
    return {
        dataset_name: _deep_merge(global_section, section)
        for dataset_name, section in config_file.items()
        if dataset_name != "global"
    }


def _is_config_section(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(k, str) for k in value)


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> ConfigOverride:
    """Recursively merge two configuration mappings."""
    merged = dict(base)
    for key, value in overrides.items():
        base_value = merged.get(key)
        if _is_config_section(value) and _is_config_section(base_value):
            merged[key] = _deep_merge(dict(base_value), dict(value))
        else:
            merged[key] = value
    return merged
