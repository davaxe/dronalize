from __future__ import annotations

import multiprocessing as mp
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, cast

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from pydantic import BaseModel, Field, RootModel

from dronalize.io.config import WriterConfig
from dronalize.processing.filters import Filter, FilterSpec
from dronalize.processing.ingest.config import LoaderConfig  # noqa: TC001
from dronalize.processing.ingest.splits import SplitConfig
from dronalize.processing.maps.config import MapConfig

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.io.config import SceneSchemaLike

TModel = TypeVar("TModel", bound=BaseModel)
ConfigOverride = dict[str, object]

__all__ = [
    "Config",
    "ConfigOverrides",
    "ExecutionConfig",
    "load_config_overrides",
    "resolve_runtime_config",
]


class ExecutionConfig(BaseModel):
    """Pydantic model for execution configuration defaults."""

    parallel: bool = False
    workers: int | None = Field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    chunksize: int | None = None

    def with_jobs(self, jobs: int) -> ExecutionConfig:
        """Return a copy with a runtime worker override applied."""
        if jobs == -1:
            return self.model_copy(update={"parallel": True, "workers": None})
        if jobs < 1:
            msg = "jobs must be at least 1."
            raise ValueError(msg)
        return self.model_copy(
            update={
                "parallel": jobs > 1,
                "workers": jobs,
            }
        )


class Config(BaseModel):
    """Pydantic model representing the structure of the configuration overrides."""

    loader: LoaderConfig
    map: MapConfig = Field(default_factory=MapConfig.default)
    split: SplitConfig = Field(default_factory=SplitConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    writer: WriterConfig = Field(default_factory=WriterConfig)

    def with_scene_schema(self, scene_schema: SceneSchemaLike | None) -> Config:
        """Return a copy with a writer scene-schema override applied."""
        if scene_schema is None:
            return self
        return self.model_copy(update={"writer": self.writer.with_scene_schema(scene_schema)})

    def with_jobs(self, jobs: int | None) -> Config:
        """Return a copy with runtime execution overrides applied."""
        if jobs is None:
            return self
        return self.model_copy(update={"execution": self.execution.with_jobs(jobs)})


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
    merged_data = _deep_merge(_config_data(default), _config_data(overrides or {}))
    merged_data = _resolve_loader_filter(merged_data, default_filter=default.loader.filter)
    return Config.model_validate(merged_data)


def _config_data(value: BaseModel | ConfigOverride) -> ConfigOverride:
    """Return plain mapping data from either a config model or a config section."""
    if not isinstance(value, BaseModel):
        return value
    return {name: _config_data(getattr(value, name)) for name in type(value).model_fields}


def _resolve_loader_filter(
    merged_data: ConfigOverride,
    *,
    default_filter: Filter | None,
) -> ConfigOverride:
    """Resolve config-facing filter specs into runtime filter objects."""
    loader_value = merged_data.get("loader")
    if not _is_config_section(loader_value):
        return merged_data

    filter_value = loader_value.get("filter")
    if filter_value is None or isinstance(filter_value, Filter):
        return merged_data
    if not _is_config_section(filter_value):
        return merged_data

    resolved_filter = FilterSpec.model_validate(filter_value).resolve(default_filter)
    resolved_loader = dict(loader_value)
    resolved_loader["filter"] = resolved_filter

    resolved_config = dict(merged_data)
    resolved_config["loader"] = resolved_loader
    return resolved_config


def _apply_global_section(config_file: dict[str, ConfigOverride]) -> dict[str, ConfigOverride]:
    """Merge the optional `global` section into each dataset-specific section."""
    global_section: ConfigOverride = config_file.get("global", {})
    return {
        dataset_name: _deep_merge(global_section, section)
        for dataset_name, section in config_file.items()
        if dataset_name != "global"
    }


def _is_config_section(value: object) -> TypeGuard[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return False
    mapping = cast("Mapping[object, object]", value)
    return all(isinstance(key, str) for key in mapping)


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
