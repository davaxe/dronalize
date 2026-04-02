"""Authoring-time configuration models loaded from project files."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dronalize.core.categories import DatasetSplit  # noqa: TC001
from dronalize.core.errors import ConfigurationError, LoaderConfigError, SplitError
from dronalize.io.config import SceneSchemaLike, WriterPrecision  # noqa: TC001
from dronalize.processing.filters.filter import AgentCheckSpecs, CleanupSpecs, SceneCheckSpecs
from dronalize.processing.filters.filter import FilterSpec as RuntimeFilterSpec
from dronalize.processing.ingest.splits import SplitModeName, SplitWeights  # noqa: TC001
from dronalize.processing.pipeline.functional.resample._common import AliasedResampling, ColumnOrder

if TYPE_CHECKING:
    from pathlib import Path


DerivativeMap = dict[int, ColumnOrder | Sequence[str]]
_MODES_REQUIRING_RATIO = {"time", "shuffled-time", "scene", "source", "auto"}
_MODES_WITH_GAP = {"time", "shuffled-time"}


class _FileModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class FileExecutionConfig(_FileModel):
    """Execution settings loaded from project config files."""

    jobs: int | Literal["auto"] | None = None
    chunksize: int | None = None

    @model_validator(mode="after")
    def _validate_jobs(self) -> FileExecutionConfig:
        if self.jobs is not None and self.jobs != "auto" and self.jobs < 1:
            msg = "jobs must be a positive integer or 'auto'."
            raise ConfigurationError(msg)
        return self


class FileWindowConfig(_FileModel):
    """Sliding-window settings loaded from project config files."""

    size: int
    step: int


class FileDerivativeColumnsConfig(_FileModel):
    """One derivative-order entry in file-based resampling config."""

    order: int = Field(ge=1)
    columns: ColumnOrder | Sequence[str]


class FileResamplingConfig(_FileModel):
    """Resampling settings loaded from project config files."""

    up: int | None = None
    down: int | None = None
    method: AliasedResampling | None = None
    position_columns: ColumnOrder | Sequence[str] | None = None
    input_derivatives: tuple[FileDerivativeColumnsConfig, ...] = ()
    output_derivatives: tuple[FileDerivativeColumnsConfig, ...] = ()
    max_gap: int | None = None
    sort: bool | None = None
    sample_time: float | None = None

    def input_derivative_map(self) -> DerivativeMap | None:
        """Return input derivatives keyed by derivative order, if configured."""
        return _derivative_map(self.input_derivatives)

    def output_derivative_map(self) -> DerivativeMap | None:
        """Return output derivatives keyed by derivative order, if configured."""
        return _derivative_map(self.output_derivatives)


class FileLoaderFilterConfig(_FileModel):
    """Filter settings loaded from project config files."""

    mode: Literal["replace", "extend"] | None = None
    remove: tuple[str, ...] = Field(default_factory=tuple)
    cleanup: CleanupSpecs = Field(default_factory=tuple)
    scene: SceneCheckSpecs = Field(default_factory=tuple)
    agent: AgentCheckSpecs = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_mode(self) -> FileLoaderFilterConfig:
        has_rules = bool(self.cleanup or self.scene or self.agent or self.remove)
        if has_rules and self.mode is None:
            msg = "filter mode is required when filter rules or remove entries are present."
            raise LoaderConfigError(msg)
        return self

    def rules(self) -> RuntimeFilterSpec:
        """Return the configured filter rules in runtime filter-spec form."""
        return RuntimeFilterSpec(cleanup=self.cleanup, scene=self.scene, agent=self.agent)


class FileLoaderConfig(_FileModel):
    """Loader settings loaded from project config files."""

    input_len: int | None = None
    output_len: int | None = None
    sample_time: float | None = None
    window: FileWindowConfig | None = None
    resampling: FileResamplingConfig | None = None
    filter: FileLoaderFilterConfig | None = None
    highway: dict[str, Any] | None = None
    options: dict[str, Any] | None = None


class FileMapConfig(_FileModel):
    """Map settings loaded from project config files."""

    enabled: bool | None = None
    min_distance: float | None = None
    interp_distance: float | None = None
    extraction: Literal["full", "relevant", "circle", "bounding_box"] | None = None
    padding: float | None = None
    radius: float | None = None
    width: float | None = None
    height: float | None = None

    @model_validator(mode="after")
    def _validate_extraction(self) -> FileMapConfig:
        if self.extraction == "relevant" and self.padding is None:
            msg = "map padding is required when extraction='relevant'."
            raise ConfigurationError(msg)
        if self.extraction == "circle" and self.radius is None:
            msg = "map radius is required when extraction='circle'."
            raise ConfigurationError(msg)
        if self.extraction == "bounding_box" and (self.width is None or self.height is None):
            msg = "map width and height are required when extraction='bounding_box'."
            raise ConfigurationError(msg)
        if self.extraction in {None, "full"} and any(
            value is not None for value in (self.padding, self.radius, self.width, self.height)
        ):
            msg = "map extraction-specific fields require an explicit extraction mode."
            raise ConfigurationError(msg)
        return self


class FileMDSConfig(_FileModel):
    """Mosaic Streaming writer settings loaded from project config files."""

    compression: str | None = None
    hashes: tuple[str, ...] | None = None
    size_limit: str | int | None = None
    exist_ok: bool | None = None


class FileWriterConfig(_FileModel):
    """Writer settings loaded from project config files."""

    scene_schema: SceneSchemaLike | None = Field(default=None, alias="schema")
    precision: WriterPrecision | None = None
    offset_positions: bool | None = None
    mds: FileMDSConfig | None = None


class FileSplitConfig(_FileModel):
    """Split settings loaded from project config files."""

    mode: SplitModeName | None = None
    ratio: SplitWeights | None = None
    gap: int | None = Field(default=None, ge=0)
    segments: int | None = Field(default=None, ge=1)
    read: tuple[DatasetSplit, ...] | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> FileSplitConfig:
        if self.mode is None:
            if any(value is not None for value in (self.ratio, self.gap, self.segments, self.read)):
                msg = "split mode is required when split settings are present."
                raise SplitError(msg)
            return self

        if self.mode in _MODES_REQUIRING_RATIO and self.ratio is None:
            msg = f"split ratio is required when mode='{self.mode}'."
            raise SplitError(msg)
        if self.gap is not None and self.mode not in _MODES_WITH_GAP:
            msg = "split gap is only valid for time and shuffled-time modes."
            raise SplitError(msg)
        if self.mode == "shuffled-time" and self.segments is None:
            msg = "split segments are required when mode='shuffled-time'."
            raise SplitError(msg)
        if self.mode != "shuffled-time" and self.segments is not None:
            msg = "split segments are only valid for mode='shuffled-time'."
            raise SplitError(msg)
        if self.mode != "native" and self.read is not None:
            msg = "split read is only valid for mode='native'."
            raise SplitError(msg)
        if self.mode in {"native", "none"} and self.ratio is not None:
            msg = f"split ratio is not valid when mode='{self.mode}'."
            raise SplitError(msg)
        return self


class FileDatasetConfig(_FileModel):
    """One dataset/global config block loaded from a project config file."""

    loader: FileLoaderConfig | None = None
    map: FileMapConfig | None = None
    split: FileSplitConfig | None = None
    execution: FileExecutionConfig | None = None
    writer: FileWriterConfig | None = None


class ConfigFile(BaseModel):
    """Typed project config file with global and dataset-specific blocks."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True
    )

    global_: FileDatasetConfig = Field(default_factory=FileDatasetConfig, alias="global")
    datasets: dict[str, FileDatasetConfig] = Field(default_factory=dict)

    def dataset_config(self, dataset_name: str) -> FileDatasetConfig:
        """Return the file-loaded config block for one dataset."""
        return self.datasets.get(dataset_name, FileDatasetConfig())


def load_project_config(config_path: Path) -> ConfigFile:
    """Load a project config file using the file-facing config schema."""
    with config_path.open("rb") as handle:
        return ConfigFile.model_validate(tomllib.load(handle))


def _derivative_map(entries: tuple[FileDerivativeColumnsConfig, ...]) -> DerivativeMap | None:
    if not entries:
        return None
    return {entry.order: entry.columns for entry in entries}
