"""Precedence and normalization logic for runtime configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import (
    ConfigurationError,
    LoaderConfigError,
    SplitConflictError,
    SplitNotSupportedError,
    SplitStrategyNotSupportedError,
)
from dronalize.core.scene import get_scene_schema
from dronalize.io.config import MDSFormatConfig
from dronalize.io.config import WriterConfig as RuntimeWriterConfig
from dronalize.processing.filters import Filter, merge_filters
from dronalize.processing.filters.filter import remove_filter_rules
from dronalize.processing.ingest.base import LoaderOptions, NoLoaderOptions
from dronalize.processing.ingest.config import HighwayParams, WindowConfig
from dronalize.processing.ingest.config import LoaderConfig as RuntimeLoaderConfig
from dronalize.processing.ingest.splits import (
    BySceneSplit,
    BySourceSplit,
    NativeSplit,
    NativeSplitSelection,
    ShuffledTimeBlockSplit,
    SplitMode,
    SplitModeName,
    SplitWeights,
    TimeBlockSplit,
    Unsplit,
)
from dronalize.processing.ingest.splits import SplitConfig as RuntimeSplitConfig
from dronalize.processing.maps.config import (
    BoundingBoxExtraction,
    CircularExtraction,
    FullMapExtraction,
    RelevantAreaExtraction,
)
from dronalize.processing.maps.config import MapConfig as RuntimeMapConfig
from dronalize.processing.pipeline.functional.resample import ResampleSpec
from dronalize.runtime.config.overrides import PlanOverrides
from dronalize.runtime.config.resolved import ResolvedConfig, ResolvedExecutionConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.datasets.registry import DatasetDescriptor
    from dronalize.runtime.config.file import (
        ConfigFile,
        FileDatasetConfig,
        FileExecutionConfig,
        FileLoaderConfig,
        FileLoaderFilterConfig,
        FileMapConfig,
        FileResamplingConfig,
        FileSplitConfig,
        FileWriterConfig,
    )

_ResolvedModelT = TypeVar("_ResolvedModelT", bound=BaseModel)


class ConfigResolver:
    """Centralized precedence and normalization for runtime configuration."""

    def resolve(
        self,
        *,
        descriptor: DatasetDescriptor,
        file_config: ConfigFile | None = None,
        cli_overrides: PlanOverrides | None = None,
    ) -> ResolvedConfig:
        """Resolve dataset defaults, file config, and CLI overrides into runtime config."""
        config = ResolvedConfig(
            loader=descriptor.default_loader_config,
            loader_options=descriptor.default_loader_options,
            map=descriptor.default_map_config,
        )

        if file_config is not None:
            for authoring_config in (
                file_config.global_,
                file_config.dataset_config(descriptor.name),
            ):
                config = self._apply_config(config, authoring_config, dataset_name=descriptor.name)

        resolved = self._apply_plan_overrides(config, cli_overrides or PlanOverrides(), descriptor)
        _validate_highway_config(descriptor=descriptor, config=resolved)
        return resolved

    def resolve_from_defaults(
        self, *, default: ResolvedConfig, overrides: FileDatasetConfig | None = None
    ) -> ResolvedConfig:
        """Resolve one authoring config block against an existing resolved config."""
        if overrides is None:
            return default
        return self._apply_config(default, overrides)

    def _apply_config(
        self,
        config: ResolvedConfig,
        authoring: FileDatasetConfig,
        *,
        dataset_name: str | None = None,
    ) -> ResolvedConfig:
        updates: dict[str, object] = {}

        if authoring.loader is not None:
            updates["loader"] = self._apply_loader_config(config.loader, authoring.loader)
            updates["loader_options"] = self._apply_loader_options(
                config.loader_options,
                authoring.loader.options,
                dataset_name=dataset_name,
            )

        if authoring.map is not None:
            updates["map"] = self._apply_map_config(config.map, authoring.map)

        if authoring.writer is not None:
            updates["writer"] = self._apply_writer_config(config.writer, authoring.writer)

        if authoring.execution is not None:
            updates["execution"] = self._apply_execution_config(
                config.execution,
                authoring.execution,
            )

        if authoring.split is not None:
            updates["split"] = self._apply_split_config(config.split, authoring.split)

        return config if not updates else config.model_copy(update=updates)

    @staticmethod
    def _apply_plan_overrides(
        config: ResolvedConfig, overrides: PlanOverrides, descriptor: DatasetDescriptor
    ) -> ResolvedConfig:
        updates: dict[str, object] = {}

        if overrides.scene_schema is not None:
            writer_data = config.writer.model_dump(round_trip=True)
            writer_data["scene_schema"] = get_scene_schema(overrides.scene_schema)
            updates["writer"] = RuntimeWriterConfig.model_validate(writer_data)

        if overrides.jobs is not None:
            updates["execution"] = _resolve_execution_jobs(config.execution, overrides.jobs)

        if overrides.include_map is False:
            updates["map"] = None
        elif overrides.include_map is True and config.map is None:
            updates["map"] = descriptor.default_map_config

        resolved = config if not updates else config.model_copy(update=updates)
        return resolved.model_copy(
            update={
                "split": _resolve_split_config(
                    base=resolved.split, overrides=overrides, descriptor=descriptor
                )
            }
        )

    @staticmethod
    def _apply_execution_config(
        base: ResolvedExecutionConfig, authoring: FileExecutionConfig
    ) -> ResolvedExecutionConfig:
        updates: dict[str, object] = {}

        if authoring.chunksize is not None:
            updates["chunksize"] = authoring.chunksize

        if authoring.jobs == "auto":
            updates["jobs"] = None
        elif authoring.jobs is not None:
            updates["jobs"] = authoring.jobs

        return _replace_model(base, **updates)

    @staticmethod
    def _apply_loader_config(
        base: RuntimeLoaderConfig, authoring: FileLoaderConfig
    ) -> RuntimeLoaderConfig:
        updates: dict[str, object] = {}

        for field_name in ("input_len", "output_len", "sample_time"):
            if (value := getattr(authoring, field_name)) is not None:
                updates[field_name] = value

        if authoring.highway is not None:
            updates["highway"] = _merge_optional_model(
                base.highway,
                HighwayParams.model_validate(authoring.highway),
                HighwayParams,
            )

        if authoring.window is not None:
            updates["window"] = _merge_optional_model(
                base.window,
                WindowConfig(size=authoring.window.size, step=authoring.window.step),
                WindowConfig,
            )

        if authoring.resampling is not None:
            updates["resampling"] = _apply_resampling_config(base.resampling, authoring.resampling)

        if authoring.filter is not None:
            updates["filter"] = _apply_filter_config(base.filter, authoring.filter)

        return _replace_model(base, **updates)

    @staticmethod
    def _apply_loader_options(
        base: LoaderOptions,
        raw_options: dict[str, object] | None,
        *,
        dataset_name: str | None,
    ) -> LoaderOptions:
        if raw_options is None:
            return base

        model_type = type(base)
        try:
            merged_options = base.model_dump(exclude_unset=False, round_trip=True)
            merged_options.update(raw_options)
            return model_type.model_validate(merged_options)
        except ValueError as exc:
            if model_type is NoLoaderOptions:
                msg = (
                    f"{dataset_name or 'This dataset'} does not expose dataset-specific loader "
                    "options. Remove [loader.options] from the config for this dataset."
                )
                raise ConfigurationError(msg) from exc

            msg = f"Invalid [loader.options] for {model_type.__name__}: {exc}"
            raise LoaderConfigError(msg) from exc

    @staticmethod
    def _apply_map_config(
        base: RuntimeMapConfig | None, authoring: FileMapConfig
    ) -> RuntimeMapConfig | None:
        if authoring.enabled is False:
            return None

        current = RuntimeMapConfig.default() if base is None else base
        updates: dict[str, object] = {}
        for field_name in ("min_distance", "interp_distance"):
            if (value := getattr(authoring, field_name)) is not None:
                updates[field_name] = value
        if authoring.extraction is not None:
            updates["extraction"] = _map_extraction(authoring)
        return _replace_model(current, **updates)

    @staticmethod
    def _apply_writer_config(
        base: RuntimeWriterConfig, authoring: FileWriterConfig
    ) -> RuntimeWriterConfig:
        updates: dict[str, object] = {}
        if authoring.scene_schema is not None:
            updates["scene_schema"] = get_scene_schema(authoring.scene_schema)
        if authoring.precision is not None:
            updates["precision"] = authoring.precision
        if authoring.offset_positions is not None:
            updates["offset_positions"] = authoring.offset_positions
        if authoring.mds is not None:
            updates["mds"] = _merge_model(
                base.mds,
                MDSFormatConfig.model_validate(
                    authoring.mds.model_dump(exclude_unset=True, round_trip=True)
                ),
            )
        return _replace_model(base, **updates)

    @staticmethod
    def _apply_split_config(
        base: RuntimeSplitConfig, authoring: FileSplitConfig
    ) -> RuntimeSplitConfig:
        updates: dict[str, object] = {}
        mode = _split_mode(authoring)
        if mode is not None:
            updates["mode"] = mode
        if authoring.ratio is not None:
            updates["ratio"] = authoring.ratio
        return _replace_model(base, **updates)


def _apply_resampling_config(
    base: ResampleSpec | None, authoring: FileResamplingConfig
) -> ResampleSpec:
    updates: dict[str, object] = {}
    for field_name in (
        "up",
        "down",
        "method",
        "position_columns",
        "max_gap",
        "sort",
        "sample_time",
    ):
        if (value := getattr(authoring, field_name)) is not None:
            updates[field_name] = value
    if (input_derivatives := authoring.input_derivative_map()) is not None:
        updates["input_derivatives"] = input_derivatives
    if (output_derivatives := authoring.output_derivative_map()) is not None:
        updates["output_derivatives"] = output_derivatives
    if base is None:
        return ResampleSpec.model_validate(updates)
    return _replace_model(base, **updates)


def _apply_filter_config(base_filter: Filter | None, authoring: FileLoaderFilterConfig) -> Filter:
    current = base_filter or Filter()
    rules = authoring.rules().resolve()
    if authoring.mode == "replace":
        current = rules
    elif authoring.mode == "extend":
        current = merge_filters(current, rules)
    return remove_filter_rules(current, authoring.remove)


def _map_extraction(authoring: FileMapConfig) -> object:
    match authoring.extraction:
        case "full":
            return FullMapExtraction()
        case "relevant":
            return RelevantAreaExtraction(padding=authoring.padding or 1.2)
        case "circle":
            return CircularExtraction(radius=authoring.radius or 0.0)
        case "bounding_box":
            return BoundingBoxExtraction(
                width=authoring.width or 0.0, height=authoring.height or 0.0
            )
        case _:
            msg = "map extraction mode must be set before resolving extraction config."
            raise ConfigurationError(msg)


def _split_mode(authoring: FileSplitConfig) -> SplitMode | None:
    if authoring.mode is None:
        return None
    if authoring.mode == "native":
        requested_splits = tuple(authoring.read or ())
        return NativeSplit(read=requested_splits)
    mode_map: dict[str, SplitMode | None] = {
        "auto": None,
        "time": TimeBlockSplit(gap=authoring.gap or 0),
        "shuffled-time": ShuffledTimeBlockSplit(
            segments=authoring.segments or 1,
            gap=authoring.gap or 0,
        ),
        "scene": BySceneSplit(),
        "source": BySourceSplit(),
        "none": Unsplit(),
    }
    return mode_map.get(authoring.mode, Unsplit())


def _resolve_execution_jobs(base: ResolvedExecutionConfig, jobs: int) -> ResolvedExecutionConfig:
    if jobs == -1:
        return base.model_copy(update={"jobs": None})
    if jobs < 1:
        msg = "jobs must be at least 1."
        raise ConfigurationError(msg)
    return base.model_copy(update={"jobs": jobs})


def _resolve_split_config(
    *, base: RuntimeSplitConfig, overrides: PlanOverrides, descriptor: DatasetDescriptor
) -> RuntimeSplitConfig:
    requested_splits = _resolve_requested_splits(overrides.read_split)
    requested_mode = overrides.split
    effective_ratio = (
        SplitWeights.from_tuple(overrides.ratio) if overrides.ratio is not None else base.ratio
    )
    effective_read = _resolve_mode_read(
        base=base,
        requested_mode=requested_mode,
        read=requested_splits,
    )
    effective_mode_name = base.mode_name if requested_mode is None else requested_mode
    effective_gap = _resolve_mode_gap(base=base, requested_mode=requested_mode, gap=overrides.gap)
    effective_segments = _resolve_mode_segments(
        base=base,
        requested_mode=requested_mode,
        segments=overrides.segments,
    )

    effective_mode_name = _resolve_effective_mode_name(
        overrides=overrides,
        descriptor=descriptor,
        effective_mode_name=effective_mode_name,
        effective_segments=effective_segments,
    )

    mode = _build_resolved_mode(
        mode_name=effective_mode_name,
        gap=effective_gap,
        segments=effective_segments,
        read=effective_read,
    )

    if isinstance(mode, Unsplit):
        return RuntimeSplitConfig(mode=mode)

    if isinstance(mode, NativeSplit):
        return _resolve_native_mode_config(mode=mode, descriptor=descriptor)

    return _resolve_custom_mode_config(
        mode=mode,
        ratio=effective_ratio,
        descriptor=descriptor,
        effective_mode_name=effective_mode_name,
    )


def _merge_model(base: _ResolvedModelT, overrides: BaseModel) -> _ResolvedModelT:
    return type(base).model_validate(
        base.model_copy(update=overrides.model_dump(exclude_unset=True, round_trip=True))
    )


def _replace_model(base: _ResolvedModelT, **updates: object) -> _ResolvedModelT:
    if not updates:
        return base
    return type(base).model_validate(base.model_copy(update=updates))


def _merge_optional_model(
    base: _ResolvedModelT | None, overrides: BaseModel, model_type: type[_ResolvedModelT]
) -> _ResolvedModelT:
    if base is None:
        return model_type.model_validate(overrides.model_dump(exclude_unset=True, round_trip=True))
    return _merge_model(base, overrides)


def _build_resolved_mode(
    *,
    mode_name: SplitModeName,
    gap: int | None,
    segments: int | None,
    read: Sequence[DatasetSplit],
) -> SplitMode:
    unique_splits = tuple(dict.fromkeys(read))
    mode_map: dict[str, SplitMode] = {
        "time": TimeBlockSplit(gap=gap or 0),
        "shuffled-time": ShuffledTimeBlockSplit(segments=segments or 1, gap=gap or 0),
        "scene": BySceneSplit(),
        "source": BySourceSplit(),
        "none": Unsplit(),
        "native": NativeSplit(read=unique_splits),
    }
    return mode_map[mode_name]


def _resolve_auto_mode(
    dataset_name: str,
    supported_modes: Sequence[SplitModeName],
    recommended_mode: SplitModeName | None,
) -> SplitModeName:
    if not supported_modes:
        msg = (
            f"{dataset_name} does not support custom split modes. "
            f"Example: {_split_example(dataset_name, 'native')}"
        )
        raise ConfigurationError(msg)
    if recommended_mode is not None:
        return recommended_mode
    if len(supported_modes) == 1:
        return supported_modes[0]
    msg = (
        f"Specify a split mode explicitly. "
        f"Example: {_split_example(dataset_name, supported_modes[0])}"
    )
    raise ConfigurationError(msg)


def _resolve_requested_splits(read_split: NativeSplitSelection) -> tuple[DatasetSplit, ...]:
    if read_split is None:
        return ()
    requested = [read_split] if isinstance(read_split, (DatasetSplit, str)) else list(read_split)
    return tuple(dict.fromkeys(DatasetSplit(value) for value in requested))


def _resolve_effective_mode_name(
    *,
    overrides: PlanOverrides,
    descriptor: DatasetDescriptor,
    effective_mode_name: SplitModeName,
    effective_segments: int | None,
) -> SplitModeName:
    _validate_split_overrides(
        mode_name=effective_mode_name,
        read_split_supplied=overrides.read_split is not None,
        ratio_supplied=overrides.ratio is not None,
        gap_supplied=overrides.gap is not None,
        segments_supplied=overrides.segments is not None,
        dataset_name=descriptor.name,
    )

    if effective_mode_name == "auto":
        effective_mode_name = _resolve_auto_mode(
            descriptor.name,
            descriptor.supported_split_strategies,
            descriptor.recommended_split_strategy,
        )

    if effective_mode_name == "shuffled-time" and effective_segments is None:
        msg = (
            "Split mode 'shuffled-time' requires --segments. "
            f"Example: {_split_example(descriptor.name, 'shuffled-time')}"
        )
        raise ConfigurationError(msg)

    return effective_mode_name


def _resolve_native_mode_config(
    *, mode: NativeSplit, descriptor: DatasetDescriptor
) -> RuntimeSplitConfig:
    if not descriptor.predefined_splits:
        raise ConfigurationError(_native_split_error_message(descriptor.name))

    resolved_splits = mode.active_splits() or tuple(descriptor.predefined_splits)
    if not resolved_splits:
        raise ConfigurationError(_native_split_error_message(descriptor.name))

    unsupported = [split for split in resolved_splits if split not in descriptor.predefined_splits]
    if unsupported:
        raise SplitNotSupportedError(descriptor.name, unsupported)
    return RuntimeSplitConfig(mode=NativeSplit(read=resolved_splits))


def _resolve_custom_mode_config(
    *,
    mode: SplitMode,
    ratio: SplitWeights | None,
    descriptor: DatasetDescriptor,
    effective_mode_name: SplitModeName,
) -> RuntimeSplitConfig:
    if mode.type not in descriptor.supported_split_strategies:
        raise SplitStrategyNotSupportedError(
            descriptor.name, mode.type, tuple(descriptor.supported_split_strategies)
        )

    if ratio is None:
        msg = (
            f"Split mode '{effective_mode_name}' requires --ratio TRAIN VAL TEST. "
            f"Example: {_split_example(descriptor.name, effective_mode_name)}"
        )
        raise ConfigurationError(msg)
    return RuntimeSplitConfig(mode=mode, ratio=ratio)


def _resolve_mode_gap(
    *, base: RuntimeSplitConfig, requested_mode: SplitModeName | None, gap: int | None
) -> int | None:
    if gap is not None:
        return gap
    if requested_mode is None and isinstance(base.mode, (TimeBlockSplit, ShuffledTimeBlockSplit)):
        return base.mode.gap
    return None


def _resolve_mode_segments(
    *, base: RuntimeSplitConfig, requested_mode: SplitModeName | None, segments: int | None
) -> int | None:
    if segments is not None:
        return segments
    if requested_mode is None and isinstance(base.mode, ShuffledTimeBlockSplit):
        return base.mode.segments
    return None


def _resolve_mode_read(
    *,
    base: RuntimeSplitConfig,
    requested_mode: SplitModeName | None,
    read: Sequence[DatasetSplit],
) -> tuple[DatasetSplit, ...]:
    if read:
        return tuple(dict.fromkeys(read))
    if requested_mode is None and isinstance(base.mode, NativeSplit):
        return base.mode.active_splits()
    return ()


def _validate_highway_config(*, descriptor: DatasetDescriptor, config: ResolvedConfig) -> None:
    if config.loader.highway is None:
        return

    from dronalize.datasets.registry import DatasetCapabilities  # noqa: PLC0415

    if descriptor.capabilities & DatasetCapabilities.HIGHWAY_PIPELINE:
        return

    msg = (
        f"{descriptor.name} does not support highway sampling. "
        "Remove [loader.highway] from the config for this dataset."
    )
    raise ConfigurationError(msg)


def _validate_split_overrides(
    *,
    mode_name: SplitModeName,
    read_split_supplied: bool,
    ratio_supplied: bool,
    gap_supplied: bool,
    segments_supplied: bool,
    dataset_name: str,
) -> None:
    if read_split_supplied and mode_name != "native":
        msg = (
            f"--read-split is only valid with --split native. "
            f"Example: {_split_example(dataset_name, 'native')}"
        )
        raise SplitConflictError(msg)
    if ratio_supplied and mode_name not in {"source", "scene", "time", "shuffled-time", "auto"}:
        msg = (
            f"--ratio is only valid with custom split modes. "
            f"Example: {_split_example(dataset_name, 'scene')}"
        )
        raise SplitConflictError(msg)
    if gap_supplied and mode_name not in {"time", "shuffled-time"}:
        msg = (
            f"--gap is only valid with --split time or --split shuffled-time. "
            f"Example: {_split_example(dataset_name, 'time')}"
        )
        raise SplitConflictError(msg)
    if segments_supplied and mode_name != "shuffled-time":
        msg = (
            f"--segments is only valid with --split shuffled-time. "
            f"Example: {_split_example(dataset_name, 'shuffled-time')}"
        )
        raise SplitConflictError(msg)


def _native_split_error_message(dataset_name: str) -> str:
    return (
        f"{dataset_name} does not expose native dataset splits. "
        f"Example: {_split_example(dataset_name, 'scene')}"
    )


def _split_example(dataset_name: str, mode_name: str) -> str:
    base = f"dronalize process {dataset_name} --input INPUT --output OUTPUT"
    if mode_name == "native":
        return f"{base} --split native --read-split train"
    if mode_name == "shuffled-time":
        return f"{base} --split shuffled-time --ratio 0.8 0.1 0.1 --segments 8 --gap 5"
    if mode_name == "time":
        return f"{base} --split time --ratio 0.8 0.1 0.1 --gap 5"
    return f"{base} --split {mode_name} --ratio 0.8 0.1 0.1"
