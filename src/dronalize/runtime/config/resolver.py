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
from dronalize.core.scene import get_trajectory_schema
from dronalize.io.config import ExportConfig as RuntimeExportConfig
from dronalize.io.config import MDSBackendConfig
from dronalize.processing.filtering import Filter, merge_filters
from dronalize.processing.filtering.filter import remove_filter_rules
from dronalize.processing.loading.base import LoaderOptions, NoLoaderOptions
from dronalize.processing.loading.config import LaneChangeSamplingConfig, WindowConfig
from dronalize.processing.loading.config import LoaderConfig as RuntimeLoaderConfig
from dronalize.processing.loading.splits import (
    NativeSplitStrategy,
    NativeSplitStrategySelection,
    NoSplitStrategy,
    SceneSplitStrategy,
    ShuffledTimeBlockStrategy,
    SourceSplitStrategy,
    SplitStrategy,
    SplitStrategyName,
    SplitWeights,
    TimeBlockStrategy,
)
from dronalize.processing.loading.splits import SplitConfig as RuntimeSplitConfig
from dronalize.processing.maps.config import (
    BoundingBoxExtraction,
    CircularExtraction,
    FullMapExtraction,
    SceneExtentExtraction,
)
from dronalize.processing.maps.config import MapConfig as RuntimeMapConfig
from dronalize.processing.pipeline.functional.resample import ResampleSpec
from dronalize.runtime.config.overrides import PlanOverrides
from dronalize.runtime.config.resolved import ResolvedConfig, ResolvedExecutionConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.datasets.registry import DatasetSpec
    from dronalize.runtime.config.file import (
        ConfigFile,
        FileDatasetConfig,
        FileExecutionConfig,
        FileExportConfig,
        FileLoaderConfig,
        FileLoaderFilterConfig,
        FileMapConfig,
        FileResamplingConfig,
        FileSplitConfig,
    )

_ResolvedModelT = TypeVar("_ResolvedModelT", bound=BaseModel)


class ConfigResolver:
    """Centralized precedence and normalization for runtime configuration."""

    def resolve(
        self,
        *,
        descriptor: DatasetSpec,
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

        if authoring.export is not None:
            updates["export"] = self._apply_writer_config(config.export, authoring.export)

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
        config: ResolvedConfig, overrides: PlanOverrides, descriptor: DatasetSpec
    ) -> ResolvedConfig:
        updates: dict[str, object] = {}

        if overrides.trajectory_schema is not None:
            export_data = config.export.model_dump(round_trip=True)
            export_data["trajectory_schema"] = get_trajectory_schema(overrides.trajectory_schema)
            updates["export"] = RuntimeExportConfig.model_validate(export_data)

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

        if authoring.lane_change_sampling is not None:
            updates["lane_change_sampling"] = _merge_optional_model(
                base.lane_change_sampling,
                LaneChangeSamplingConfig.model_validate(authoring.lane_change_sampling),
                LaneChangeSamplingConfig,
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
        base: RuntimeExportConfig, authoring: FileExportConfig
    ) -> RuntimeExportConfig:
        updates: dict[str, object] = {}
        if authoring.trajectory_schema is not None:
            updates["trajectory_schema"] = get_trajectory_schema(authoring.trajectory_schema)
        if authoring.precision is not None:
            updates["precision"] = authoring.precision
        if authoring.recenter_positions is not None:
            updates["recenter_positions"] = authoring.recenter_positions
        if authoring.mds is not None:
            updates["mds"] = _merge_model(
                base.mds,
                MDSBackendConfig.model_validate(
                    authoring.mds.model_dump(exclude_unset=True, round_trip=True)
                ),
            )
        return _replace_model(base, **updates)

    @staticmethod
    def _apply_split_config(
        base: RuntimeSplitConfig, authoring: FileSplitConfig
    ) -> RuntimeSplitConfig:
        updates: dict[str, object] = {}
        strategy = _split_strategy(authoring)
        if strategy is not None:
            updates["strategy"] = strategy
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
        case "scene_extent":
            return SceneExtentExtraction(padding=authoring.padding or 1.2)
        case "circle":
            return CircularExtraction(radius=authoring.radius or 0.0)
        case "bounding_box":
            return BoundingBoxExtraction(
                width=authoring.width or 0.0, height=authoring.height or 0.0
            )
        case _:
            msg = "map extraction strategy must be set before resolving extraction config."
            raise ConfigurationError(msg)


def _split_strategy(authoring: FileSplitConfig) -> SplitStrategy | None:
    if authoring.strategy is None:
        return None
    if authoring.strategy == "native":
        requested_splits = tuple(authoring.read or ())
        return NativeSplitStrategy(read=requested_splits)
    strategy_map: dict[str, SplitStrategy | None] = {
        "auto": None,
        "time": TimeBlockStrategy(gap=authoring.gap or 0),
        "shuffled-time": ShuffledTimeBlockStrategy(
            segments=authoring.segments or 1,
            gap=authoring.gap or 0,
        ),
        "scene": SceneSplitStrategy(),
        "source": SourceSplitStrategy(),
        "none": NoSplitStrategy(),
    }
    return strategy_map.get(authoring.strategy, NoSplitStrategy())


def _resolve_execution_jobs(base: ResolvedExecutionConfig, jobs: int) -> ResolvedExecutionConfig:
    if jobs == -1:
        return base.model_copy(update={"jobs": None})
    if jobs < 1:
        msg = "jobs must be at least 1."
        raise ConfigurationError(msg)
    return base.model_copy(update={"jobs": jobs})


def _resolve_split_config(
    *, base: RuntimeSplitConfig, overrides: PlanOverrides, descriptor: DatasetSpec
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
    effective_strategy_name = base.strategy_name if requested_mode is None else requested_mode
    effective_gap = _resolve_mode_gap(base=base, requested_mode=requested_mode, gap=overrides.gap)
    effective_segments = _resolve_mode_segments(
        base=base,
        requested_mode=requested_mode,
        segments=overrides.segments,
    )

    effective_strategy_name = _resolve_effective_strategy_name(
        overrides=overrides,
        descriptor=descriptor,
        effective_strategy_name=effective_strategy_name,
        effective_segments=effective_segments,
    )

    mode = _build_resolved_mode(
        strategy_name=effective_strategy_name,
        gap=effective_gap,
        segments=effective_segments,
        read=effective_read,
    )

    if isinstance(mode, NoSplitStrategy):
        return RuntimeSplitConfig(strategy=mode)

    if isinstance(mode, NativeSplitStrategy):
        return _resolve_native_mode_config(mode=mode, descriptor=descriptor)

    return _resolve_custom_mode_config(
        mode=mode,
        ratio=effective_ratio,
        descriptor=descriptor,
        effective_strategy_name=effective_strategy_name,
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
    strategy_name: SplitStrategyName,
    gap: int | None,
    segments: int | None,
    read: Sequence[DatasetSplit],
) -> SplitStrategy:
    unique_splits = tuple(dict.fromkeys(read))
    mode_map: dict[str, SplitStrategy] = {
        "time": TimeBlockStrategy(gap=gap or 0),
        "shuffled-time": ShuffledTimeBlockStrategy(segments=segments or 1, gap=gap or 0),
        "scene": SceneSplitStrategy(),
        "source": SourceSplitStrategy(),
        "none": NoSplitStrategy(),
        "native": NativeSplitStrategy(read=unique_splits),
    }
    return mode_map[strategy_name]


def _resolve_auto_mode(
    dataset_name: str,
    supported_modes: Sequence[SplitStrategyName],
    recommended_mode: SplitStrategyName | None,
) -> SplitStrategyName:
    if not supported_modes:
        msg = (
            f"{dataset_name} does not support custom split strategies. "
            f"Example: {_split_example(dataset_name, 'native')}"
        )
        raise ConfigurationError(msg)
    if recommended_mode is not None:
        return recommended_mode
    if len(supported_modes) == 1:
        return supported_modes[0]
    msg = (
        f"Specify a split strategy explicitly. "
        f"Example: {_split_example(dataset_name, supported_modes[0])}"
    )
    raise ConfigurationError(msg)


def _resolve_requested_splits(read_split: NativeSplitStrategySelection) -> tuple[DatasetSplit, ...]:
    if read_split is None:
        return ()
    requested = [read_split] if isinstance(read_split, (DatasetSplit, str)) else list(read_split)
    return tuple(dict.fromkeys(DatasetSplit(value) for value in requested))


def _resolve_effective_strategy_name(
    *,
    overrides: PlanOverrides,
    descriptor: DatasetSpec,
    effective_strategy_name: SplitStrategyName,
    effective_segments: int | None,
) -> SplitStrategyName:
    _validate_split_overrides(
        strategy_name=effective_strategy_name,
        read_split_supplied=overrides.read_split is not None,
        ratio_supplied=overrides.ratio is not None,
        gap_supplied=overrides.gap is not None,
        segments_supplied=overrides.segments is not None,
        dataset_name=descriptor.name,
    )

    if effective_strategy_name == "auto":
        effective_strategy_name = _resolve_auto_mode(
            descriptor.name,
            descriptor.supported_split_strategies,
            descriptor.recommended_split_strategy,
        )

    if effective_strategy_name == "shuffled-time" and effective_segments is None:
        msg = (
            "Split strategy 'shuffled-time' requires --segments. "
            f"Example: {_split_example(descriptor.name, 'shuffled-time')}"
        )
        raise ConfigurationError(msg)

    return effective_strategy_name


def _resolve_native_mode_config(
    *, mode: NativeSplitStrategy, descriptor: DatasetSpec
) -> RuntimeSplitConfig:
    if not descriptor.predefined_splits:
        raise ConfigurationError(_native_split_error_message(descriptor.name))

    resolved_splits = mode.active_splits() or tuple(descriptor.predefined_splits)
    if not resolved_splits:
        raise ConfigurationError(_native_split_error_message(descriptor.name))

    unsupported = [split for split in resolved_splits if split not in descriptor.predefined_splits]
    if unsupported:
        raise SplitNotSupportedError(descriptor.name, unsupported)
    return RuntimeSplitConfig(strategy=NativeSplitStrategy(read=resolved_splits))


def _resolve_custom_mode_config(
    *,
    mode: SplitStrategy,
    ratio: SplitWeights | None,
    descriptor: DatasetSpec,
    effective_strategy_name: SplitStrategyName,
) -> RuntimeSplitConfig:
    if mode.type not in descriptor.supported_split_strategies:
        raise SplitStrategyNotSupportedError(
            descriptor.name, mode.type, tuple(descriptor.supported_split_strategies)
        )

    if ratio is None:
        msg = (
            f"Split strategy '{effective_strategy_name}' requires --ratio TRAIN VAL TEST. "
            f"Example: {_split_example(descriptor.name, effective_strategy_name)}"
        )
        raise ConfigurationError(msg)
    return RuntimeSplitConfig(strategy=mode, ratio=ratio)


def _resolve_mode_gap(
    *, base: RuntimeSplitConfig, requested_mode: SplitStrategyName | None, gap: int | None
) -> int | None:
    if gap is not None:
        return gap
    if requested_mode is None and isinstance(
        base.strategy, (TimeBlockStrategy, ShuffledTimeBlockStrategy)
    ):
        return base.strategy.gap
    return None


def _resolve_mode_segments(
    *, base: RuntimeSplitConfig, requested_mode: SplitStrategyName | None, segments: int | None
) -> int | None:
    if segments is not None:
        return segments
    if requested_mode is None and isinstance(base.strategy, ShuffledTimeBlockStrategy):
        return base.strategy.segments
    return None


def _resolve_mode_read(
    *,
    base: RuntimeSplitConfig,
    requested_mode: SplitStrategyName | None,
    read: Sequence[DatasetSplit],
) -> tuple[DatasetSplit, ...]:
    if read:
        return tuple(dict.fromkeys(read))
    if requested_mode is None and isinstance(base.strategy, NativeSplitStrategy):
        return base.strategy.active_splits()
    return ()


def _validate_highway_config(*, descriptor: DatasetSpec, config: ResolvedConfig) -> None:
    if config.loader.lane_change_sampling is None:
        return

    from dronalize.datasets.registry import DatasetCapabilities  # noqa: PLC0415

    if descriptor.capabilities & DatasetCapabilities.LANE_CHANGE_SAMPLING:
        return

    msg = (
        f"{descriptor.name} does not support lane-change sampling. "
        "Remove [loader.lane_change_sampling] from the config for this dataset."
    )
    raise ConfigurationError(msg)


def _validate_split_overrides(
    *,
    strategy_name: SplitStrategyName,
    read_split_supplied: bool,
    ratio_supplied: bool,
    gap_supplied: bool,
    segments_supplied: bool,
    dataset_name: str,
) -> None:
    if read_split_supplied and strategy_name != "native":
        msg = (
            f"--read-split is only valid with --split native. "
            f"Example: {_split_example(dataset_name, 'native')}"
        )
        raise SplitConflictError(msg)
    if ratio_supplied and strategy_name not in {"source", "scene", "time", "shuffled-time", "auto"}:
        msg = (
            f"--ratio is only valid with custom split strategies. "
            f"Example: {_split_example(dataset_name, 'scene')}"
        )
        raise SplitConflictError(msg)
    if gap_supplied and strategy_name not in {"time", "shuffled-time"}:
        msg = (
            f"--gap is only valid with --split time or --split shuffled-time. "
            f"Example: {_split_example(dataset_name, 'time')}"
        )
        raise SplitConflictError(msg)
    if segments_supplied and strategy_name != "shuffled-time":
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


def _split_example(dataset_name: str, strategy_name: str) -> str:
    base = f"dronalize process {dataset_name} --input INPUT --output OUTPUT"
    if strategy_name == "native":
        return f"{base} --split native --read-split train"
    if strategy_name == "shuffled-time":
        return f"{base} --split shuffled-time --ratio 0.8 0.1 0.1 --segments 8 --gap 5"
    if strategy_name == "time":
        return f"{base} --split time --ratio 0.8 0.1 0.1 --gap 5"
    return f"{base} --split {strategy_name} --ratio 0.8 0.1 0.1"
