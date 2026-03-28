# ruff: noqa: PLC0415
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import dronalize.core.errors as dronalize_exceptions
from dronalize.datasets import DatasetDescriptor, get
from dronalize.io.formats import OutputFormat
from dronalize.processing.ingest.splits import NativeSplit, Unsplit
from dronalize.runtime.config import Config, load_config_overrides, resolve_runtime_config
from dronalize.runtime.execution.parallel.executor import ParallelExecutor
from dronalize.runtime.execution.sequential import SequentialExecutor

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence
    from pathlib import Path

    from dronalize.core.categories import DatasetSplit
    from dronalize.core.scene import SceneSchema
    from dronalize.io.config import SceneSchemaLike, WriterConfig
    from dronalize.processing.ingest.base import BaseSceneLoader
    from dronalize.processing.ingest.config import LoaderConfig
    from dronalize.processing.ingest.splits import SplitRequest, SplitStrategyName
    from dronalize.runtime.execution.executor import ObservableWritingExecutor, WriterFactory


@dataclass(slots=True, frozen=True)
class ProcessingSummary:
    """Display-ready summary of a prepared processing job."""

    title: str
    rows: tuple[tuple[str, str], ...]


@dataclass(slots=True, frozen=True)
class DatasetJob:
    """Resolved, side-effect-free plan for one dataset processing run."""

    descriptor: DatasetDescriptor
    data_root: Path
    output_dir: Path
    output_format: OutputFormat
    config: Config
    split_request: SplitRequest
    limit: int | None
    seed: int | None

    @property
    def parallel(self) -> bool:
        """Return whether this planned run should execute in parallel."""
        return self.config.execution.parallel

    def loader_splits(self) -> tuple[DatasetSplit, ...] | None:
        """Resolve which predefined splits, if any, the loader should read."""
        return self.split_request.loader_splits()

    def writer_splits(self) -> tuple[DatasetSplit, ...] | None:
        """Resolve which split directories, if any, the writer should create."""
        return self.split_request.writer_splits()

    def build_loader(self) -> BaseSceneLoader[object]:
        """Instantiate the dataset loader for this job."""
        return self.descriptor.build_loader(
            self.data_root,
            loader_config=self.config.loader,
            map_config=self.config.map,
            splits=self.loader_splits(),
            split_request=self.split_request,
            output_schema=self.config.writer.scene_schema,
        )

    def build_executor(self, loader: BaseSceneLoader[object]) -> ObservableWritingExecutor:
        """Construct the runtime executor for this job."""
        if self.parallel:
            return ParallelExecutor(
                loader,
                workers=self.config.execution.workers,
                chunksize=self.config.execution.chunksize,
                limit=self.limit,
            )

        return SequentialExecutor(loader, limit=self.limit)

    def build_writer_factory(self) -> WriterFactory:
        """Build the worker-local writer factory for this job."""
        writer_splits = self.writer_splits()
        splits = None if writer_splits is None else tuple(dict.fromkeys(writer_splits))
        return _get_writer(
            self.output_format,
            self.output_dir,
            loader_config=self.config.loader,
            writer_config=self.config.writer,
            source_scene_schema=self.descriptor.native_schema,
            splits=splits,
            parallel=self.parallel,
            has_map=self.config.map.include_map,
        )

    def summary(self) -> ProcessingSummary:
        """Return a display-ready summary of this prepared job."""
        loader, writer = self.config.loader, self.config.writer

        raw_rows = [
            ("Dataset", self.descriptor.name),
            ("Input directory", str(self.data_root)),
            ("Output directory", str(self.output_dir)),
            ("Output format", self.output_format.value),
            ("Scene schema", f"{writer.scene_schema.name} ({writer.feature_dim} features)"),
            ("Window @ Hz", self._window_summary(loader)),
            ("Execution", self._execution_summary()),
            *self._split_summary_rows(),
            ("Source limit", str(self.limit)) if self.limit is not None else None,
            (
                "Random seed",
                str(seed),
            )
            if (seed := self._effective_random_seed()) is not None
            else None,
        ]

        return ProcessingSummary(
            title="Processing Plan",
            rows=tuple(row for row in raw_rows if row is not None),
        )

    def _execution_summary(self) -> str:
        if not self.parallel:
            return "sequential"
        workers = self.config.execution.workers
        if workers is None:
            return "parallel (auto workers)"
        return f"parallel ({workers} worker{'s' if workers != 1 else ''})"

    @staticmethod
    def _window_summary(loader: LoaderConfig) -> str:
        return (
            f"{loader.resampled_input_len}/{loader.resampled_output_len}"
            f" @ {1 / loader.post_sample_time:.1f} Hz"
        )

    def _split_summary_rows(self) -> tuple[tuple[str, str], ...]:
        if isinstance(self.split_request.strategy, NativeSplit):
            splits = self.loader_splits()
            return (("Native splits", self._format_splits(splits)),) if splits else ()
        if not isinstance(self.split_request.strategy, Unsplit):
            return (
                ("Split assignment", self.split_request.strategy_name.replace("_", " ")),
                (
                    "Output splits",
                    self._format_weighted_splits(self.split_request.active()),
                ),
            )
        return ()

    def _effective_random_seed(self) -> int | None:
        return self.split_request.seed

    @staticmethod
    def _format_splits(splits: Iterable[DatasetSplit]) -> str:
        return ", ".join(split.value for split in splits)

    @staticmethod
    def _format_weighted_splits(groups: Iterable[tuple[DatasetSplit, float]]) -> str:
        total_weight = sum(weight for _, weight in groups)
        if total_weight <= 0:
            return "single output directory"

        formatted = [f"{split.value} ({weight / total_weight:.0%})" for split, weight in groups]
        return ", ".join(formatted)

    @contextmanager
    def open(self) -> Generator[DatasetRun]:
        """Open a live execution context for this job."""
        with self.descriptor.execution_scope(self.data_root, self.config.loader, self.config.map):
            loader = self.build_loader()
            executor = self.build_executor(loader)
            yield DatasetRun(job=self, executor=executor)

    def run(self) -> None:
        """Open this job and execute it immediately."""
        with self.open() as run:
            run.run()


@dataclass(slots=True)
class DatasetRun:
    """Live execution context with resources opened and progress observable."""

    job: DatasetJob
    executor: ObservableWritingExecutor
    _writer_factory: WriterFactory | None = field(default=None, init=False, repr=False)

    def summary(self) -> ProcessingSummary:
        """Return the display-ready summary for the backing job."""
        return self.job.summary()

    def run(self) -> None:
        """Execute the active run."""
        if self._writer_factory is None:
            self._writer_factory = self.job.build_writer_factory()

        self.executor.execute(writer_factory=self._writer_factory)


def prepare_dataset(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    output_format: str = "mds",
    scene_schema: SceneSchemaLike | None = None,
    config_path: Path | None = None,
    jobs: int | None = None,
    limit: int | None = None,
    split: Sequence[DatasetSplit | str] | DatasetSplit | str | None = None,
    split_strategy: SplitStrategyName | None = None,
    split_weights: tuple[float, float, float] | None = None,
    split_gap: int | None = None,
    split_n_segments: int | None = None,
    seed: int | None = None,
) -> DatasetJob:
    """Resolve user-facing options into a reusable processing job."""
    if not input_dir.exists():
        msg = f"Input directory {input_dir} does not exist."
        raise FileNotFoundError(msg)
    descriptor = get(dataset)

    config = _resolve_job_config(
        descriptor,
        config_path=config_path,
        scene_schema=scene_schema,
        jobs=jobs,
        split=split,
        split_strategy=split_strategy,
        split_weights=split_weights,
        split_gap=split_gap,
        split_n_segments=split_n_segments,
    )

    return DatasetJob(
        descriptor=descriptor,
        data_root=input_dir,
        output_dir=output_dir,
        output_format=_resolve_output_format(output_format),
        config=config,
        split_request=config.split.request(seed),
        limit=limit,
        seed=seed,
    )


def _resolve_job_config(
    descriptor: DatasetDescriptor,
    *,
    config_path: Path | None,
    scene_schema: SceneSchemaLike | None,
    jobs: int | None,
    split: Sequence[DatasetSplit | str] | DatasetSplit | str | None,
    split_strategy: SplitStrategyName | None,
    split_weights: tuple[float, float, float] | None,
    split_gap: int | None,
    split_n_segments: int | None,
) -> Config:
    config = resolve_runtime_config(
        default=Config(loader=descriptor.default_config, map=descriptor.default_map_config),
        overrides=_load_dataset_overrides(config_path, dataset_name=descriptor.name),
    )
    config.split = config.split.resolve_runtime_overrides(
        split=split,
        split_strategy_name=split_strategy,
        split_weights=split_weights,
        split_gap=split_gap,
        split_n_segments=split_n_segments,
        dataset_name=descriptor.name,
        predefined_splits=descriptor.predefined_splits,
        supported_split_strategies=descriptor.supported_split_strategies,
        recommended_split_strategy=descriptor.recommended_split_strategy,
    )
    return config.with_scene_schema(scene_schema).with_jobs(jobs)


def _load_dataset_overrides(
    config_path: Path | None,
    *,
    dataset_name: str,
) -> dict[str, object]:
    if config_path is None:
        return {}
    return load_config_overrides(config_path).for_dataset(dataset_name)


def _resolve_output_format(output_format: str) -> OutputFormat:
    try:
        return OutputFormat(output_format)
    except ValueError as exc:
        msg = f"Unsupported output format: {output_format}"
        raise dronalize_exceptions.ConfigurationError(msg) from exc


def _get_writer(
    output_format: OutputFormat,
    output_dir: Path,
    *,
    loader_config: LoaderConfig,
    writer_config: WriterConfig,
    source_scene_schema: SceneSchema,
    splits: tuple[DatasetSplit, ...] | None,
    parallel: bool,
    has_map: bool,
) -> WriterFactory:
    """Resolve an output writer factory from a normalized format identifier."""
    match output_format:
        case OutputFormat.MDS:
            from dronalize.io.writers.mds import MDSSceneWriter

            return MDSSceneWriter.as_factory(
                output_dir,
                config=writer_config,
                loader_config=loader_config,
                source_scene_schema=source_scene_schema,
                splits=splits,
                parallel=parallel,
                has_map=has_map,
            )
        case OutputFormat.DUMMY:
            from dronalize.io.writers.dummy import DummyWriter

            return DummyWriter.as_factory(log=True)
