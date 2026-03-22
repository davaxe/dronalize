# ruff: noqa: PLC0415
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import dronalize.exceptions as dronalize_exceptions
import dronalize.execution.common as ex_common
from dronalize.config.config import Config, load_config_overrides, resolve_runtime_config
from dronalize.datasets import DatasetDescriptor, get
from dronalize.execution.assigner import ConstantAssigner, StatelessWeightedAssigner
from dronalize.execution.parallel.executor import ParallelExecutor
from dronalize.execution.sequential import SequentialExecutor
from dronalize.storage.formats import OutputFormat

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path

    from dronalize.categories import DatasetSplit
    from dronalize.config.loader import LoaderConfig
    from dronalize.config.writer import SceneSchemaLike, WriterConfig
    from dronalize.execution.assigner import SplitAssigner
    from dronalize.execution.executor import ObservableWritingExecutor, WriterFactory
    from dronalize.loading import BaseSceneLoader
    from dronalize.scene import SceneSchema


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
    split_plan: ex_common.SplitPlan
    limit: int | None
    seed: int | None

    @property
    def parallel(self) -> bool:
        """Return whether this planned run should execute in parallel."""
        return self.config.execution.parallel

    def loader_splits(self) -> list[DatasetSplit] | None:
        """Resolve which predefined splits, if any, the loader should read."""
        return self.split_plan.loader_splits(self.descriptor.predefined_splits)

    def writer_splits(self) -> list[DatasetSplit] | None:
        """Resolve which split directories, if any, the writer should create."""
        return self.split_plan.writer_splits()

    def split_assigner(self) -> SplitAssigner | None:
        """Build a split assigner for modes that derive splits during writing."""
        groups = self.writer_splits()
        if groups is None or len(groups) == 0:
            return None

        if self.split_plan.custom_weights is not None:
            return StatelessWeightedAssigner(
                groups=groups,
                weights=self.split_plan.weights(),
                seed=self.seed,
            )

        if len(self.descriptor.predefined_splits) == 0 and len(groups) == 1:
            return ConstantAssigner(groups[0])

        return None

    def build_loader(self) -> BaseSceneLoader[object]:
        """Instantiate the dataset loader for this job."""
        return self.descriptor.build_loader(
            self.data_root,
            loader_config=self.config.loader,
            map_config=self.config.map,
            splits=self.loader_splits(),
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
        splits = self.descriptor.predefined_splits

        raw_rows = [
            ("Dataset", self.descriptor.name),
            ("Input directory", str(self.data_root)),
            ("Output directory", str(self.output_dir)),
            ("Output format", self.output_format.value),
            ("Scene schema", f"{writer.scene_schema.name} ({writer.feature_dim} features)"),
            (
                "Horizon (in/out)",
                f"{loader.resampled_input_len} / {loader.resampled_output_len} frames",
            ),
            ("Sample rate", f"{1 / loader.post_sample_time:.1f} Hz"),
            ("Execution", self._execution_summary()),
            ("Available splits", self._format_splits(splits)) if splits else None,
            ("Read splits", self._loader_split_summary()),
            ("Write outputs", self._writer_split_summary()),
            ("Source limit", str(self.limit)) if self.limit is not None else None,
            ("Random seed", str(self.seed)) if self.seed is not None else None,
        ]

        return ProcessingSummary(
            title="Processing Plan",
            rows=tuple(row for row in raw_rows if row is not None),
        )

    def _execution_summary(self) -> str:
        if not self.parallel:
            return "sequential"
        workers = self.config.execution.workers
        return f"parallel ({workers} worker{'s' if workers != 1 else ''})"

    def _loader_split_summary(self) -> str:
        return self._format_splits(s) if (s := self.loader_splits()) else "all available data"

    def _writer_split_summary(self) -> str:
        if self.split_plan.custom_weights is not None:
            return self._format_weighted_splits(self.split_plan.active_custom_groups())
        return self._format_splits(s) if (s := self.writer_splits()) else "single output directory"

    @staticmethod
    def _format_splits(splits: list[DatasetSplit]) -> str:
        return ", ".join(split.value for split in splits)

    @staticmethod
    def _format_weighted_splits(groups: list[tuple[DatasetSplit, float]]) -> str:
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
    _split_assigner: SplitAssigner | None = field(default=None, init=False, repr=False)

    def summary(self) -> ProcessingSummary:
        """Return the display-ready summary for the backing job."""
        return self.job.summary()

    def run(self) -> None:
        """Execute the active run."""
        if self._writer_factory is None:
            self._writer_factory = self.job.build_writer_factory()
        if self._split_assigner is None:
            self._split_assigner = self.job.split_assigner()

        self.executor.execute(
            writer_factory=self._writer_factory,
            split_assigner=self._split_assigner,
        )


def prepare_dataset(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    split: ex_common.SplitType | Sequence[ex_common.SplitType] | None = None,
    output_format: str = "mds",
    scene_schema: SceneSchemaLike | None = None,
    config_path: Path | None = None,
    jobs: int | None = None,
    limit: int | None = None,
    custom_split: tuple[float, float, float] | None = None,
    seed: int | None = None,
) -> DatasetJob:
    """Resolve user-facing options into a reusable processing job."""
    if not input_dir.exists():
        msg = f"Input directory {input_dir} does not exist."
        raise FileNotFoundError(msg)

    split_plan = ex_common.resolve_split_plan(split, custom_split)
    descriptor = get(dataset)
    ex_common.validate_split_plan(
        split_plan,
        dataset_name=descriptor.name,
        predefined_splits=descriptor.predefined_splits,
    )

    config = _resolve_job_config(
        descriptor,
        config_path=config_path,
        scene_schema=scene_schema,
        jobs=jobs,
    )

    return DatasetJob(
        descriptor=descriptor,
        data_root=input_dir,
        output_dir=output_dir,
        output_format=_resolve_output_format(output_format),
        config=config,
        split_plan=split_plan,
        limit=limit,
        seed=seed,
    )


def _resolve_job_config(
    descriptor: DatasetDescriptor,
    *,
    config_path: Path | None,
    scene_schema: SceneSchemaLike | None,
    jobs: int | None,
) -> Config:
    config_overrides: dict[str, object] = {}
    if config_path is not None:
        config_overrides = load_config_overrides(config_path).for_dataset(descriptor.name)

    default_config = Config(loader=descriptor.default_config, map=descriptor.default_map_config)
    config = resolve_runtime_config(default=default_config, overrides=config_overrides)

    if scene_schema is not None:
        writer_config = type(config.writer).model_validate({
            **config.writer.model_dump(),
            "scene_schema": scene_schema,
        })
        config = config.model_copy(update={"writer": writer_config})

    if jobs is not None:
        parallel = jobs > 1
        if jobs == -1:
            jobs = None
            parallel = True
        elif jobs < 1:
            msg = "jobs must be at least 1."
            raise ValueError(msg)
        config = config.model_copy(
            update={
                "execution": config.execution.model_copy(
                    update={"parallel": parallel, "workers": jobs},
                ),
            },
        )

    return config


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
            from dronalize.storage.writers.mds import MDSSceneWriter

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
            from dronalize.storage.writers._dummy import DummyWriter

            return DummyWriter.as_factory(log=False)
