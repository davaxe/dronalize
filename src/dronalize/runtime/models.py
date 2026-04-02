"""Runtime plan and execution models for dataset processing runs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING

from dronalize.io.formats import OutputFormat
from dronalize.runtime.parallel.executor import ParallelExecutor
from dronalize.runtime.sequential import SequentialExecutor

if TYPE_CHECKING:
    from collections.abc import Generator
    from contextlib import AbstractContextManager
    from pathlib import Path

    from dronalize.core.categories import DatasetSplit
    from dronalize.core.scene import SceneSchema
    from dronalize.datasets.registry import DatasetDescriptor
    from dronalize.io.config import WriterConfig
    from dronalize.processing.ingest.base import BaseSceneLoader
    from dronalize.processing.ingest.config import LoaderConfig
    from dronalize.processing.ingest.splits import SplitConfig
    from dronalize.runtime.config import ResolvedConfig
    from dronalize.runtime.executor import ObservableWritingExecutor, WriterFactory


@dataclass(slots=True, frozen=True)
class SummarySection:
    """One titled section in a rendered processing summary."""

    title: str
    rows: tuple[tuple[str, str], ...]


@dataclass(slots=True, frozen=True)
class ProcessingSummary:
    """Display-ready summary of a prepared processing plan."""

    title: str
    sections: tuple[SummarySection, ...]

    @property
    def rows(self) -> tuple[tuple[str, str], ...]:
        """Return all summary rows flattened across sections."""
        return tuple(row for section in self.sections for row in section.rows)


@dataclass(slots=True, frozen=True)
class DatasetPlan:
    """Resolved, side-effect-free plan for one dataset processing run."""

    descriptor: DatasetDescriptor
    """Underlying dataset descriptor that this plan is based on."""
    data_root: Path
    """Root directory where the dataset files are located."""
    output_dir: Path
    """Directory where processed dataset files should be written."""
    output_format: OutputFormat
    """Output format to write the processed dataset in."""
    config: ResolvedConfig
    """Fully resolved configuration for this plan, including defaults and overrides."""
    split_request: SplitConfig
    """Split configuration for this plan."""
    limit: int | None
    """Optional limit on the number of sources to process."""
    seed: int | None
    """Optional random seed for reproducible processing runs."""

    @property
    def parallel(self) -> bool:
        """Return whether this plan should execute in parallel."""
        return self.config.execution.parallel

    def loader_splits(self) -> tuple[DatasetSplit, ...] | None:
        """Return the predefined dataset splits that the loader should read."""
        return self.split_request.loader_splits()

    def writer_splits(self) -> tuple[DatasetSplit, ...] | None:
        """Return the output split directories that the writer should create."""
        return self.split_request.writer_splits()

    def summary(self) -> ProcessingSummary:
        """Build a display-ready summary for this plan."""
        summarize_plan = import_module("dronalize.runtime.summary").summarize_plan
        return summarize_plan(self)

    def open(self) -> AbstractContextManager[DatasetRun]:
        """Open a live execution context for this plan."""
        return open_plan(self)

    def run(self) -> None:
        """Open this plan and execute it immediately."""
        run_plan(self)


@dataclass(slots=True)
class DatasetRun:
    """Live execution context with open resources and an attached executor."""

    plan: DatasetPlan
    executor: ObservableWritingExecutor
    _writer_factory: WriterFactory | None = field(default=None, init=False, repr=False)

    def ensure_writer_factory(self) -> WriterFactory:
        """Return the cached writer factory, creating it on first use."""
        if self._writer_factory is None:
            self._writer_factory = _build_writer_factory(self.plan)
        return self._writer_factory

    def summary(self) -> ProcessingSummary:
        """Return the display-ready summary for the backing plan."""
        return self.plan.summary()

    def run(self) -> None:
        """Execute the active run."""
        execute_run(self)


@contextmanager
def open_plan(plan: DatasetPlan) -> Generator[DatasetRun, None, None]:
    """Open a live execution context for ``plan``.

    Parameters
    ----------
    plan : DatasetPlan
        Fully resolved processing plan to open.

    Yields
    ------
    DatasetRun
        Live run object with dataset-specific resources opened and an executor
        ready to process data.
    """
    with plan.descriptor.execution_scope(plan.data_root, plan.config.loader, plan.config.map):
        loader = _build_loader(plan)
        executor = _build_executor(plan, loader)
        yield DatasetRun(plan=plan, executor=executor)


def execute_run(run: DatasetRun) -> None:
    """Execute a previously opened run."""
    run.executor.execute(writer_factory=run.ensure_writer_factory())


def run_plan(plan: DatasetPlan) -> None:
    """Open ``plan`` and execute it immediately."""
    with open_plan(plan) as run:
        run.run()


def _resolve_writer_factory(
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
    """Resolve a writer factory from a normalized output-format identifier."""
    match output_format:
        case OutputFormat.MDS:
            from dronalize.io.writers.mds import MDSSceneWriter  # noqa: PLC0415

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
            from dronalize.io.writers.dummy import DummyWriter  # noqa: PLC0415

            return DummyWriter.as_factory(log=True)


def _build_loader(plan: DatasetPlan) -> BaseSceneLoader[object]:
    """Instantiate the dataset loader described by ``plan``."""
    return plan.descriptor.build_loader(
        plan.data_root,
        loader_config=plan.config.loader,
        loader_options=plan.config.loader_options,
        map_config=plan.config.map,
        splits=plan.loader_splits(),
        split_request=plan.split_request,
        output_schema=plan.config.writer.scene_schema,
    )


def _build_executor(
    plan: DatasetPlan, loader: BaseSceneLoader[object]
) -> ObservableWritingExecutor:
    """Construct the runtime executor for ``plan`` and ``loader``."""
    if plan.parallel:
        return ParallelExecutor(
            loader,
            workers=plan.config.execution.jobs,
            chunksize=plan.config.execution.chunksize,
            limit=plan.limit,
        )

    return SequentialExecutor(loader, limit=plan.limit)


def _build_writer_factory(plan: DatasetPlan) -> WriterFactory:
    """Build the worker-local writer factory for ``plan``."""
    writer_splits = plan.writer_splits()
    splits = None if writer_splits is None else tuple(dict.fromkeys(writer_splits))
    return _resolve_writer_factory(
        plan.output_format,
        plan.output_dir,
        loader_config=plan.config.loader,
        writer_config=plan.config.writer,
        source_scene_schema=plan.descriptor.native_schema,
        splits=splits,
        parallel=plan.parallel,
        has_map=plan.config.map is not None,
    )
