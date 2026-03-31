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
    """One titled section in a processing summary."""

    title: str
    rows: tuple[tuple[str, str], ...]


@dataclass(slots=True, frozen=True)
class ProcessingSummary:
    """Display-ready summary of a prepared processing plan."""

    title: str
    sections: tuple[SummarySection, ...]

    @property
    def rows(self) -> tuple[tuple[str, str], ...]:
        """Return the summary rows flattened across all sections."""
        return tuple(row for section in self.sections for row in section.rows)


@dataclass(slots=True, frozen=True)
class DatasetPlan:
    """Resolved, side-effect-free plan for one dataset processing run."""

    descriptor: DatasetDescriptor
    data_root: Path
    output_dir: Path
    output_format: OutputFormat
    config: ResolvedConfig
    split_request: SplitConfig
    limit: int | None
    seed: int | None

    @property
    def parallel(self) -> bool:
        """Return whether this plan should execute in parallel."""
        return self.config.execution.parallel

    def loader_splits(self) -> tuple[DatasetSplit, ...] | None:
        """Resolve which predefined splits, if any, the loader should read."""
        return self.split_request.loader_splits()

    def writer_splits(self) -> tuple[DatasetSplit, ...] | None:
        """Resolve which split directories, if any, the writer should create."""
        return self.split_request.writer_splits()

    def summary(self) -> ProcessingSummary:
        """Return a display-ready summary for this plan."""
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
    """Live execution context with resources opened and progress observable."""

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
    """Open a live execution context for a plan."""
    with plan.descriptor.execution_scope(plan.data_root, plan.config.loader, plan.config.map):
        loader = _build_loader(plan)
        executor = _build_executor(plan, loader)
        yield DatasetRun(plan=plan, executor=executor)


def execute_run(run: DatasetRun) -> None:
    """Execute an open run."""
    run.executor.execute(writer_factory=run.ensure_writer_factory())


def run_plan(plan: DatasetPlan) -> None:
    """Open a plan and execute it immediately."""
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
    """Resolve an output writer factory from a normalized format identifier."""
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
    """Instantiate the dataset loader for a plan."""
    return plan.descriptor.build_loader(
        plan.data_root,
        loader_config=plan.config.loader,
        map_config=plan.config.map,
        splits=plan.loader_splits(),
        split_request=plan.split_request,
        output_schema=plan.config.writer.scene_schema,
    )


def _build_executor(
    plan: DatasetPlan, loader: BaseSceneLoader[object]
) -> ObservableWritingExecutor:
    """Construct the runtime executor for a plan."""
    if plan.parallel:
        return ParallelExecutor(
            loader,
            workers=plan.config.execution.jobs,
            chunksize=plan.config.execution.chunksize,
            limit=plan.limit,
        )

    return SequentialExecutor(loader, limit=plan.limit)


def _build_writer_factory(plan: DatasetPlan) -> WriterFactory:
    """Build the worker-local writer factory for a plan."""
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
