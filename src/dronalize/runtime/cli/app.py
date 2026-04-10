"""Typer application for the optional dronalize CLI."""

# ruff: noqa: PLC0415
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeVar

import click
import typer
from pydantic import ValidationError
from rich import print as rprint

from dronalize.config.runtime import RuntimeOverride
from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import (
    ConfigurationError,
    DatasetNotFoundError,
    DronalizeError,
    MissingOptionalDependencyError,
    SplitError,
)
from dronalize.io.formats import StorageBackend
from dronalize.runtime.cli.formatting import (
    PLAN_NOTICE,
    build_available_datasets_table,
    build_dataset_inspect_tables,
    build_processing_summary_table,
    build_split_support_tables,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dronalize.runtime.plans import RunPlan

app: typer.Typer = typer.Typer(help="Trajectory data processing package.", no_args_is_help=True)
_T = TypeVar("_T")

SplitStrategy = Literal["none", "native", "scene", "source", "time", "shuffled-time"]

InputDir = Annotated[
    Path, typer.Option("--input", "-i", help="Directory containing the raw dataset.")
]
OutputDir = Annotated[
    Path, typer.Option("--output", "-o", help="Directory to save the processed dataset.")
]
Split = Annotated[
    SplitStrategy | None,
    typer.Option(
        "--split", "-s", help="Split mode to use.", show_default=False, rich_help_panel="Split"
    ),
]
ReadSplit = Annotated[
    list[DatasetSplit] | None,
    typer.Option(
        "--read-split",
        "-rs",
        help="Dataset-defined partition(s) to read when --split native is selected.",
        show_default=False,
        rich_help_panel="Split",
    ),
]
Config = Annotated[
    Path | None, typer.Option("--config", "-c", help="Path to the optional configuration file.")
]
Jobs = Annotated[
    int | None,
    typer.Option(
        "--jobs",
        "-j",
        help="Worker count override. Values greater than 1 enable parallel execution.",
    ),
]
Progress = Annotated[
    bool, typer.Option("--progress/--no-progress", help="Show progress during processing.")
]
Limit = Annotated[
    int | None, typer.Option("--limit", "-l", help="Limit the number of scenes to process.")
]
Seed = Annotated[
    int | None,
    typer.Option(
        "--seed", help="Random seed used by split assignment and other randomized operations."
    ),
]
StorageBackendOption = Annotated[
    StorageBackend, typer.Option("--storage-backend", help="Storage backend for processed data.")
]
TrajectorySchema = Annotated[
    str | None, typer.Option("--scene-schema", help="Scene schema to persist in exported output.")
]
SplitRatio = Annotated[
    tuple[float, float, float] | None,
    typer.Option(
        "--ratio",
        help="Train/val/test ratio used by source, scene, time, and shuffled-time split modes.",
        rich_help_panel="Split",
    ),
]
SplitGap = Annotated[
    int | None,
    typer.Option("--gap", help="Gap inserted between time partitions.", rich_help_panel="Split"),
]
SplitSegments = Annotated[
    int | None,
    typer.Option(
        "--segments",
        help="Number of contiguous temporal segments used by shuffled-time split mode.",
        rich_help_panel="Split",
    ),
]
Force = Annotated[
    bool, typer.Option("--force", "-f", help="Force processing without confirmation.")
]
DatasetName = Annotated[
    str, typer.Argument(help="The name of the dataset to apply the command to.")
]
Plan = Annotated[
    bool, typer.Option("--plan/--no-plan", help="Show the processing plan without executing it.")
]
IncludeMap = Annotated[
    bool | None, typer.Option("--include-map/--no-map", help="Include the map (if available).")
]


@app.command()
def process(
    *,
    dataset: DatasetName,
    input_dir: InputDir,
    output_dir: OutputDir,
    split: Split = None,
    read_split: ReadSplit = None,
    config: Config = None,
    jobs: Jobs = None,
    progress: Progress = True,
    limit: Limit = None,
    seed: Seed = None,
    storage_backend: StorageBackendOption = StorageBackend.MDS,
    trajectory_schema: TrajectorySchema = None,
    ratio: SplitRatio = None,
    gap: SplitGap = None,
    segments: SplitSegments = None,
    force: Force = False,
    plan: Plan = False,
    include_map: IncludeMap = None,
) -> None:
    """[bold]Process a specified dataset[/bold]."""
    from dronalize.runtime._internal.runner import open_job
    from dronalize.runtime.api import run_job

    job = _run_cli_action(
        lambda: _resolve_cli_job(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            storage_backend=storage_backend,
            config=config,
            split=split,
            read_split=read_split,
            jobs=jobs,
            trajectory_schema=trajectory_schema,
            ratio=ratio,
            gap=gap,
            segments=segments,
            include_map=include_map,
            limit=limit,
            seed=seed,
            input_dir_exists=not plan,
        )
    )

    if plan:
        rprint(PLAN_NOTICE)
        rprint(build_processing_summary_table(job))
        return

    if not force:
        rprint("\n", build_processing_summary_table(job))
        _ = typer.confirm("Proceed with this processing plan?", abort=True)

    if not progress:
        _ = _run_cli_action(lambda: run_job(job))
        return

    from dronalize.io.backends.registry import build_writer_factory
    from dronalize.runtime.cli.progress import run_with_rich_progress

    with open_job(job) as run:
        run_with_rich_progress(
            run.executor,
            lambda: _run_cli_action(
                lambda: run.executor.execute(writer_factory=build_writer_factory(job))
            ),
            enable=True,
        )
        job.write_manifests()


@app.command()
def available(
    *,
    details: Annotated[
        bool,
        typer.Option("--details/--no-details", help="Show detailed information about datasets."),
    ] = True,
) -> None:
    """[bold]List available datasets[/bold]."""
    from dronalize.datasets import available as available_datasets
    from dronalize.datasets.registry import get

    descriptors = _run_cli_action(lambda: [get(dataset) for dataset in available_datasets()])
    rprint("\n", build_available_datasets_table(descriptors, details=details))


@app.command()
def inspect(dataset: DatasetName) -> None:
    """[bold]Inspect details for a specified dataset[/bold]."""
    from dronalize.datasets.registry import get

    descriptor = _run_cli_action(lambda: get(dataset))
    _print_tables(build_dataset_inspect_tables(descriptor))


@app.command()
def show_config(
    dataset: DatasetName,
    split: Split = None,
    read_split: ReadSplit = None,
    config: Config = None,
    jobs: Jobs = None,
    storage_backend: StorageBackendOption = StorageBackend.MDS,
    trajectory_schema: TrajectorySchema = None,
    ratio: SplitRatio = None,
    gap: SplitGap = None,
    segments: SplitSegments = None,
    include_map: IncludeMap = None,
) -> None:
    """[bold]Show the resolved configuration for a specified dataset[/bold]."""
    job = _run_cli_action(
        lambda: _resolve_cli_job(
            dataset=dataset,
            input_dir=Path(),
            output_dir=Path(),
            storage_backend=storage_backend,
            config=config,
            split=split,
            read_split=read_split,
            jobs=jobs,
            trajectory_schema=trajectory_schema,
            ratio=ratio,
            gap=gap,
            segments=segments,
            include_map=include_map,
            input_dir_exists=False,
        )
    )
    rprint("\n[bold]Resolved Config:[/bold]")
    rprint(job.resolved_config)


@app.command()
def split_support(dataset: DatasetName) -> None:
    """[bold]Show the split support for a specified dataset[/bold]."""
    from dronalize.datasets.registry import get

    descriptor = _run_cli_action(lambda: get(dataset))
    _print_tables(build_split_support_tables(descriptor))


def main() -> None:
    """Run the optional Dronalize CLI application."""
    app()


def _print_tables(tables: Iterable[object]) -> None:
    for index, table in enumerate(tables):
        rprint("\n", table) if index == 0 else rprint(table)


def _run_cli_action(action: Callable[[], _T]) -> _T:
    try:
        return action()
    except click.ClickException:
        raise
    except FileNotFoundError as exc:
        raise _file_error(exc) from exc
    except ValidationError as exc:
        raise click.UsageError(_format_validation_error(exc)) from exc
    except (ConfigurationError, DatasetNotFoundError, SplitError) as exc:
        raise click.UsageError(str(exc)) from exc
    except MissingOptionalDependencyError as exc:
        raise click.ClickException(str(exc)) from exc
    except DronalizeError as exc:
        raise click.ClickException(str(exc)) from exc


def _file_error(exc: FileNotFoundError) -> click.ClickException:
    if exc.filename is None:
        return click.UsageError(str(exc))
    return click.FileError(filename=str(exc.filename))


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["Invalid configuration:"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "value"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)


def _resolve_cli_job(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    storage_backend: StorageBackend,
    config: Path | None,
    split: SplitStrategy | None,
    read_split: list[DatasetSplit] | None,
    jobs: int | None,
    trajectory_schema: str | None,
    ratio: tuple[float, float, float] | None,
    gap: int | None,
    segments: int | None,
    include_map: bool | None,
    limit: int | None = None,
    seed: int | None = None,
    input_dir_exists: bool = True,
) -> RunPlan:
    from dronalize.runtime import ProcessRequest, resolve_job

    return resolve_job(
        ProcessRequest(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            limit=limit,
            seed=seed,
            storage_backend=storage_backend,
            config_path=config,
            overrides=RuntimeOverride.from_inputs(
                split_strategy=split,
                read_split=read_split,
                jobs=jobs,
                trajectory_schema=trajectory_schema,
                ratio=ratio,
                gap=gap,
                segments=segments,
            ),
            include_map=include_map,
            input_dir_exists=input_dir_exists,
        )
    )
