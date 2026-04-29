"""Typer application for the optional Dronalize CLI."""

# ruff: noqa: PLC0415
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeVar

import click
import typer
from pydantic import ValidationError
from rich import print as rprint

from dronalize import __version__
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

    from dronalize.runtime.types import ExecutionPlan


app: typer.Typer = typer.Typer(help="Trajectory data processing toolkit.", no_args_is_help=True)
_T = TypeVar("_T")

ReadStrategy = Literal["all", "native"]
AssignStrategy = Literal["none", "preserve-native", "scene", "source", "time", "shuffled-time"]

InputDir = Annotated[
    Path, typer.Option("--input", "-i", help="Directory containing the raw dataset.")
]
OutputDir = Annotated[
    Path, typer.Option("--output", "-o", help="Directory to save the processed dataset.")
]
Read = Annotated[
    ReadStrategy | None,
    typer.Option("--read", help="Read mode to use.", show_default=False, rich_help_panel="Read"),
]
ReadSplit = Annotated[
    list[DatasetSplit] | None,
    typer.Option(
        "--read-split",
        help="Dataset-defined partition(s) to read when --read native is selected.",
        show_default=False,
        rich_help_panel="Read",
    ),
]
Assign = Annotated[
    AssignStrategy | None,
    typer.Option(
        "--assign",
        help="Output assignment mode to use.",
        show_default=False,
        rich_help_panel="Assign",
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
    StorageBackend,
    typer.Option("--storage-backend", "--sb", help="Storage backend for processed data."),
]
TrajectorySchema = Annotated[
    str | None, typer.Option("--scene-schema", help="Scene schema to persist in exported output.")
]
SplitRatio = Annotated[
    tuple[float, float, float] | None,
    typer.Option(
        "--ratio",
        help="Train/val/test ratio used by scene, source, time, and shuffled-time assignment.",
        rich_help_panel="Assign",
    ),
]
SplitGap = Annotated[
    int | None,
    typer.Option("--gap", help="Gap inserted between time partitions.", rich_help_panel="Assign"),
]
SplitSegments = Annotated[
    int | None,
    typer.Option(
        "--segments",
        help="Number of contiguous temporal segments used by shuffled-time assignment.",
        rich_help_panel="Assign",
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


def _version_callback(*, version: bool) -> None:
    if version:
        rprint(f"dronalize version: {__version__}")
        raise typer.Exit


@app.callback()
def version(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = None,
) -> None:
    """Version callback to display the version."""
    _ = version


@app.command()
def process(
    *,
    dataset: DatasetName,
    input_dir: InputDir,
    output_dir: OutputDir,
    read: Read = None,
    read_split: ReadSplit = None,
    assign: Assign = None,
    config: Config = None,
    jobs: Jobs = None,
    progress: Progress = True,
    limit: Limit = None,
    seed: Seed = None,
    storage_backend: StorageBackendOption = StorageBackend.PICKLE,
    trajectory_schema: TrajectorySchema = None,
    ratio: SplitRatio = None,
    gap: SplitGap = None,
    segments: SplitSegments = None,
    force: Force = False,
    plan_mode: Plan = False,
    include_map: IncludeMap = None,
) -> None:
    """[bold]Process a specified dataset[/bold]."""
    from dronalize.runtime._internal.runner import open_execution_session
    from dronalize.runtime.api import execute_plan

    plan = _run_cli_action(
        lambda: _resolve_cli_plan(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            storage_backend=storage_backend,
            config=config,
            read=read,
            read_split=read_split,
            assign=assign,
            jobs=jobs,
            trajectory_schema=trajectory_schema,
            ratio=ratio,
            gap=gap,
            segments=segments,
            include_map=include_map,
            limit=limit,
            seed=seed,
            input_dir_exists=not plan_mode,
        )
    )

    if plan_mode:
        rprint(PLAN_NOTICE)
        rprint(build_processing_summary_table(plan))
        return

    if not force:
        rprint("\n", build_processing_summary_table(plan))
        _ = typer.confirm("Proceed with this processing plan?", abort=True)

    if not progress:
        _ = _run_cli_action(lambda: execute_plan(plan))
        return

    from dronalize.io.backends.registry import build_writer_factory
    from dronalize.runtime.cli.progress import run_with_rich_progress

    with open_execution_session(plan) as run:
        run_with_rich_progress(
            run.executor,
            lambda: _run_cli_action(
                lambda: run.executor.execute(writer_factory=build_writer_factory(plan))
            ),
            enable=True,
        )
        plan.write_manifests()


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
    read: Read = None,
    read_split: ReadSplit = None,
    assign: Assign = None,
    config: Config = None,
    jobs: Jobs = None,
    storage_backend: StorageBackendOption = StorageBackend.PICKLE,
    trajectory_schema: TrajectorySchema = None,
    ratio: SplitRatio = None,
    gap: SplitGap = None,
    segments: SplitSegments = None,
    include_map: IncludeMap = None,
) -> None:
    """[bold]Show the resolved configuration for a specified dataset[/bold]."""
    plan = _run_cli_action(
        lambda: _resolve_cli_plan(
            dataset=dataset,
            input_dir=Path(),
            output_dir=Path(),
            storage_backend=storage_backend,
            config=config,
            read=read,
            read_split=read_split,
            assign=assign,
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
    rprint(plan.resolved_config)


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


def _resolve_cli_plan(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    storage_backend: StorageBackend,
    config: Path | None,
    read: ReadStrategy | None,
    read_split: list[DatasetSplit] | None,
    assign: AssignStrategy | None,
    jobs: int | None,
    trajectory_schema: str | None,
    ratio: tuple[float, float, float] | None,
    gap: int | None,
    segments: int | None,
    include_map: bool | None,
    limit: int | None = None,
    seed: int | None = None,
    input_dir_exists: bool = True,
) -> ExecutionPlan:
    from dronalize.runtime import ExecutionRequest, resolve_request

    return resolve_request(
        ExecutionRequest(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            limit=limit,
            seed=seed,
            storage_backend=storage_backend,
            config_path=config,
            overrides=RuntimeOverride.from_inputs(
                read_strategy=read,
                read_split=read_split,
                assign_strategy=assign,
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
