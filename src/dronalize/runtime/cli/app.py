"""Typer application for the optional dronalize CLI."""

# ruff: noqa: PLC0415
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TypeVar

import click
import typer
from pydantic import ValidationError
from rich import print as rprint

from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import (
    ConfigurationError,
    DatasetNotFoundError,
    DronalizeError,
    MissingOptionalDependencyError,
    SplitError,
)
from dronalize.io.formats import OutputFormat
from dronalize.processing.ingest.splits import SplitModeName
from dronalize.runtime.cli.formatting import (
    PLAN_NOTICE,
    build_available_datasets_table,
    build_dataset_inspect_tables,
    build_processing_summary_table,
    build_split_support_tables,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

app: typer.Typer = typer.Typer(help="Trajectory data processing package.", no_args_is_help=True)
_T = TypeVar("_T")

InputDir = Annotated[
    Path, typer.Option("--input", "-i", help="Directory containing the raw dataset.")
]
OutputDir = Annotated[
    Path,
    typer.Option("--output", "-o", help="Directory to save the processed dataset."),
]
Split = Annotated[
    SplitModeName | None,
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
    int | None, typer.Option("--limit", "-l", help="Limit the number of samples to process.")
]
Seed = Annotated[
    int | None,
    typer.Option(
        "--seed",
        help="Random seed used by custom split assignment and other randomized operations.",
    ),
]
OutputFormatOption = Annotated[
    OutputFormat, typer.Option("--output-format", help="Output format for processed data.")
]
SceneSchema = Annotated[
    str | None, typer.Option("--scene-schema", help="Scene schema to persist in writer output.")
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
    typer.Option(
        "--gap",
        help="Gap inserted between time or shuffled-time partitions.",
        rich_help_panel="Split",
    ),
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
DatasetName = Annotated[str, typer.Argument(help="The name of the dataset apply the command to.")]

Plan = Annotated[
    bool, typer.Option("--plan/--no-plan", help="Show the processing plan without executing it.")
]
IncludeMap = Annotated[
    bool | None, typer.Option("--include-map/--no-map", help="Include the map (if availabe).")
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
    output_format: OutputFormatOption = OutputFormat.MDS,
    scene_schema: SceneSchema = None,
    ratio: SplitRatio = None,
    gap: SplitGap = None,
    segments: SplitSegments = None,
    force: Force = False,
    plan: Plan = False,
    include_map: IncludeMap = None,
) -> None:
    """[bold]Process a specified dataset[/bold]."""
    from dronalize.runtime import plan_dataset

    plan_obj = _run_cli_action(
        lambda: plan_dataset(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            split=split,
            read_split=read_split,
            jobs=jobs,
            limit=limit,
            seed=seed,
            output_format=output_format,
            scene_schema=scene_schema,
            ratio=ratio,
            gap=gap,
            segments=segments,
            config_path=config,
            include_map=include_map,
            input_dir_exists=not plan,
        )
    )

    if plan:
        rprint(PLAN_NOTICE)
        rprint(build_processing_summary_table(plan_obj.summary()))
        return

    if not force:
        rprint("\n", build_processing_summary_table(plan_obj.summary()))
        _ = typer.confirm("Proceed with this processing plan?", abort=True)

    from dronalize.runtime.cli.progress import run_with_rich_progress

    with plan_obj.open() as run:
        run_with_rich_progress(run.executor, run.run, enable=progress)


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
    from dronalize.datasets import get

    descriptors = _run_cli_action(lambda: [get(dataset) for dataset in available_datasets()])
    rprint("\n", build_available_datasets_table(descriptors, details=details))


@app.command()
def inspect(dataset: DatasetName) -> None:
    """[bold]Inspect details for a specified dataset[/bold]."""
    from dronalize.datasets import get

    descriptor = _run_cli_action(lambda: get(dataset))
    _print_tables(build_dataset_inspect_tables(descriptor))


@app.command()
def show_config(
    dataset: DatasetName,
    split: Split = None,
    read_split: ReadSplit = None,
    config: Config = None,
    jobs: Jobs = None,
    output_format: OutputFormatOption = OutputFormat.MDS,
    scene_schema: SceneSchema = None,
    ratio: SplitRatio = None,
    gap: SplitGap = None,
    segments: SplitSegments = None,
) -> None:
    """[bold]Show the raw configuration for a specified dataset[/bold].

    The options and arguments mirrors those of the [code]process[/code] command,
    but instead of executing the processing job, it simply prints the effective
    configuration that would be used for processing. This is useful for
    debugging and verification purposes.

    """
    from dronalize.runtime import plan_dataset

    plan_obj = _run_cli_action(
        lambda: plan_dataset(
            dataset=dataset,
            input_dir=Path(),
            output_dir=Path(),
            split=split,
            read_split=read_split,
            jobs=jobs,
            output_format=output_format,
            scene_schema=scene_schema,
            ratio=ratio,
            gap=gap,
            segments=segments,
            config_path=config,
        )
    )
    rprint("\n[bold]Effective Configuration:[/bold]")
    rprint(plan_obj.config)


@app.command()
def split_support(dataset: DatasetName) -> None:
    """[bold]Show the split support for a specified dataset[/bold]."""
    from dronalize.datasets import get

    descriptor = _run_cli_action(lambda: get(dataset))
    _print_tables(build_split_support_tables(descriptor))


def main() -> None:
    """Run the Typer CLI application."""
    app()


def _print_tables(tables: Iterable[object]) -> None:
    for index, table in enumerate(tables):
        rprint("\n", table) if index == 0 else rprint(table)


def _run_cli_action(action: Callable[[], _T]) -> _T:
    """Execute one CLI action and translate known user-facing errors."""
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
