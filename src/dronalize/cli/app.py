"""Typer application for the optional dronalize CLI."""

# ruff: noqa: PLC0415
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import typer
from rich import box
from rich import print as rprint
from rich.table import Table

from dronalize.categories import DatasetSplit

if TYPE_CHECKING:
    from dronalize.execution.runner import ProcessingSummary

app: typer.Typer = typer.Typer(help="Trajectory data processing package.", no_args_is_help=True)
_ = DatasetSplit | Path
OutputFormatLiteral = Literal["mds", "zarr", "dummy"]


@app.command()
def process(
    *,
    dataset: Annotated[str, typer.Argument(help="The name of the dataset to process.")],
    input_dir: Annotated[
        Path,
        typer.Option("--input", "-i", help="Directory containing the raw dataset."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory to save the processed dataset."),
    ],
    split: Annotated[
        list[DatasetSplit] | None,
        typer.Option(
            "--split",
            "-s",
            help="The predefined split of the dataset to process.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to the optional configuration file."),
    ] = None,
    jobs: Annotated[
        int | None,
        typer.Option(
            "--jobs",
            "-j",
            help="Worker count override. Values greater than 1 enable parallel execution.",
        ),
    ] = None,
    progress: Annotated[
        bool, typer.Option("--progress/--no-progress", help="Show progress during processing.")
    ] = True,
    limit: Annotated[
        int | None, typer.Option("--limit", "-l", help="Limit the number of samples to process.")
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Random seed.")] = None,
    output_format: Annotated[
        OutputFormatLiteral,
        typer.Option("--output-format", help="Output format for processed data."),
    ] = "mds",
    scene_schema: Annotated[
        str | None,
        typer.Option("--scene-schema", help="Scene schema to persist in writer output."),
    ] = None,
    custom_split: Annotated[
        tuple[float, float, float] | None,
        typer.Option("--custom-split", help="Custom split ratios for train/val/test splits."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force processing without confirmation.")
    ] = False,
) -> None:
    """[bold]Process a specified dataset[/bold]."""
    from dronalize.execution import prepare_dataset

    job = prepare_dataset(
        dataset=dataset,
        input_dir=input_dir,
        output_dir=output_dir,
        split=split,
        config_path=config,
        jobs=jobs,
        limit=limit,
        custom_split=custom_split,
        seed=seed,
        output_format=output_format,
        scene_schema=scene_schema,
    )

    if not force:
        rprint(_render_processing_summary(job.summary()))
        _ = typer.confirm("Proceed with this processing plan?", abort=True)

    from dronalize.cli.progress import run_with_rich_progress

    with job.open() as run:
        run_with_rich_progress(run.executor, run.run, enable=progress)


@app.command()
def available(
    *,
    details: Annotated[
        bool,
        typer.Option("--details/--no-details", help="Show detailed information about datasets."),
    ] = True,
) -> None:
    """List available datasets."""
    from dronalize.datasets import available as available_datasets
    from dronalize.datasets import get

    table = Table(
        title="Available Datasets",
        show_edge=True,
        show_lines=False,
        style="white bold",
    )
    table.add_column("Dataset Name", style="cyan", no_wrap=True)
    if details:
        table.add_column("Horizon (In/Out)", style="magenta")
        table.add_column("Sample Frequency", justify="left", style="yellow")
        table.add_column("Map", style="green", justify="center")
        table.add_column("Predefined splits", style="blue")

    for dataset in available_datasets():
        descriptor = get(dataset)
        if not details:
            table.add_row(descriptor.name)
            continue

        cfg = descriptor.default_config
        horizon = f"{cfg.input_len:>3} / {cfg.output_len:<3}"
        sample_frequency = f"{1 / cfg.sample_time:>3.1f} Hz"
        has_map = "[green]✓[/green]" if descriptor.has_map else "[red]✗[/red]"
        splits_display = (
            ", ".join(split.name.lower() for split in sorted(descriptor.predefined_splits))
            if descriptor.predefined_splits
            else "-"
        )
        table.add_row(descriptor.name, horizon, sample_frequency, has_map, splits_display)

    rprint("\n", table)


def main() -> None:
    """Run the Typer CLI application."""
    app()


def _render_processing_summary(summary: ProcessingSummary) -> Table:
    table = Table(title=summary.title, show_header=False, box=box.ROUNDED)
    table.add_column(style="cyan", justify="left", no_wrap=True)
    table.add_column(style="magenta")

    for label, value in summary.rows:
        table.add_row(label, value)

    return table
