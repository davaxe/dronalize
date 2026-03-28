"""Typer application for the optional dronalize CLI."""

# ruff: noqa: PLC0415
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich import print as rprint
from rich.table import Table

from dronalize.core.categories import DatasetSplit
from dronalize.io.formats import OutputFormat
from dronalize.processing.ingest.splits import SplitStrategyName

app: typer.Typer = typer.Typer(help="Trajectory data processing package.", no_args_is_help=True)
# Typer resolves these annotation types at runtime when building the CLI.
_RUNTIME_OPTION_TYPES = (DatasetSplit, Path, SplitStrategyName)
_DATASET_SPLIT_DISPLAY_ORDER = {
    DatasetSplit.TRAIN: 0,
    DatasetSplit.VAL: 1,
    DatasetSplit.TEST: 2,
}


def _format_split_support(
    predefined_splits: list[DatasetSplit],
    supported_split_strategies: list[SplitStrategyName],
    recommended_split_strategy: SplitStrategyName | None,
) -> str:
    """Return a compact summary of native and custom split support."""
    native = ", ".join(
        split.value
        for split in sorted(
            predefined_splits,
            key=lambda split: _DATASET_SPLIT_DISPLAY_ORDER.get(
                split, len(_DATASET_SPLIT_DISPLAY_ORDER)
            ),
        )
    )
    custom_strategies = [
        f"{strategy}[yellow]*[/yellow]" if strategy == recommended_split_strategy else strategy
        for strategy in supported_split_strategies
    ]
    custom = ", ".join(custom_strategies)
    if native and custom:
        return f"{native} [dim]•[/dim] {custom}"
    if native:
        return native
    if custom:
        return custom
    return "-"


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
            help="Dataset-defined partition(s) to read, such as train/val/test.",
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
        bool,
        typer.Option("--progress/--no-progress", help="Show progress during processing."),
    ] = True,
    limit: Annotated[
        int | None,
        typer.Option("--limit", "-l", help="Limit the number of samples to process."),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Random seed used by custom split assignment and other randomized operations.",
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--output-format", help="Output format for processed data."),
    ] = OutputFormat.MDS,
    scene_schema: Annotated[
        str | None,
        typer.Option("--scene-schema", help="Scene schema to persist in writer output."),
    ] = None,
    split_strategy: Annotated[
        SplitStrategyName | None,
        typer.Option(
            "--split-strategy",
            help=(
                "Loader-side split strategy. Use 'auto' to pick the recommended "
                "strategy when the loader exposes one."
            ),
        ),
    ] = None,
    split_weights: Annotated[
        tuple[float, float, float] | None,
        typer.Option(
            "--split-weights",
            help="Custom train/val/test weights used by loader-side split assignment.",
        ),
    ] = None,
    split_gap: Annotated[
        int | None,
        typer.Option(
            "--split-gap",
            help="Gap inserted between time-block or shuffled time-block partitions.",
        ),
    ] = None,
    split_n_segments: Annotated[
        int | None,
        typer.Option(
            "--split-n-segments",
            help=(
                "Number of contiguous temporal segments used by the "
                "'shuffled_time_blocks' split strategy."
            ),
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force processing without confirmation."),
    ] = False,
) -> None:
    """[bold]Process a specified dataset[/bold]."""
    from dronalize.runtime.execution import prepare_dataset

    job = prepare_dataset(
        dataset=dataset,
        input_dir=input_dir,
        output_dir=output_dir,
        split=split,
        jobs=jobs,
        limit=limit,
        seed=seed,
        output_format=output_format,
        scene_schema=scene_schema,
        split_strategy=split_strategy,
        split_weights=split_weights,
        split_gap=split_gap,
        split_n_segments=split_n_segments,
        config_path=config,
    )

    if not force:
        summary = job.summary()
        table = Table(
            title=summary.title,
            show_header=False,
            box=box.MINIMAL_DOUBLE_HEAD,
            title_justify="left",
        )
        table.add_column(style="cyan", justify="left", no_wrap=True)
        table.add_column(style="magenta")
        for label, value in summary.rows:
            table.add_row(label, value)
        rprint("\n", table)
        _ = typer.confirm("Proceed with this processing plan?", abort=True)

    from dronalize.runtime.cli.progress import run_with_rich_progress

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
        box=box.MINIMAL_DOUBLE_HEAD,
        show_edge=True,
        show_lines=False,
        header_style="bold",  # Applied bold exclusively to headers
        row_styles=["", "dim"],
    )
    table.add_column("Dataset", style="cyan", no_wrap=True)
    if details:
        table.caption = (
            " Native splits are listed first, followed by custom strategies.\n"
            " [yellow]*[/yellow] marks the recommended custom strategy."
        )
        table.caption_justify = "left"
        table.caption_style = "dim"

        table.add_column("Window @ Hz", justify="right", style="magenta", no_wrap=True)
        table.add_column("Map", justify="center")
        table.add_column("Split Support", style="blue")

    for dataset in available_datasets():
        descriptor = get(dataset)
        if not details:
            table.add_row(descriptor.name)
            continue

        cfg = descriptor.default_config
        input_len_padded = f"{cfg.input_len:>2}"
        output_len_padded = f"{cfg.output_len:<3}"
        freq_padded = f"{1 / cfg.sample_time:>4.1f}"
        window = f"{input_len_padded}/{output_len_padded} @ {freq_padded}Hz"
        has_map = "[green]✓[/green]" if descriptor.has_map else "[red]✗[/red]"
        split_support = _format_split_support(
            descriptor.predefined_splits,
            descriptor.supported_split_strategies,
            descriptor.recommended_split_strategy,
        )
        table.add_row(
            descriptor.name,
            window,
            has_map,
            split_support,
        )
    rprint("\n", table)


def main() -> None:
    """Run the Typer CLI application."""
    app()
