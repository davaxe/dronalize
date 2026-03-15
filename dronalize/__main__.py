from pathlib import Path
from typing import Annotated, Literal

import typer

app: typer.Typer = typer.Typer(help="Trajectory data processing package.", no_args_is_help=True)


# Redefined here to avoid importing runner module
OutputFormat = Literal["mds"]


@app.command()
def process(
    *,
    dataset: Annotated[str, typer.Argument(help="The name of the dataset to process.")],
    input_dir: Annotated[
        Path, typer.Option("--input", "-i", help="Directory containing the raw dataset.")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="Directory to save the processed dataset.")
    ],
    split: Annotated[
        list[str] | None,
        typer.Option(
            "--split",
            "-s",
            help="The split of the dataset to process.",
            show_default=False,
        ),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to the optional configuration file.")
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
        bool, typer.Option("--progress/--no-progress", help="Show progress bar during processing.")
    ] = True,
    limit: Annotated[
        int | None, typer.Option("--limit", "-l", help="Limit the number of samples to process.")
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Random seed."),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--output-format", help="Output format for processed data."),
    ] = "mds",
    custom_split: Annotated[
        tuple[float, float, float] | None,
        typer.Option("--custom-split", help="Custom split ratios for train/val/test splits."),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force processing without confirmation.")
    ] = False,
) -> None:
    """[bold]Process a specified dataset[/bold].

    Requires the name of the dataset to process. Available datasets can
    be listed using the [bold cyan]available[/cyan bold] command.

    [bold underline]Configuration[/bold underline]
    The dataset can be customized using a TOML configuration file. This can
    be specified using the `--config` option. This config can be used to
    override defaults when it comes to things like filtering, resampling and map
    processing. For more details on the configuration options, see the
    documentation.

    [bold underline]Splits[/bold underline]
    For dataset with predefined splits, the split argument can be used to
    specify which split to process. If not specified, the default is to process
    all available data together without dividing the output into splits. If the
    dataset does not have the specified split, an error will be raised. The
    possible splits are "train", "val", and "test".

    Example usage:
        dronalize process my_dataset --split train

    For multiple splits (note -s can be used):
        dronalize process my_dataset -s train -s val

    [bold underline]Custom splits[/bold underline]
    In the case where a custom randomized split is desired the `--custom-split`
    option can be used to specify the split ratios for train/val/test splits.
    Three values must be provided, representing the ratios for train, val, and
    test splits. Putting a ratio to zero will exclude that split from the
    dataset completely. Custom split ratios override any predefined dataset
    splits.

    Example usage:
        dronalize process my_dataset --custom-split 0.7 0.2 0.1

    If the `--custom-split` option is used, no split should be specified using
    the `--split` option.

    """
    if not force:
        from rich import print as rprint  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415

        summary_table = Table(title="Processing Configuration", show_header=False)
        summary_table.add_column("Parameter", style="cyan", justify="right")
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row("Dataset", dataset)
        summary_table.add_row("Input Directory", str(input_dir))
        summary_table.add_row("Output Directory", str(output_dir))
        summary_table.add_row("Output Format", str(output_format))

        if split:
            summary_table.add_row("Split", str(split))
        if custom_split:
            summary_table.add_row("Custom Split", str(custom_split))
        if limit:
            summary_table.add_row("Limit", str(limit))

        # 2. Display the table
        rprint(summary_table)
        if not typer.confirm("Proceed with these settings?"):
            rprint("[bold red]Processing aborted.[/bold red]")
            raise typer.Abort

    from dronalize.execution import runner  # noqa: PLC0415

    runner.entrypoint(
        dataset=dataset,
        input_dir=input_dir,
        output_dir=output_dir,
        split=split,
        config_path=config,
        jobs=jobs,
        progress=progress,
        limit=limit,
        custom_split=custom_split,
        seed=seed,
        output_format=output_format,
    )


@app.command()
def available(
    *,
    details: Annotated[
        bool,
        typer.Option("--details/--no-details", help="Show detailed information about datasets."),
    ] = True,
) -> None:
    """[bold]List available datasets[/bold]."""
    # Lazy import to make CLI more responsive.
    from rich import print as rprint  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    from dronalize.datasets import available as _available  # noqa: PLC0415
    from dronalize.datasets import get as _get  # noqa: PLC0415

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

    for dataset in _available():
        descriptor = _get(dataset)
        if details:
            cfg = descriptor.default_config
            horizon = f"{cfg.input_len:>3} / {cfg.output_len:<3}"
            sample_frequency = f"{1 / cfg.sample_time:>3.1f} Hz"
            has_map = "[green]✓[/green]" if descriptor.has_map else "[red]✗[/red]"
            if descriptor.predefined_splits:
                splits_display = ", ".join(
                    s.name.lower() for s in sorted(descriptor.predefined_splits)
                )
            else:
                splits_display = "-"

            table.add_row(descriptor.name, horizon, sample_frequency, has_map, splits_display)
        else:
            table.add_row(descriptor.name)

    rprint("\n", table)


def main() -> None:
    """CLI entry point for dronalize."""
    app()
