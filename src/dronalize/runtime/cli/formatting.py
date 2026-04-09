"""Presentation helpers for the optional dronalize CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich import box
from rich.table import Table

from dronalize.core.categories import DatasetSplit
from dronalize.datasets.registry import DatasetCapabilities

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import BaseModel

    from dronalize.datasets.registry import DatasetSpec
    from dronalize.processing.loading.config import LoaderConfig
    from dronalize.processing.loading.splits import SplitStrategyName
    from dronalize.processing.maps.config import MapConfig
    from dronalize.runtime.models import ProcessingSummary

_DATASET_SPLIT_DISPLAY_ORDER = {DatasetSplit.TRAIN: 0, DatasetSplit.VAL: 1, DatasetSplit.TEST: 2}
_CUSTOM_SPLIT_STRATEGIES: tuple[SplitStrategyName, ...] = (
    "source",
    "scene",
    "time",
    "shuffled-time",
)
_CUSTOM_SPLIT_STRATEGY_DESCRIPTIONS: dict[SplitStrategyName, str] = {
    "source": "Assign complete sources such as recordings to a single split.",
    "scene": "Assign individual scenes while keeping each scene fully intact.",
    "time": "Split long recordings into chronological train/val/test blocks.",
    "shuffled-time": "Split long recordings into multiple temporal segments before routing.",
    "auto": "Resolve the recommended custom split strategy automatically.",
    "native": "Use dataset-defined train/val/test partitions as-is.",
    "none": "Process all data without split routing.",
}
PLAN_NOTICE = (
    "\n[bold yellow]This is the processing plan. No changes have been made yet.[/bold yellow]"
)
_CAPABILITY_BADGES: tuple[tuple[DatasetCapabilities, str, str], ...] = (
    (DatasetCapabilities.MAP_AVAILABLE, "green", "map"),
    (DatasetCapabilities.NATIVE_SPLITS, "cyan", "native splits"),
    (DatasetCapabilities.CUSTOM_SPLITS, "blue", "custom split strategies"),
    (DatasetCapabilities.LANE_CHANGE_SAMPLING, "magenta", "lane-change sampling"),
    (DatasetCapabilities.EXECUTION_SCOPE, "yellow", "runtime context"),
    (DatasetCapabilities.LOADER_OPTIONS, "red", "loader options"),
)


def build_processing_summary_table(summary: ProcessingSummary) -> Table:
    """Return a display table for a prepared processing summary."""
    table = Table(
        title=summary.title, show_header=False, box=box.MINIMAL_DOUBLE_HEAD, title_justify="left"
    )
    table.add_column(style="bright_cyan", justify="left", no_wrap=True)
    table.add_column(style="bright_magenta")
    for label, value in summary.rows:
        table.add_row(label, value)
    return table


def build_available_datasets_table(descriptors: Sequence[DatasetSpec], *, details: bool) -> Table:
    """Return a table describing the available datasets."""
    table = Table(
        title="Available datasets",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_edge=True,
        show_lines=False,
        header_style="bold",
        row_styles=["", "dim"],
    )
    table.add_column("Dataset", style="bright_cyan", no_wrap=True)
    if details:
        table.caption = (
            " Native splits are listed first, followed by custom split strategies.\n"
            " [yellow]*[/yellow] marks the recommended custom strategy."
        )
        table.caption_justify = "left"
        table.caption_style = "dim"
        table.add_column("Window @ Hz", justify="right", style="bright_magenta", no_wrap=True)
        table.add_column("Map", justify="center")
        table.add_column("Split support", style="bright_blue")

    for descriptor in descriptors:
        if not details:
            table.add_row(descriptor.name)
            continue

        cfg = descriptor.default_loader_config
        window = f"{cfg.input_len:>2}/{cfg.output_len:<3} @ {1 / cfg.sample_time:>4.1f}Hz"
        has_map = "[green]yes[/green]" if descriptor.has_map else "[dim]no[/dim]"
        table.add_row(
            descriptor.name,
            window,
            has_map,
            _format_split_support(
                descriptor.predefined_splits,
                descriptor.supported_split_strategies,
                descriptor.recommended_split_strategy,
            ),
        )
    return table


def build_dataset_inspect_tables(descriptor: DatasetSpec) -> tuple[Table, ...]:
    """Return the tables used by the `inspect` command."""
    config = descriptor.default_loader_config
    map_config = descriptor.default_map_config

    overview = _detail_table(title=f"Dataset inspect: {descriptor.name}")
    overview.add_row("Dataset", descriptor.name)
    overview.add_row("Capabilities", _format_capabilities(descriptor.capabilities))
    overview.add_row(
        "Native schema",
        f"{descriptor.native_schema.name} ({descriptor.native_schema.feature_dim} features)",
    )
    overview.add_row("Schema fields", ", ".join(descriptor.native_schema.semantic_fields()))
    overview.add_row(
        "Split support",
        _format_split_support(
            descriptor.predefined_splits,
            descriptor.supported_split_strategies,
            descriptor.recommended_split_strategy,
        ),
    )

    loader_defaults = _detail_table(title="Default loader config")
    loader_defaults.add_row(
        "Source window",
        _format_base_window(config.input_len, config.output_len, config.sample_time),
    )
    loader_defaults.add_row(
        "Effective window",
        (
            f"{config.resampled_input_len}/{config.resampled_output_len}"
            f" @ {1 / config.post_sample_time:.1f} Hz"
        ),
    )
    loader_defaults.add_row("Resampling", _format_resampling(config))
    loader_defaults.add_row("Windowing", _format_windowing(config))
    loader_defaults.add_row("Filter rules", _format_filter_rules(config))
    loader_defaults.add_row("Options", _format_options(descriptor.default_loader_options))

    tables: list[Table] = [overview, loader_defaults]
    if loader_options := _loader_option_rows(descriptor):
        loader_option_table = _detail_table(title="Loader-specific Options")
        for name, default in loader_options:
            loader_option_table.add_row(name, default)
        tables.append(loader_option_table)

    map_defaults = _detail_table(title="Default map config")
    map_defaults.add_row("Enabled", _format_flag(enabled=map_config is not None))
    if map_config is not None:
        map_defaults.add_row("Extraction", _format_map_extraction(map_config))
        map_defaults.add_row("Min distance", _format_optional_float(map_config.min_distance))
        map_defaults.add_row("Interp distance", _format_optional_float(map_config.interp_distance))
    tables.append(map_defaults)
    return tuple(tables)


def build_split_support_tables(descriptor: DatasetSpec) -> tuple[Table, ...]:
    """Return the tables used by the `split-support` command."""
    summary = _detail_table(title=f"Split support: {descriptor.name}")
    summary.add_row("Dataset", descriptor.name)
    summary.add_row("Native splits", _format_split_list(descriptor.predefined_splits))
    summary.add_row(
        "Custom strategies",
        _format_mode_list(
            descriptor.supported_split_strategies, descriptor.recommended_split_strategy
        ),
    )
    summary.add_row(
        "Recommended custom strategy",
        descriptor.recommended_split_strategy or "[dim]none[/dim]",
    )

    native_table = Table(
        title="Native split matrix",
        box=box.MINIMAL_DOUBLE_HEAD,
        title_justify="left",
        header_style="bold",
    )
    native_table.add_column("Split", style="bright_cyan", no_wrap=True)
    native_table.add_column("Supported", justify="center")
    supported_splits = set(descriptor.predefined_splits)
    for split in _sorted_splits(list(DatasetSplit)):
        native_table.add_row(split.value, _format_flag(enabled=split in supported_splits))

    custom_table = Table(
        title="Custom split strategies",
        box=box.MINIMAL_DOUBLE_HEAD,
        title_justify="left",
        header_style="bold",
    )
    custom_table.add_column("Strategy", style="bright_cyan", no_wrap=True)
    custom_table.add_column("Supported", justify="center")
    custom_table.add_column("Recommended", justify="center")
    custom_table.add_column("Details", style="bright_magenta")

    supported_strategies = set(descriptor.supported_split_strategies)
    for strategy in _CUSTOM_SPLIT_STRATEGIES:
        row_style = "" if strategy in supported_strategies else "dim"
        custom_table.add_row(
            strategy,
            _format_flag(enabled=strategy in supported_strategies),
            _format_flag(enabled=strategy == descriptor.recommended_split_strategy),
            _CUSTOM_SPLIT_STRATEGY_DESCRIPTIONS[strategy],
            style=row_style,
        )

    return summary, native_table, custom_table


def _detail_table(title: str, *, caption: str | None = None) -> Table:
    """Return a standard two-column detail table."""
    table = Table(
        title=title,
        show_header=False,
        box=box.MINIMAL_DOUBLE_HEAD,
        title_justify="left",
        caption=caption,
        caption_style="dim",
        caption_justify="left",
    )
    table.add_column(style="bright_cyan", justify="left", no_wrap=True)
    table.add_column(style="bright_magenta")
    return table


def _format_split_support(
    predefined_splits: list[DatasetSplit],
    supported_split_strategies: list[SplitStrategyName],
    recommended_split_strategy: SplitStrategyName | None,
) -> str:
    """Return a compact summary of native and custom split support."""
    native = ", ".join(split.value for split in _sorted_splits(predefined_splits))
    custom = _format_supported_modes(supported_split_strategies, recommended_split_strategy)
    if native and custom:
        return f"{native} [dim]|[/dim] {custom}"
    if native:
        return native
    if custom:
        return custom
    return "-"


def _format_flag(*, enabled: bool) -> str:
    """Return a consistent yes/no flag for terminal output."""
    return "[green]yes[/green]" if enabled else "[dim]no[/dim]"


def _format_capabilities(capabilities: DatasetCapabilities) -> str:
    """Return a compact badge row describing enabled dataset capabilities."""
    badges = [
        f"[black on {color}] {label} [/]"
        for flag, color, label in _CAPABILITY_BADGES
        if capabilities & flag
    ]
    return " ".join(badges) if badges else "[dim]none[/dim]"


def _sorted_splits(predefined_splits: list[DatasetSplit]) -> list[DatasetSplit]:
    """Return predefined splits in stable train/val/test display order."""
    return sorted(
        predefined_splits,
        key=lambda split: _DATASET_SPLIT_DISPLAY_ORDER.get(
            split, len(_DATASET_SPLIT_DISPLAY_ORDER)
        ),
    )


def _format_split_list(predefined_splits: list[DatasetSplit]) -> str:
    """Return a readable list of predefined dataset splits."""
    splits = _sorted_splits(predefined_splits)
    return ", ".join(split.value for split in splits) if splits else "[dim]none[/dim]"


def _format_mode_list(
    supported_split_strategies: list[SplitStrategyName],
    recommended_split_strategy: SplitStrategyName | None,
) -> str:
    """Return a readable list of supported custom split strategies."""
    if not supported_split_strategies:
        return "[dim]none[/dim]"
    return _format_supported_modes(supported_split_strategies, recommended_split_strategy)


def _format_base_window(input_len: int, output_len: int, sample_time: float) -> str:
    """Return the original un-resampled sequence summary."""
    return f"{input_len}/{output_len} @ {1 / sample_time:.1f} Hz"


def _format_resampling(config: LoaderConfig) -> str:
    """Return a short description of the active resampling configuration."""
    spec = config.resampling
    if spec is None or spec.no_resampling:
        return "none"
    return f"{spec.up}:{spec.down} ({spec.method.value})"


def _format_windowing(config: LoaderConfig) -> str:
    """Return a short description of the active sliding-window configuration."""
    window = config.window
    if window is None:
        return "disabled"
    return f"{window.size} frames, step {window.step}"


def _format_filter_rules(config: LoaderConfig) -> str:
    """Return a readable summary of configured filter rules."""
    scene_filter = config.filter
    if scene_filter is None:
        return "none"

    groups: list[str] = []
    if scene_filter.cleanup_rules:
        groups.append("cleanup: " + ", ".join(rule.name() for rule in scene_filter.cleanup_rules))
    if scene_filter.scene_rules:
        groups.append("scene: " + ", ".join(rule.name() for rule in scene_filter.scene_rules))
    if scene_filter.agent_rules:
        groups.append("agent: " + ", ".join(rule.name() for rule in scene_filter.agent_rules))
    return " | ".join(groups) if groups else "none"


def _format_options(options: BaseModel) -> str:
    """Return a readable summary of loader options."""
    options_dict: dict[str, Any] = {}
    if hasattr(options, "model_dump"):
        options_dict = options.model_dump(exclude_defaults=False)
    if not options_dict:
        return "none"
    return ", ".join(f"{key}={value!r}" for key, value in sorted(options_dict.items()))


def _format_map_extraction(map_config: MapConfig) -> str:
    """Return a readable summary of the default map extraction mode."""
    extraction = map_config.extraction
    match extraction.mode:
        case "full":
            return "full map"
        case "scene_extent":
            return f"scene extent (padding={extraction.padding:g})"
        case "circle":
            return f"circle (radius={extraction.radius:g})"
        case "bounding_box":
            return f"bounding box ({extraction.width:g} x {extraction.height:g})"


def _format_optional_float(value: float | None) -> str:
    """Return a compact float or a dimmed placeholder."""
    return f"{value:g}" if value is not None else "[dim]none[/dim]"


def _format_supported_modes(
    supported_modes: list[SplitStrategyName], recommended_mode: SplitStrategyName | None
) -> str:
    return ", ".join(
        f"{mode}[yellow]*[/yellow]" if mode == recommended_mode else mode
        for mode in supported_modes
    )


def _loader_option_rows(descriptor: DatasetSpec) -> list[tuple[str, str]]:
    """Return typed loader option fields and their defaults."""
    default_values = descriptor.default_loader_options.model_dump(exclude_defaults=False)
    rows: list[tuple[str, str]] = []
    for name, field in descriptor.loader_options_type.model_fields.items():
        value = default_values.get(name)
        default = "[yellow]required[/yellow]" if field.is_required() else repr(value)
        rows.append((name, default))
    return rows
