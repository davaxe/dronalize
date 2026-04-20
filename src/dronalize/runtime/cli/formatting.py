"""Presentation helpers for the optional dronalize CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich import box
from rich.table import Table

from dronalize.config.models import (
    NativeSplitConfig,
    NoSplitConfig,
    SceneSplitConfig,
    ShuffledTimeSplitConfig,
    SourceSplitConfig,
    SplitConfig,
    SplitConfigUnion,
    TimeSplitConfig,
    effective_scene_window,
)
from dronalize.core.scene import get_trajectory_schema

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.config.models import MapConfig, ScenesConfig, ScreeningConfig
    from dronalize.datasets.registry import DatasetSpec
    from dronalize.runtime.types import ExecutionPlan

PLAN_NOTICE = (
    "\n[bold yellow]This is the processing plan. No changes have been made yet.[/bold yellow]"
)


def build_processing_summary_table(plan: ExecutionPlan) -> Table:
    """Build a rich table summarizing one resolved processing plan."""
    table = Table(
        title=f"Processing plan: {plan.dataset}",
        show_header=False,
        box=box.MINIMAL_DOUBLE_HEAD,
        title_justify="left",
    )
    table.add_column(style="bright_cyan", justify="left", no_wrap=True)
    table.add_column(style="bright_magenta")
    for label, value in summarize_plan(plan):
        table.add_row(label, value)
    return table


def summarize_plan(plan: ExecutionPlan) -> tuple[tuple[str, str], ...]:
    """Return label/value rows for a resolved plan summary."""
    output_config = plan.output.inner
    return (
        ("Dataset", plan.dataset),
        ("Input", str(plan.data_root)),
        ("Output", str(plan.output_dir)),
        ("Backend", plan.storage_backend.value),
        ("Workers", str(plan.runtime.jobs)),
        ("Limit", "none" if plan.limit is None else str(plan.limit)),
        ("Schema", get_trajectory_schema(output_config.trajectory_schema).name),
        ("Map", "yes" if plan.map is not None else "no"),
        ("Split strategy", _run_plan_split_strategy(plan)),
        ("Split details", _split_detail(plan)),
        ("Options", _format_options(plan.loader.dataset)),
    )


def build_available_datasets_table(descriptors: Sequence[DatasetSpec], *, details: bool) -> Table:
    """Build a rich table showing the available dataset registry entries."""
    table = Table(
        title="Available datasets",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_edge=True,
        header_style="bold",
        row_styles=["", "dim"],
    )
    table.add_column("Dataset", style="bright_cyan", no_wrap=True)
    if details:
        table.add_column("Window @ Hz", justify="right", style="bright_magenta", no_wrap=True)
        table.add_column("Map", justify="center")
        table.add_column("Native splits", style="bright_blue")
        table.add_column("Time split", justify="center")

    for descriptor in descriptors:
        if not details:
            table.add_row(descriptor.name)
            continue

        cfg = descriptor.default_config.scenes
        window = _format_base_window(cfg.history_frames, cfg.future_frames, cfg.sample_time)
        has_map = "[green]yes[/green]" if descriptor.has_map else "[dim]no[/dim]"
        native_splits = (
            ", ".join(split.value for split in descriptor.native_splits or ()) or "[dim]none[/dim]"
        )
        table.add_row(
            descriptor.name,
            window,
            has_map,
            native_splits,
            _format_flag(enabled=descriptor.split_support.time_block),
        )
    return table


def build_dataset_inspect_tables(descriptor: DatasetSpec) -> tuple[Table, ...]:
    """Build one or more inspection tables for a dataset descriptor."""
    scenes = descriptor.default_config.scenes
    screening_config = descriptor.default_config.screening
    map_config = descriptor.default_config.map if descriptor.has_map else None
    output_config = descriptor.default_config.output
    split_config = descriptor.default_config.split

    overview = _detail_table(title=f"Dataset inspect: {descriptor.name}")
    overview.add_row("Dataset", descriptor.name)
    overview.add_row(
        "Native schema",
        f"{descriptor.native_schema.name} ({descriptor.native_schema.feature_dim} features)",
    )
    overview.add_row("Schema fields", ", ".join(descriptor.native_schema.semantic_fields()))
    overview.add_row(
        "Native splits",
        ", ".join(split.value for split in descriptor.native_splits or ()) or "[dim]none[/dim]",
    )
    overview.add_row("Supported split modes", _format_split_modes(descriptor))
    overview.add_row("Map", _format_flag(enabled=descriptor.has_map))

    scene_defaults = _detail_table(title="Default scene settings")
    scene_defaults.add_row(
        "Source window",
        _format_base_window(scenes.history_frames, scenes.future_frames, scenes.sample_time),
    )
    scene_defaults.add_row("Effective window", _format_effective_window(scenes))
    scene_defaults.add_row("Resampling", _format_resampling(scenes))
    scene_defaults.add_row("Windowing", _format_windowing(scenes))
    scene_defaults.add_row("Screening rules", _format_screening_rules(screening_config))
    scene_defaults.add_row("Dataset config", _format_options(descriptor.default_dataset_options()))

    output_defaults = _detail_table(title="Default output settings")
    output_defaults.add_row("Schema", get_trajectory_schema(output_config.trajectory_schema).name)
    output_defaults.add_row("Precision", output_config.precision)
    output_defaults.add_row(
        "Recenter positions", _format_flag(enabled=output_config.recenter_positions)
    )

    split_defaults = _detail_table(title="Default split settings")
    for label, value in _split_config_rows(split_config):
        split_defaults.add_row(label, value)

    tables: list[Table] = [overview, scene_defaults, output_defaults, split_defaults]
    if descriptor.dataset_options_model.model_fields:
        option_table = _detail_table(title="Dataset Config")
        for name, default in _loader_option_rows(descriptor):
            option_table.add_row(name, default)
        tables.append(option_table)

    map_defaults = _detail_table(title="Default map settings")
    map_defaults.add_row("Enabled", _format_flag(enabled=map_config is not None))
    if map_config is not None:
        map_defaults.add_row("Extraction", _format_map_extraction(map_config))
        map_defaults.add_row("Min distance", _format_optional_float(map_config.min_distance))
        map_defaults.add_row("Interp distance", _format_optional_float(map_config.interp_distance))
    tables.append(map_defaults)
    return tuple(tables)


def build_split_support_tables(descriptor: DatasetSpec) -> tuple[Table, ...]:
    """Build rich tables describing split support for one dataset."""
    summary = _detail_table(title=f"Split support: {descriptor.name}")
    summary.add_row("Dataset", descriptor.name)
    summary.add_row(
        "Native splits",
        ", ".join(split.value for split in descriptor.native_splits or ()) or "[dim]none[/dim]",
    )
    summary.add_row("Supported split modes", _format_split_modes(descriptor))
    return (summary,)


def _split_detail(plan: ExecutionPlan) -> str:
    split = plan.loader.split
    if split is None or split.strategy == "none":
        return "all data"
    if split.strategy == "native":
        selected = split.read or plan.descriptor.native_splits
        return ", ".join(s.value for s in selected) if selected else "all native splits"
    if split.strategy in {"scene", "source"}:
        return ", ".join(f"{s.value}={w:.2f}" for s, w in split.active())
    if split.strategy == "time":
        return f"{', '.join(f'{s.value}={w:.2f}' for s, w in split.active())}; gap={split.gap}"
    if split.strategy == "shuffled-time":
        return (
            f"{', '.join(f'{s.value}={w:.2f}' for s, w in split.active())}; "
            f"gap={split.gap}; segments={split.segments}"
        )
    return str(split.strategy)


def _detail_table(title: str, *, caption: str | None = None) -> Table:
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


def _format_flag(*, enabled: bool) -> str:
    return "[green]yes[/green]" if enabled else "[dim]no[/dim]"


def _format_split_modes(descriptor: DatasetSpec) -> str:
    modes: list[str] = []
    if descriptor.native_splits:
        modes.append("native")
    if descriptor.split_support.scene:
        modes.append("scene")
    if descriptor.split_support.source:
        modes.append("source")
    if descriptor.split_support.time_block:
        modes.extend(["time", "shuffled-time"])
    return ", ".join(modes) if modes else "[dim]none[/dim]"


def _format_base_window(history_frames: int, future_frames: int, sample_time: float) -> str:
    return f"{history_frames}/{future_frames} @ {1 / sample_time:.1f} Hz"


def _format_effective_window(config: ScenesConfig) -> str:
    history_frames, future_frames, sample_time = effective_scene_window(config)
    return f"{history_frames}/{future_frames} @ {1 / sample_time:.1f} Hz"


def _format_resampling(config: ScenesConfig) -> str:
    spec = config.resample
    if spec is None or (spec.up == 1 and spec.down == 1):
        return "none"
    emit: list[str] = []
    if spec.emit_velocity:
        emit.append("velocity")
    if spec.emit_acceleration:
        emit.append("acceleration")
    extras = "" if not emit else f"; emit={', '.join(emit)}"
    return f"{spec.up}:{spec.down} ({spec.method}){extras}"


def _format_windowing(config: ScenesConfig) -> str:
    window = config.window
    if window is None:
        return "disabled"
    total_frames = config.history_frames + config.future_frames
    return f"{total_frames} frames, step {window.step}"


def _format_screening_rules(config: ScreeningConfig | None) -> str:
    if config is None:
        return "none"
    groups: list[str] = []
    if config.cleanup:
        groups.append("cleanup: " + ", ".join(config.cleanup))
    if config.scene:
        groups.append("scene: " + ", ".join(config.scene))
    if config.agent:
        groups.append("agent: " + ", ".join(config.agent))
    return " | ".join(groups) if groups else "none"


def _format_options(options: BaseModel | object | None) -> str:
    if not isinstance(options, BaseModel):
        return "none"

    options_dict = options.model_dump(exclude_defaults=False)
    if not options_dict:
        return "none"
    return ", ".join(f"{key}={value!r}" for key, value in sorted(options_dict.items()))


def _format_map_extraction(map_config: MapConfig) -> str:
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
    return f"{value:g}" if value is not None else "[dim]none[/dim]"


def _loader_option_rows(descriptor: DatasetSpec) -> list[tuple[str, str]]:
    default_values = descriptor.default_dataset_options().model_dump(exclude_defaults=False)
    rows: list[tuple[str, str]] = []
    for name, field in descriptor.dataset_options_model.model_fields.items():
        value = default_values.get(name)
        default = "[yellow]required[/yellow]" if field.is_required() else repr(value)
        rows.append((name, default))
    return rows


def _run_plan_split_strategy(plan: ExecutionPlan) -> str:
    split = plan.loader.split
    return "none" if split is None else str(split.strategy)


def _split_config_rows(split_config: SplitConfig) -> tuple[tuple[str, str], ...]:
    split_root = split_config.root
    rows: list[tuple[str, str]] = [("Strategy", split_root.strategy)]
    rows.extend(_split_config_detail_rows(split_root))
    return tuple(rows)


def _split_config_detail_rows(split_config: SplitConfigUnion) -> list[tuple[str, str]]:
    match split_config:
        case NoSplitConfig():
            return [("Selection", "all data")]
        case NativeSplitConfig(splits=splits):
            ordered = sorted(splits, key=_split_sort_key)
            selected = ", ".join(split.value for split in ordered)
            return [("Native splits", selected)]
        case SceneSplitConfig(ratio=ratio) | SourceSplitConfig(ratio=ratio):
            return [("Ratio", _format_ratio(ratio.train, ratio.val, ratio.test))]
        case TimeSplitConfig(ratio=ratio, gap=gap):
            return [("Ratio", _format_ratio(ratio.train, ratio.val, ratio.test)), ("Gap", str(gap))]
        case ShuffledTimeSplitConfig(ratio=ratio, gap=gap, segments=segments):
            return [
                ("Ratio", _format_ratio(ratio.train, ratio.val, ratio.test)),
                ("Gap", str(gap)),
                ("Segments", str(segments)),
            ]


def _format_ratio(train: float, val: float, test: float) -> str:
    return f"train={train:g}, val={val:g}, test={test:g}"


def _split_sort_key(split: object) -> str:
    return getattr(split, "value", str(split))
