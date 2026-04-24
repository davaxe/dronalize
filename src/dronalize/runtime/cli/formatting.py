"""Presentation helpers for the optional dronalize CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich import box
from rich.table import Table

from dronalize.config.models import (
    AssignConfig,
    AssignUnion,
    NoAssign,
    PreserveNativeAssign,
    ReadConfig,
    ReadNative,
    ReadUnion,
    SceneAssign,
    ShuffledTimeBlockAssign,
    SourceAssign,
    TimeBlockAssign,
    effective_scene_window,
)
from dronalize.core.categories import EdgeType
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
        ("Read strategy", plan.loader.read.strategy),
        ("Read details", _read_detail(plan)),
        ("Assign strategy", plan.assignment.strategy),
        ("Assign details", _assignment_detail(plan)),
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
        table.add_column("Time assign", justify="center")

    for descriptor in descriptors:
        if not details:
            table.add_row(descriptor.name)
            continue

        cfg = descriptor.default_config.scenes
        window = _format_base_window(cfg.history_frames, cfg.future_frames, cfg.sample_time)
        has_map = "[green]yes[/green]" if descriptor.has_map else "[dim]no[/dim]"
        native_splits = (
            ", ".join(split.value for split in descriptor.supported_native_splits or ())
            or "[dim]none[/dim]"
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
    read_config = descriptor.default_config.read
    assign_config = descriptor.default_config.assign

    overview = _detail_table(title=f"Dataset inspect: {descriptor.name}")
    overview.add_row("Dataset", descriptor.name)
    overview.add_row(
        "Native schema",
        f"{descriptor.native_schema.name} ({descriptor.native_schema.feature_dim} features)",
    )
    overview.add_row("Schema fields", ", ".join(descriptor.native_schema.semantic_fields()))
    overview.add_row(
        "Native splits",
        ", ".join(split.value for split in descriptor.supported_native_splits or ())
        or "[dim]none[/dim]",
    )
    overview.add_row("Read modes", _format_read_modes(descriptor))
    overview.add_row("Assignment modes", _format_assignment_modes(descriptor))
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

    read_defaults = _detail_table(title="Default read settings")
    for label, value in _read_config_rows(read_config):
        read_defaults.add_row(label, value)

    assign_defaults = _detail_table(title="Default assignment settings")
    for label, value in _assign_config_rows(assign_config):
        assign_defaults.add_row(label, value)

    tables: list[Table] = [
        overview,
        scene_defaults,
        output_defaults,
        read_defaults,
        assign_defaults,
    ]
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
        map_defaults.add_row("Edge types", _format_map_edge_types(map_config))
    tables.append(map_defaults)
    return tuple(tables)


def build_split_support_tables(descriptor: DatasetSpec) -> tuple[Table, ...]:
    """Build rich tables describing read and assignment support for one dataset."""
    summary = _detail_table(title=f"Split support: {descriptor.name}")
    summary.add_row("Dataset", descriptor.name)
    summary.add_row(
        "Native splits",
        ", ".join(split.value for split in descriptor.supported_native_splits or ())
        or "[dim]none[/dim]",
    )
    summary.add_row("Read modes", _format_read_modes(descriptor))
    summary.add_row("Assignment modes", _format_assignment_modes(descriptor))
    return (summary,)


def _read_detail(plan: ExecutionPlan) -> str:
    native_splits = plan.loader.read.native_splits
    if native_splits is None:
        return "all sources"
    return ", ".join(split.value for split in native_splits)


def _assignment_detail(plan: ExecutionPlan) -> str:
    assignment = plan.assignment
    if assignment.strategy == "none":
        return "unsplit output"
    if assignment.strategy == "preserve-native":
        selected = plan.loader.read.native_splits
        return ", ".join(s.value for s in selected) if selected else "native labels"
    if assignment.strategy in {"scene", "source"}:
        return ", ".join(f"{s.value}={w:.2f}" for s, w in assignment.active())
    if assignment.strategy == "time":
        return (
            f"{', '.join(f'{s.value}={w:.2f}' for s, w in assignment.active())}; "
            f"gap={assignment.gap}"
        )
    if assignment.strategy == "shuffled-time":
        return (
            f"{', '.join(f'{s.value}={w:.2f}' for s, w in assignment.active())}; "
            f"gap={assignment.gap}; segments={assignment.segments}"
        )
    return assignment.strategy


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


def _format_read_modes(descriptor: DatasetSpec) -> str:
    modes = ["all"]
    if descriptor.supported_native_splits:
        modes.append("native")
    return ", ".join(modes)


def _format_assignment_modes(descriptor: DatasetSpec) -> str:
    modes = ["none"]
    if descriptor.supported_native_splits:
        modes.append("preserve-native")
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
            return f"scene extent (padding={extraction.padding:g}, shape={extraction.shape})"
        case "circle":
            return f"circle (radius={extraction.radius:g})"
        case "trajectory_buffer":
            return f"trajectory buffer (radius={extraction.radius:g})"
        case "bounding_box":
            return f"bounding box ({extraction.width:g} x {extraction.height:g})"


def _format_map_edge_types(map_config: MapConfig) -> str:
    edge_types = map_config.edge_types
    if edge_types is None:
        return "all"

    parts: list[str] = []
    if edge_types.include is not None:
        parts.append("include=" + ",".join(_format_edge_type_value(edge_type) for edge_type in edge_types.include))
    if edge_types.exclude:
        parts.append("exclude=" + ",".join(_format_edge_type_value(edge_type) for edge_type in edge_types.exclude))
    if edge_types.remap:
        parts.append(
            "remap="
            + ",".join(
                f"{_format_edge_type_value(source)}->{_format_edge_type_value(target)}"
                for source, target in edge_types.remap.items()
            )
        )
    return " | ".join(parts) if parts else "all"


def _format_edge_type_value(value: object) -> str:
    try:
        return EdgeType.from_value(value).name
    except (TypeError, ValueError):
        return str(value)


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


def _read_config_rows(read_config: ReadConfig) -> tuple[tuple[str, str], ...]:
    read_root = read_config.root
    rows: list[tuple[str, str]] = [("Strategy", read_root.strategy)]
    rows.extend(_read_config_detail_rows(read_root))
    return tuple(rows)


def _read_config_detail_rows(read_config: ReadUnion) -> list[tuple[str, str]]:
    match read_config:
        case ReadNative(splits=splits):
            if splits is None:
                return [("Native splits", "all native splits")]
            selected = ", ".join(split.value for split in splits)
            return [("Native splits", selected)]
        case _:
            return [("Selection", "all available inputs")]


def _assign_config_rows(assign_config: AssignConfig) -> tuple[tuple[str, str], ...]:
    assign_root = assign_config.root
    rows: list[tuple[str, str]] = [("Strategy", assign_root.strategy)]
    rows.extend(_assign_config_detail_rows(assign_root))
    return tuple(rows)


def _assign_config_detail_rows(assign_config: AssignUnion) -> list[tuple[str, str]]:
    match assign_config:
        case NoAssign():
            return [("Selection", "unsplit output")]
        case PreserveNativeAssign():
            return [("Selection", "preserve native labels")]
        case SceneAssign(ratio=ratio) | SourceAssign(ratio=ratio):
            return [("Ratio", _format_ratio(ratio.train, ratio.val, ratio.test))]
        case TimeBlockAssign(ratio=ratio, gap=gap):
            return [("Ratio", _format_ratio(ratio.train, ratio.val, ratio.test)), ("Gap", str(gap))]
        case ShuffledTimeBlockAssign(ratio=ratio, gap=gap, segments=segments):
            return [
                ("Ratio", _format_ratio(ratio.train, ratio.val, ratio.test)),
                ("Gap", str(gap)),
                ("Segments", str(segments)),
            ]


def _format_ratio(train: float, val: float, test: float) -> str:
    return f"train={train:g}, val={val:g}, test={test:g}"
