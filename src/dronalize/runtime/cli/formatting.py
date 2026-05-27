"""Presentation helpers for the optional dronalize CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich import box
from rich.table import Table

from dronalize.config.models.map import (
    BoundingBoxExtraction,
    CircularExtraction,
    FullMapExtraction,
    SceneExtentExtraction,
    TrajectoryBufferExtraction,
)
from dronalize.config.models.scenes import effective_scene_window
from dronalize.config.models.split import NoAssign, PreserveNativeAssign, ReadAll, ReadNative
from dronalize.core.categories import DatasetSplit, EdgeType
from dronalize.core.scene import get_trajectory_schema
from dronalize.io.base import storage_backend_name

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dronalize.config.models.map import MapConfig
    from dronalize.config.models.scenes import ScenesConfig
    from dronalize.config.models.screening import ScreeningConfig
    from dronalize.config.models.split import (
        AssignConfig,
        AssignUnion,
        ReadConfig,
        ReadUnion,
        SplitWeights,
    )
    from dronalize.datasets.registry import (
        DatasetDescriptor,
        DatasetTemporalSupport,
        DatasetWindowingSupport,
    )
    from dronalize.processing.models import ReadSelection, SplitAssignmentPlan
    from dronalize.runtime.types import ExecutionPlan

PLAN_NOTICE = (
    "\n[bold yellow]This is the processing plan. No changes have been made yet.[/bold yellow]"
)

Row = tuple[str, str]
Section = tuple[str, tuple[Row, ...]]
_SPLIT_ORDER = {DatasetSplit.TRAIN: 0, DatasetSplit.VAL: 1, DatasetSplit.TEST: 2}


def build_processing_summary_table(plan: ExecutionPlan) -> Table:
    """Build a rich table summarizing one resolved processing plan."""
    return _detail_table(title=f"Processing plan: {plan.dataset}", rows=summarize_plan(plan))


def summarize_plan(plan: ExecutionPlan) -> tuple[Row, ...]:
    """Return label/value rows for a resolved plan summary."""
    output_config = plan.output.config
    return (
        ("Dataset", plan.dataset),
        ("Input", str(plan.data_root)),
        ("Output", str(plan.output_dir)),
        ("Backend", storage_backend_name(plan.storage_backend)),
        ("Workers", str(plan.runtime.jobs)),
        ("Limit", "none" if plan.limit is None else str(plan.limit)),
        ("Schema", get_trajectory_schema(output_config.trajectory_schema).name),
        ("Map", _format_flag(enabled=plan.map is not None)),
        ("Read", _inline_config_rows(_read_request_rows(plan.loader.read))),
        (
            "Assign",
            _inline_config_rows(_assignment_request_rows(plan.assignment, plan.loader.read)),
        ),
        ("Options", _format_options(plan.loader.loader_options)),
    )


def build_available_datasets_table(
    descriptors: Sequence[DatasetDescriptor], *, details: bool
) -> Table:
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
        table.add_column("Split assign")

    for descriptor in descriptors:
        if not details:
            table.add_row(descriptor.name)
            continue

        table.add_row(
            descriptor.name,
            _format_window_config(descriptor.default_config.scenes),
            _format_flag(enabled=descriptor.feature_support.map),
            _format_splits(descriptor.supported_native_splits, empty="[dim]none[/dim]"),
            _format_assignment_modes(descriptor, compact=True),
        )
    return table


def build_dataset_inspect_tables(descriptor: DatasetDescriptor) -> tuple[Table, ...]:
    """Build one or more inspection tables for a dataset descriptor."""
    return (
        _section_table(
            title=f"Dataset inspect: {descriptor.name}",
            sections=_dataset_inspect_sections(descriptor),
        ),
    )


def build_split_support_tables(descriptor: DatasetDescriptor) -> tuple[Table, ...]:
    """Build rich tables describing read and assignment support for one dataset."""
    rows: tuple[Row, ...] = (
        ("Dataset", descriptor.name),
        (
            "Native splits",
            _format_splits(descriptor.supported_native_splits, empty="[dim]none[/dim]"),
        ),
        ("Read modes", "all, native" if descriptor.supported_native_splits else "all"),
        ("Assignment modes", _format_assignment_modes(descriptor)),
    )
    return (_detail_table(title=f"Split support: {descriptor.name}", rows=rows),)


def _dataset_inspect_sections(descriptor: DatasetDescriptor) -> tuple[Section, ...]:
    default_config = descriptor.default_config
    scenes = default_config.scenes
    map_config = default_config.map if descriptor.feature_support.map else None
    output_config = default_config.output
    schema = descriptor.native_schema
    effective_horizon, _, effective_sample_time = effective_scene_window(scenes)

    sections: list[Section] = [
        (
            "Dataset",
            (
                ("Name", descriptor.name),
                ("Native schema", f"{schema.name} ({schema.feature_dim} features)"),
                ("Schema fields", ", ".join(schema.semantic_fields())),
                (
                    "Native splits",
                    _format_splits(descriptor.supported_native_splits, empty="[dim]none[/dim]"),
                ),
                ("Read modes", "all, native" if descriptor.supported_native_splits else "all"),
                ("Assignment modes", _format_assignment_modes(descriptor)),
                ("Map", _format_flag(enabled=descriptor.feature_support.map)),
                (
                    "Lane-change sampling",
                    _format_flag(enabled=descriptor.feature_support.lane_change_sampling),
                ),
            ),
        ),
        (
            "Scenes",
            (
                ("Configured horizon", _format_window_config(scenes)),
                ("Effective horizon", _format_window(effective_horizon, effective_sample_time)),
                ("Resampling", _format_resampling(scenes)),
                ("Sliding windows", _format_windowing(scenes)),
                ("Screening rules", _format_screening_rules(default_config.screening)),
            ),
        ),
        ("Temporal", _temporal_support_rows(descriptor.temporal_support, scenes)),
        (
            "Output",
            (
                ("Schema", get_trajectory_schema(output_config.trajectory_schema).name),
                ("Precision", output_config.precision),
                ("Recenter positions", _format_flag(enabled=output_config.recenter_positions)),
            ),
        ),
        ("Read", _read_config_rows(default_config.read)),
        ("Assignment", _assign_config_rows(default_config.assign)),
        ("Map", _map_config_rows(map_config)),
    ]
    if descriptor.loader_options_model.model_fields:
        sections.append(("Loader options", tuple(_loader_option_rows(descriptor))))
    return tuple(sections)


def _detail_table(title: str, rows: Iterable[Row]) -> Table:
    table = Table(title=title, show_header=False, box=box.MINIMAL_DOUBLE_HEAD, title_justify="left")
    table.add_column(style="bright_cyan", justify="left", no_wrap=True)
    table.add_column(style="bright_magenta")
    for label, value in rows:
        table.add_row(label, value)
    return table


def _section_table(title: str, sections: Iterable[Section]) -> Table:
    table = Table(
        title=title,
        box=box.MINIMAL_DOUBLE_HEAD,
        title_justify="left",
        header_style="bold",
        row_styles=["", "dim"],
    )
    table.add_column("Section", style="bright_cyan", no_wrap=True)
    table.add_column("Setting", style="bright_blue", no_wrap=True)
    table.add_column("Value", style="bright_magenta")
    for section, rows in sections:
        row_count = len(rows)
        for index, (label, value) in enumerate(rows):
            table.add_row(
                section if index == 0 else "", label, value, end_section=index == row_count - 1
            )
    return table


def _format_flag(*, enabled: bool) -> str:
    return "[green]yes[/green]" if enabled else "[dim]no[/dim]"


def _format_assignment_modes(descriptor: DatasetDescriptor, *, compact: bool = False) -> str:
    modes = [] if compact else ["none"]
    if descriptor.supported_native_splits:
        modes.append("native" if compact else "preserve-native")
    if descriptor.split_support.scene:
        modes.append("scene")
    if descriptor.split_support.source:
        modes.append("source")
    if descriptor.split_support.time_block:
        modes.extend(("time", "shuffled-time"))
    return ", ".join(modes) if modes else "none"


def _format_window_config(config: ScenesConfig) -> str:
    return _format_window(config.horizon_frames, config.sample_time)


def _format_window(horizon_frames: int, sample_time: float) -> str:
    return f"{horizon_frames} @ {1 / sample_time:.1f} Hz"


def _format_resampling(config: ScenesConfig) -> str:
    spec = config.resample
    if spec is None or (spec.up == 1 and spec.down == 1):
        return "none"
    emitted = ", ".join(
        name
        for enabled, name in (
            (spec.emit_velocity, "velocity"),
            (spec.emit_acceleration, "acceleration"),
        )
        if enabled
    )
    extras = "" if not emitted else f"; emit={emitted}"
    return f"{spec.up}:{spec.down} ({spec.method}){extras}"


def _format_windowing(config: ScenesConfig) -> str:
    window = config.window
    if window is None:
        return "disabled"
    return f"enabled, {window.policy}, step {window.step}"


def _temporal_support_rows(
    support: DatasetTemporalSupport | None, scenes: ScenesConfig
) -> tuple[Row, ...]:
    if support is None:
        return (
            ("Source unit", "[dim]unknown[/dim]"),
            ("Source bounds", "[dim]unknown[/dim]"),
            ("Configured horizon fits", "[dim]unknown[/dim]"),
        )

    windowing = support.windowing
    rows: list[Row] = [
        ("Source unit", support.source_unit),
        ("Source bounds", _format_frame_bounds(support)),
        ("Configured horizon fits", _format_horizon_fit(scenes.horizon_frames, support)),
        ("Windowing default", _format_flag(enabled=windowing.enabled_by_default)),
        ("Supported policies", _format_window_policies(windowing)),
        ("Window validation", windowing.validation),
    ]
    if windowing.max_window_frames is not None:
        rows.append(("Max supported horizon", f"{windowing.max_window_frames} frames"))
    return tuple(rows)


def _format_frame_bounds(support: DatasetTemporalSupport) -> str:
    bounds = support.source_frame_bounds
    if bounds.min_frames is None and bounds.max_frames is None:
        return f"unknown ({bounds.confidence})"

    pieces: list[str] = []
    if bounds.min_frames is not None or bounds.max_frames is not None:
        lo = "?" if bounds.min_frames is None else str(bounds.min_frames)
        hi = "?" if bounds.max_frames is None else str(bounds.max_frames)
        pieces.append(f"{lo}-{hi} frames")
    pieces.append(bounds.confidence)
    return ", ".join(pieces)


def _format_horizon_fit(horizon_frames: int, support: DatasetTemporalSupport) -> str:
    max_frames = support.windowing.max_window_frames
    if max_frames is None:
        max_frames = support.source_frame_bounds.max_frames
    if max_frames is None:
        return "[dim]unknown[/dim]"
    if horizon_frames <= max_frames:
        return f"[green]yes[/green] ({horizon_frames} <= {max_frames} frames)"
    return f"[red]no[/red] ({horizon_frames} > {max_frames} frames)"


def _format_window_policies(windowing: DatasetWindowingSupport) -> str:
    policies = list(windowing.supported_policies)
    if windowing.default_policy in policies:
        policies.remove(windowing.default_policy)
        ordered = [f"{windowing.default_policy} (default)", *policies]
    else:
        ordered = [f"{windowing.default_policy} (default, unsupported)", *policies]
    return ", ".join(ordered)


def _format_screening_rules(config: ScreeningConfig | None) -> str:
    if config is None:
        return "none"
    groups = [
        f"{label}: {', '.join(rules)}"
        for label, rules in (
            ("cleanup", config.cleanup),
            ("scene", config.scene),
            ("agent", config.agent),
        )
        if rules
    ]
    return " | ".join(groups) if groups else "none"


def _format_options(options: BaseModel | object | None) -> str:
    if not isinstance(options, BaseModel):
        return "none"

    options_dict = options.model_dump(exclude_defaults=False)
    if not options_dict:
        return "none"
    return ", ".join(f"{key}={value!r}" for key, value in sorted(options_dict.items()))


def _format_map_extraction(map_config: MapConfig) -> str:
    match map_config.extraction:
        case FullMapExtraction():
            return "full map"
        case SceneExtentExtraction(padding=padding, shape=shape):
            return f"scene extent (padding={padding:g}, shape={shape})"
        case CircularExtraction(radius=radius):
            return f"circle (radius={radius:g})"
        case TrajectoryBufferExtraction(radius=radius):
            return f"trajectory buffer (radius={radius:g})"
        case BoundingBoxExtraction(width=width, height=height):
            return f"bounding box ({width:g} x {height:g})"


def _format_map_edge_types(map_config: MapConfig) -> str:
    edge_types = map_config.edge_types
    if edge_types is None:
        return "all"

    parts: list[str] = []
    if edge_types.include is not None:
        parts.append(
            "include="
            + ",".join(sorted(_format_edge_type_value(value) for value in edge_types.include))
        )
    if edge_types.exclude:
        parts.append(
            "exclude="
            + ",".join(sorted(_format_edge_type_value(value) for value in edge_types.exclude))
        )
    if edge_types.remap:
        remaps = sorted(
            (_format_edge_type_value(source), _format_edge_type_value(target))
            for source, target in edge_types.remap.items()
        )
        parts.append("remap=" + ",".join(f"{source}->{target}" for source, target in remaps))
    return " | ".join(parts) if parts else "all"


def _format_edge_type_value(value: str | int | EdgeType) -> str:
    try:
        return EdgeType.from_value(value).name
    except (TypeError, ValueError):
        return str(value)


def _format_optional_float(value: float | None) -> str:
    return f"{value:g}" if value is not None else "[dim]none[/dim]"


def _loader_option_rows(descriptor: DatasetDescriptor) -> tuple[Row, ...]:
    return tuple(
        (
            name,
            "[yellow]required[/yellow]"
            if field.is_required()
            else repr(field.get_default(call_default_factory=True)),
        )
        for name, field in descriptor.loader_options_model.model_fields.items()
    )


def _read_request_rows(read: ReadSelection) -> tuple[Row, ...]:
    return _read_rows(read.config, native_splits=read.native_splits)


def _read_config_rows(read_config: ReadConfig) -> tuple[Row, ...]:
    return _read_rows(read_config.root)


def _read_rows(
    read_config: ReadUnion, *, native_splits: Sequence[DatasetSplit] | None = None
) -> tuple[Row, ...]:
    rows: list[Row] = [("Strategy", read_config.strategy)]
    match read_config:
        case ReadNative(splits=splits):
            selected = native_splits if native_splits is not None else splits
            rows.append(("Native splits", _format_splits(selected, empty="all native splits")))
        case ReadAll():
            rows.append(("Selection", "all available inputs"))
    return tuple(rows)


def _assignment_request_rows(
    assignment: SplitAssignmentPlan, read: ReadSelection | None = None
) -> tuple[Row, ...]:
    return _assignment_rows(
        assignment.config, native_splits=None if read is None else read.native_splits
    )


def _assign_config_rows(assign_config: AssignConfig) -> tuple[Row, ...]:
    return _assignment_rows(assign_config.root)


def _assignment_rows(
    assign_config: AssignUnion, *, native_splits: Sequence[DatasetSplit] | None = None
) -> tuple[Row, ...]:
    rows: list[Row] = [("Strategy", assign_config.strategy)]
    if isinstance(assign_config, NoAssign):
        rows.append(("Selection", "unsplit output"))
        return tuple(rows)

    if isinstance(assign_config, PreserveNativeAssign):
        detail = (
            _format_splits(native_splits, empty="native labels")
            if native_splits is not None
            else "preserve native labels"
        )
        rows.append(("Selection", detail))
        return tuple(rows)

    rows.append(("Ratio", _format_ratio(assign_config.ratio)))
    gap = getattr(assign_config, "gap", None)
    segments = getattr(assign_config, "segments", None)
    if gap is not None:
        rows.append(("Gap", str(gap)))
    if segments is not None:
        rows.append(("Segments", str(segments)))
    return tuple(rows)


def _map_config_rows(map_config: MapConfig | None) -> tuple[Row, ...]:
    if map_config is None:
        return (("Enabled", _format_flag(enabled=False)),)
    return (
        ("Enabled", _format_flag(enabled=True)),
        ("Extraction", _format_map_extraction(map_config)),
        ("Min distance", _format_optional_float(map_config.min_distance)),
        ("Interp distance", _format_optional_float(map_config.interpolation_distance)),
        ("Edge types", _format_map_edge_types(map_config)),
    )


def _inline_config_rows(rows: Sequence[Row]) -> str:
    _, strategy = rows[0]
    details = rows[1:]
    if not details:
        return strategy
    return f"{strategy}; " + "; ".join(
        value if label == "Selection" else f"{label.lower()}={value}" for label, value in details
    )


def _format_splits(
    splits: Sequence[DatasetSplit] | frozenset[DatasetSplit] | None, *, empty: str
) -> str:
    return (
        empty
        if not splits
        else ", ".join(split.value for split in sorted(splits, key=lambda s: _SPLIT_ORDER[s]))
    )


def _format_ratio(ratio: SplitWeights) -> str:
    return f"train={ratio.train:g}, val={ratio.val:g}, test={ratio.test:g}"
