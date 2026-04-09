"""Human-readable summaries for prepared runtime processing plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.processing.loading.splits import (
    NativeSplitStrategy,
    NoSplitStrategy,
    ShuffledTimeBlockStrategy,
    TimeBlockStrategy,
)
from dronalize.runtime.models import ProcessingSummary, SummarySection

if TYPE_CHECKING:
    from collections.abc import Iterable

    from dronalize.core.categories import DatasetSplit
    from dronalize.processing.loading.base import LoaderOptions
    from dronalize.processing.loading.config import LoaderConfig
    from dronalize.processing.maps.config import MapConfig
    from dronalize.runtime.models import DatasetPlan


def summarize_plan(plan: DatasetPlan) -> ProcessingSummary:
    """Return a structured display summary for a dataset plan."""
    loader, export = plan.config.loader, plan.config.export

    sections = (
        SummarySection(
            title="Overview",
            rows=(
                ("Dataset", plan.descriptor.name),
                ("Input directory", str(plan.data_root)),
                ("Output directory", str(plan.output_dir)),
                ("Storage backend", plan.storage_backend.value),
                (
                    "Trajectory schema",
                    f"{export.trajectory_schema.name} ({export.feature_dim} features)",
                ),
            ),
        ),
        SummarySection(
            title="Transformations",
            rows=_non_empty_rows(
                ("Source window", _source_window_summary(loader)),
                ("Effective window", _window_summary(loader)),
                ("Resampling", _resampling_summary(loader)),
                ("Filtering", _filter_summary(loader)),
                ("Map", _map_summary(plan.config.map)),
                ("Loader options", _format_options(plan.config.loader_options))
                if _has_loader_options(plan.config.loader_options)
                else None,
            ),
        ),
        SummarySection(
            title="Execution",
            rows=_non_empty_rows(
                ("Execution", _execution_summary(plan)),
                ("Scene limit", str(plan.limit)) if plan.limit is not None else None,
                ("Random seed", str(seed))
                if (seed := plan.split_request.seed) is not None
                else None,
            ),
        ),
        SummarySection(title="Splits", rows=_split_summary_rows(plan)),
    )

    return ProcessingSummary(
        title="Processing Plan", sections=tuple(section for section in sections if section.rows)
    )


def _execution_summary(plan: DatasetPlan) -> str:
    if not plan.parallel:
        return "sequential"
    jobs = plan.config.execution.jobs
    if jobs is None:
        return "parallel (auto workers)"
    return f"parallel ({jobs} worker{'s' if jobs != 1 else ''})"


def _source_window_summary(loader: LoaderConfig) -> str:
    return f"{loader.input_len}/{loader.output_len} @ {1 / loader.sample_time:.1f} Hz"


def _window_summary(loader: LoaderConfig) -> str:
    return (
        f"{loader.resampled_input_len}/{loader.resampled_output_len}"
        f" @ {1 / loader.post_sample_time:.1f} Hz"
    )


def _resampling_summary(loader: LoaderConfig) -> str:
    spec = loader.resampling
    if spec is None or spec.no_resampling:
        return "none"
    return f"{spec.up}:{spec.down} ({spec.method.value})"


def _filter_summary(loader: LoaderConfig) -> str:
    scene_filter = loader.filter
    if scene_filter is None:
        return "none"

    parts = [
        f"{label}: {', '.join(rule.name() for rule in rules)}"
        for label, rules in (
            ("cleanup", scene_filter.cleanup_rules),
            ("scene", scene_filter.scene_rules),
            ("agent", scene_filter.agent_rules),
        )
        if rules
    ]
    return " | ".join(parts) if parts else "none"


def _map_summary(map_config: MapConfig | None) -> str:
    if map_config is None:
        return "disabled"

    details = [_map_extraction_summary(map_config)]
    if map_config.min_distance is not None:
        details.append(f"min_distance={map_config.min_distance:g}")
    if map_config.interp_distance is not None:
        details.append(f"interp_distance={map_config.interp_distance:g}")
    return f"enabled ({', '.join(details)})"


def _map_extraction_summary(map_config: MapConfig) -> str:
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


def _split_summary_rows(plan: DatasetPlan) -> tuple[tuple[str, str], ...]:
    if isinstance(plan.split_request.strategy, NativeSplitStrategy):
        splits = plan.loader_splits()
        rows = [("Split strategy", "native")]
        if splits:
            rows.append(("Read splits", _format_splits(splits)))
        return tuple(rows)
    if isinstance(plan.split_request.strategy, NoSplitStrategy):
        return ()

    rows = [("Split strategy", plan.split_request.strategy_name)]
    if (details := _split_details_summary(plan)) is not None:
        rows.append(("Time split settings", details))
    rows.append(("Output ratio", _format_weighted_splits(plan.split_request.active())))
    return tuple(rows)


def _split_details_summary(plan: DatasetPlan) -> str | None:
    strategy = plan.split_request.strategy
    if isinstance(strategy, TimeBlockStrategy):
        return f"gap={strategy.gap} frames"
    if isinstance(strategy, ShuffledTimeBlockStrategy):
        return f"segments={strategy.segments}, gap={strategy.gap} frames"
    return None


def _format_splits(splits: Iterable[DatasetSplit]) -> str:
    return ", ".join(split.value for split in splits)


def _format_weighted_splits(groups: Iterable[tuple[DatasetSplit, float]]) -> str:
    total_weight = sum(weight for _, weight in groups)
    if total_weight <= 0:
        return "single output directory"

    formatted = [f"{split.value} ({weight / total_weight:.0%})" for split, weight in groups]
    return ", ".join(formatted)


def _has_loader_options(options: LoaderOptions) -> bool:
    return bool(options.model_dump(exclude_defaults=True))


def _format_options(options: LoaderOptions) -> str:
    values = options.model_dump(exclude_defaults=True)
    return ", ".join(f"{key}={value!r}" for key, value in sorted(values.items()))


def _non_empty_rows(*rows: tuple[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(row for row in rows if row is not None)
