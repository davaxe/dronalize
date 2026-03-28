from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Any

import altair as alt
import polars as pl

from dronalize.plot import plot_trajectories, plot_trajectories_on_map

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from dronalize.core.scene import Scene
    from dronalize.processing.ingest.loader import SceneLoader


def debug_visualize_scenes(
    source: SceneLoader,
    *,
    max_scenes: int = 1,
    skip_scenes: int = 0,
    step: int = 1,
    group_by: str = "id",
    n_groups: int | None = None,
    group_sample_seed: int | None = None,
    highlight_frame: int | Sequence[int] | None = None,
    include_map: bool = True,
    include_map_nodes: bool = False,
    show: bool = True,
    title_prefix: str | None = None,
    trajectory_kwargs: dict[str, Any] | None = None,
    overlay_kwargs: dict[str, Any] | None = None,
) -> list[alt.TopLevelMixin]:
    _ = alt.renderers.enable("browser")

    trajectory_kwargs = {} if trajectory_kwargs is None else dict(trajectory_kwargs)
    overlay_kwargs = {} if overlay_kwargs is None else dict(overlay_kwargs)
    charts: list[alt.TopLevelMixin] = []

    scenes: Iterator[Scene] = iter(source.scenes())
    for scene in islice(scenes, skip_scenes, skip_scenes + max_scenes * step, step):
        title_parts: list[str] = []
        if title_prefix:
            title_parts.append(title_prefix)
        title_parts.append(f"scene {scene.scene_number}")
        if scene.map_key is not None:
            title_parts.append(f"map={scene.map_key}")
        if scene.split_assignment:
            title_parts.append(f"split assignment={scene.split_assignment.value}")
        title = " | ".join(title_parts)

        map_graph = None
        if include_map:
            map_graph = scene.resolve_map()

        if map_graph is not None:
            chart = plot_trajectories_on_map(
                scene.frame,
                map_graph,
                group_by=group_by,
                n_groups=n_groups,
                group_sample_seed=group_sample_seed,
                highlight_frame=highlight_frame,
                include_map_nodes=include_map_nodes,
                title=title,
                **overlay_kwargs,
            )
        else:
            chart = plot_trajectories(
                scene.frame,
                group_by=group_by,
                n_groups=n_groups,
                group_sample_seed=group_sample_seed,
                highlight_frame=highlight_frame,
                title=title,
                **trajectory_kwargs,
            )

        if show:
            chart.show()
        charts.append(chart)
    return charts


def _debug_block_split_snapshot(
    lf: pl.LazyFrame,
    *,
    frame_column: str,
    group_columns: Sequence[str],
    split_column: str,
    partition_column: str,
    original_frame_column: str,
) -> None:
    """Write a sorted debug snapshot and print split/partition summaries."""
    df = lf.collect().sort(original_frame_column)

    print(f"[block split debug] columns: {', '.join(df.columns)}")

    if split_column not in df.columns or original_frame_column not in df.columns:
        print("[block split debug] missing required debug columns; skipping summary")
        return

    frame_key_columns = [*group_columns, original_frame_column]
    total_frames = int(df.select(pl.struct(*frame_key_columns).n_unique()).item())
    split_summary_exprs: list[pl.Expr] = [
        pl.len().alias("rows"),
        pl.struct(*frame_key_columns).n_unique().alias("frames"),
        pl.col(original_frame_column).min().alias("original_frame_min"),
        pl.col(original_frame_column).max().alias("original_frame_max"),
        pl.col(frame_column).min().alias("frame_min"),
        pl.col(frame_column).max().alias("frame_max"),
    ]
    if partition_column in df.columns:
        split_summary_exprs.append(pl.col(partition_column).n_unique().alias("partitions"))

    split_summary = df.group_by(split_column).agg(*split_summary_exprs).sort("original_frame_min")
    split_summary = split_summary.with_columns(
        (pl.col("rows") / df.height * 100).round(2).alias("pct_total_rows"),
        (pl.col("frames") / total_frames * 100).round(2).alias("pct_total_frames"),
    )
    print("[block split debug] split summary:")
    print(split_summary)

    if partition_column not in df.columns:
        return

    partition_summary = (
        df
        .group_by(split_column, partition_column)
        .agg(
            pl.len().alias("rows"),
            pl.col(original_frame_column).min().alias("original_frame_min"),
            pl.col(original_frame_column).max().alias("original_frame_max"),
            pl.col(frame_column).min().alias("frame_min"),
            pl.col(frame_column).max().alias("frame_max"),
        )
        .sort("original_frame_min")
    )
    print("[block split debug] partition summary:")
    print(partition_summary)
