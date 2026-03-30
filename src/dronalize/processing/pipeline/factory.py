"""Pipeline assembly helpers for trajectory processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import dronalize.processing.pipeline.transforms as tr
from dronalize.processing.pipeline.builder import TrajectoryPipelineBuilder
from dronalize.processing.pipeline.pipeline import Pipeline
from dronalize.processing.pipeline.splitting import apply_split_partition

if TYPE_CHECKING:
    from dronalize.processing.filters import Filter
    from dronalize.processing.ingest.config import WindowParams
    from dronalize.processing.pipeline.functional.resample import ResampleSpec
    from dronalize.processing.pipeline.spec import TrajectorySpec


def trajectory_pipeline(spec: TrajectorySpec) -> Pipeline:
    """Build a trajectory-processing pipeline from a declarative spec."""
    builder = TrajectoryPipelineBuilder(spec)
    if spec.extension is not None:
        spec.extension.extend(builder)

    scene_id_column = builder.scene_id_column if builder.uses_scene_id() else None
    pipeline = apply_split_partition(
        Pipeline(),
        split_request=spec.split_request,
        time_column=spec.columns.frame,
        group_by=spec.window_by,
    )
    pipeline = pipeline.compose(builder.pre_window)
    pipeline = _apply_window(
        pipeline,
        frame_column=spec.columns.frame,
        window_group_columns=builder.window_group_columns(),
        window_spec=spec.config.window,
    )
    pipeline = pipeline.compose(builder.post_window)
    pipeline = _attach_scene_id(
        pipeline, scene_key_columns=builder.scene_key_columns(), scene_id_column=scene_id_column
    )
    pipeline = _apply_filter(
        pipeline,
        filter_spec=spec.config.filter,
        scene_id_column=scene_id_column,
        agent_id_column=spec.columns.agent_id,
        frame_column=spec.columns.frame,
        category_column=spec.columns.category,
    )
    pipeline = pipeline.compose(builder.post_filter)
    return _apply_resample_and_yield(
        pipeline,
        frame_column=spec.columns.frame,
        agent_id_column=spec.columns.agent_id,
        scene_id_column=scene_id_column,
        resampling=spec.config.resampling,
    )


def _attach_scene_id(
    pipeline: Pipeline, *, scene_key_columns: list[str], scene_id_column: str | None
) -> Pipeline:
    if scene_id_column is None:
        return pipeline

    def _with_scene_id(df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df
            .with_row_index("_scene_row_order")
            .with_columns(
                pl.col("_scene_row_order").min().over(scene_key_columns).alias("_scene_first_row")
            )
            .with_columns(
                (pl.col("_scene_first_row").rank("dense") - 1)
                .cast(pl.UInt32)
                .alias(scene_id_column)
            )
            .drop("_scene_row_order", "_scene_first_row")
        )

    return pipeline.then(_with_scene_id, name="attach_scene_id")


def _apply_window(
    pipeline: Pipeline,
    *,
    frame_column: str,
    window_group_columns: list[str],
    window_spec: WindowParams | None,
) -> Pipeline:
    if window_spec is None:
        return pipeline
    return pipeline.then(
        tr.window(
            window_spec.window_size,
            window_spec.step_size,
            group_by=window_group_columns or None,
            sliding_col=frame_column,
        )
    )


def _apply_filter(
    pipeline: Pipeline,
    *,
    filter_spec: Filter | None,
    scene_id_column: str | None,
    agent_id_column: str,
    frame_column: str,
    category_column: str,
) -> Pipeline:
    if filter_spec is not None:
        return pipeline.then(
            tr.filter_scene(
                filter_spec,
                group_by=scene_id_column,
                agent_id=agent_id_column,
                frame_column=frame_column,
                category_column=category_column,
            )
        )
    return pipeline


def _apply_resample_and_yield(
    pipeline: Pipeline,
    *,
    frame_column: str,
    agent_id_column: str,
    scene_id_column: str | None,
    resampling: ResampleSpec | None,
) -> Pipeline:
    pipeline = pipeline.then(
        tr.resample(
            spec=resampling,
            frame_column=frame_column,
            group_by=(
                [scene_id_column, agent_id_column]
                if scene_id_column is not None
                else [agent_id_column]
            ),
        )
    )
    if scene_id_column is not None:
        pipeline = pipeline.then_flat_map(tr.group_by_yield(scene_id_column))
    return pipeline
