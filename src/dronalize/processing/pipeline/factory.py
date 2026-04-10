"""Pipeline assebly helpers fr trajectory processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import dronalize.processing.pipeline.transforms as tr
from dronalize.processing.screening.screen import Screen
from dronalize.processing.pipeline._internal import SPLIT_PARTITION_COLUMN
from dronalize.processing.pipeline.builder import TrajectoryPipelineBuilder
from dronalize.processing.pipeline.functional.resample import ResampleMethod, ResampleSpec
from dronalize.processing.pipeline.pipeline import Pipeline
from dronalize.processing.pipeline.splitting import apply_split_partition

if TYPE_CHECKING:
    from dronalize.config.sections import ResampleConfig, ScenesConfig, WindowConfig
    from dronalize.processing.pipeline.spec import TrajectorySpec


def trajectory_pipeline(spec: TrajectorySpec) -> Pipeline:
    """Build a trajectory-processing pipeline from a declarative spec."""
    builder = TrajectoryPipelineBuilder(spec)
    scenes_config = builder.scenes

    if spec.extension is not None:
        spec.extension.extend(builder)

    scene_id_column = builder.scene_id_column if builder.uses_scene_id() else None
    pipeline = apply_split_partition(
        Pipeline(),
        split_request=builder.split_request,
        time_column=spec.columns.frame,
        group_by=spec.window_by,
    )
    pipeline = pipeline.compose(builder.pre_window)
    pipeline = _apply_window(
        pipeline,
        frame_column=spec.columns.frame,
        window_group_columns=builder.window_group_columns(),
        window_spec=scenes_config.window,
        window_size=_total_frames(scenes_config),
    )
    pipeline = pipeline.compose(builder.post_window)
    pipeline = _attach_scene_id(
        pipeline, scene_key_columns=builder.scene_key_columns(), scene_id_column=scene_id_column
    )
    pipeline = _apply_screening(
        pipeline,
        screening_spec=(
            Screen.from_config(spec.plan.screening) if spec.plan.screening is not None else None
        ),
        scene_id_column=scene_id_column,
        agent_id_column=spec.columns.agent_id,
        frame_column=spec.columns.frame,
        category_column=spec.columns.category,
    )
    pipeline = pipeline.compose(builder.post_screening)
    return _apply_resample_and_yield(
        pipeline,
        frame_column=spec.columns.frame,
        agent_id_column=spec.columns.agent_id,
        scene_id_column=scene_id_column,
        resampling=_compile_resample_config(scenes_config, scenes_config.resample),
        drop=_factory_generated_drop_columns(builder, scene_id_column=scene_id_column),
    )


def _total_frames(config: ScenesConfig) -> int:
    return config.history_frames + config.future_frames


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
    window_spec: WindowConfig | None,
    window_size: int,
) -> Pipeline:
    if window_spec is None:
        return pipeline
    return pipeline.then(
        tr.window(
            window_size,
            window_spec.step,
            group_by=window_group_columns or None,
            sliding_col=frame_column,
        )
    )


def _apply_screening(
    pipeline: Pipeline,
    *,
    screening_spec: Screen | None,
    scene_id_column: str | None,
    agent_id_column: str,
    frame_column: str,
    category_column: str,
) -> Pipeline:
    if screening_spec is not None:
        return pipeline.then(
            tr.screen_scene(
                screening_spec,
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
    drop: list[str] | None = None,
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
    if drop is not None:
        pipeline = pipeline.then(tr.select(pl.all().exclude(drop)))
    return pipeline


def _factory_generated_drop_columns(
    builder: TrajectoryPipelineBuilder, *, scene_id_column: str | None
) -> list[str] | None:
    drop_columns: list[str] = []

    if builder.has_window:
        drop_columns.append("window_index")
    if builder.split_columns():
        drop_columns.append(SPLIT_PARTITION_COLUMN)
    if scene_id_column is not None:
        drop_columns.append(scene_id_column)

    return drop_columns or None


def _compile_resample_config(
    scenes_config: ScenesConfig, resample_config: ResampleConfig | None
) -> ResampleSpec | None:
    if resample_config is None:
        return None

    return ResampleSpec(
        up=resample_config.up,
        down=resample_config.down,
        sample_time=scenes_config.sample_time,
        coordinates=resample_config.coordinates,
        method=ResampleMethod(resample_config.method),
        max_gap=resample_config.max_gap,
        emit_velocity=resample_config.emit_velocity,
        emit_acceleration=resample_config.emit_acceleration,
    )
