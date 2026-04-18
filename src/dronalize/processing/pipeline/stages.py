"""Stage definitions and core stage builders for trajectory pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import dronalize.processing.pipeline.transforms as tr
from dronalize.config.models import ShuffledTimeSplitConfig, TimeSplitConfig
from dronalize.core.polars_ops import normalize_group_by
from dronalize.processing.pipeline.functional.resample import ResampleMethod, ResampleSpec
from dronalize.processing.pipeline.pipeline import Pipeline
from dronalize.processing.screening.screen import Screen

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.processing.models import SplitRequest
    from dronalize.processing.pipeline.spec import BuildContext


def build_split_stage(ctx: BuildContext) -> Pipeline:
    """Build the optional split-partition stage."""
    if not ctx.split_columns or ctx.split_request is None:
        return Pipeline()
    return _split_partition_pipeline(
        ctx.split_request,
        time_column=ctx.frame_column,
        group_by=ctx.spec.window_by,
        split_column="split",
        partition_column=ctx.split_columns[0],
    )


def build_window_stage(ctx: BuildContext) -> Pipeline:
    """Build the optional sliding-window stage."""
    window_spec = ctx.scenes.window
    if window_spec is None:
        return Pipeline()

    return Pipeline().then(
        tr.window(
            ctx.scenes.history_frames + ctx.scenes.future_frames,
            window_spec.step,
            group_by=list(ctx.window_group_columns) or None,
            sliding_col=ctx.frame_column,
        )
    )


def build_scene_id_stage(ctx: BuildContext) -> Pipeline:
    """Build the optional scene-ID materialization stage."""
    scene_id_column = ctx.scene_id_column
    if scene_id_column is None:
        return Pipeline()

    if not ctx.scene_key_columns:
        return Pipeline().then(
            tr.with_columns(pl.lit(0).cast(pl.UInt32).alias(scene_id_column)),
            name="attach_scene_id",
        )

    def _with_scene_id(df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df
            .with_row_index("_scene_row_order")
            .with_columns(
                pl
                .col("_scene_row_order")
                .min()
                .over(list(ctx.scene_key_columns))
                .alias("_scene_first_row")
            )
            .with_columns(
                (pl.col("_scene_first_row").rank("dense") - 1)
                .cast(pl.UInt32)
                .alias(scene_id_column)
            )
            .drop("_scene_row_order", "_scene_first_row")
        )

    return Pipeline().then(_with_scene_id, name="attach_scene_id")


def build_screening_stage(ctx: BuildContext) -> Pipeline:
    """Build the optional screening stage."""
    screening_spec = (
        Screen.from_config(ctx.plan.screening) if ctx.plan.screening is not None else None
    )
    if screening_spec is None:
        return Pipeline()

    return Pipeline().then(
        tr.screen_scene(
            screening_spec,
            columns=ctx.spec.columns,
            group_by=ctx.scene_id_column,
            mark_passed_agents=True,
        )
    )


def build_output_stage(ctx: BuildContext) -> Pipeline:
    """Build the resample, fan-out, and output-shaping stage."""
    group_by = [ctx.agent_id_column]
    if ctx.scene_id_column is not None:
        group_by.insert(0, ctx.scene_id_column)

    pipeline = Pipeline().then(
        tr.resample(
            spec=_compile_resample_config(ctx), frame_column=ctx.frame_column, group_by=group_by
        )
    )

    if ctx.scene_id_column is not None:
        pipeline = pipeline.then_flat_map(
            tr.group_by_yield(ctx.scene_id_column, drop_group_cols=False)
        )

    if ctx.drop_columns:
        pipeline = pipeline.then(tr.select(pl.all().exclude(list(ctx.drop_columns))))

    return pipeline


def _compile_resample_config(ctx: BuildContext) -> ResampleSpec | None:
    resample_config = ctx.scenes.resample
    if resample_config is None:
        return None

    return ResampleSpec(
        up=resample_config.up,
        down=resample_config.down,
        sample_time=ctx.scenes.sample_time,
        coordinates=resample_config.coordinates,
        method=ResampleMethod(resample_config.method),
        max_gap=resample_config.max_gap,
        emit_velocity=resample_config.emit_velocity,
        emit_acceleration=resample_config.emit_acceleration,
    )


_RAW_SPLIT_ASSIGNMENT_COLUMN = "_split_assignment"
_RAW_SPLIT_PARTITION_COLUMN = "_split_partition_raw"
_RAW_SPLIT_SEGMENT_COLUMN = "_split_segment"


def _finalize_split_pipeline(
    pipeline: Pipeline,
    *,
    split_labels: dict[int, str],
    split_source_column: str,
    split_column: str,
    partition_source_column: str | None = None,
    partition_column: str | None = None,
    group_columns: Sequence[str] | None = None,
) -> Pipeline:
    new_cols = [pl.col(split_source_column).replace_strict(split_labels).alias(split_column)]
    drop_cols: set[str] = {split_source_column} if split_source_column != split_column else set()

    if partition_source_column and partition_column:
        if group_columns:
            partition_keys = (*group_columns, partition_source_column)
            new_cols.append(
                pl
                .struct(*(pl.col(column) for column in partition_keys))
                .hash()
                .alias(partition_column)
            )
        else:
            new_cols.append(pl.col(partition_source_column).alias(partition_column))
        if partition_source_column != partition_column:
            drop_cols.add(partition_source_column)

    finalized = pipeline.with_columns(*new_cols)
    if drop_cols:
        finalized = finalized.then(tr.select(pl.all().exclude(list(drop_cols))))
    return finalized


def _split_partition_pipeline(
    request: SplitRequest,
    *,
    time_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
    split_column: str = "split",
    partition_column: str | None = None,
) -> Pipeline:
    """Build a pipeline for partition-based splitting."""
    split_labels = {i: split.value for i, split in enumerate(request.active_splits())}
    group_cols = normalize_group_by(group_by)
    config = request.config
    weights = request.active_weights()

    if isinstance(config, TimeSplitConfig):
        split_source_column = _RAW_SPLIT_PARTITION_COLUMN if partition_column else split_column
        pipeline = Pipeline().then(
            tr.block_partition_cumulative(
                weights=weights,
                time_column=time_column,
                group_by=group_by,
                gap=config.gap,
                partition_column=split_source_column,
            )
        )
        partition_source_column = split_source_column if partition_column else None
    elif isinstance(config, ShuffledTimeSplitConfig):
        split_source_column = _RAW_SPLIT_ASSIGNMENT_COLUMN
        pipeline = Pipeline().then(
            tr.block_partition_shuffle(
                weights=weights,
                segments=config.segments,
                time_column=time_column,
                group_by=group_by,
                gap=config.gap,
                assignment_column=split_source_column,
                segment_column=_RAW_SPLIT_SEGMENT_COLUMN,
                seed=request.seed,
            )
        )
        partition_source_column = _RAW_SPLIT_SEGMENT_COLUMN
    else:
        return Pipeline()

    return _finalize_split_pipeline(
        pipeline,
        split_labels=split_labels,
        split_source_column=split_source_column,
        split_column=split_column,
        partition_source_column=partition_source_column,
        partition_column=partition_column,
        group_columns=group_cols,
    )
