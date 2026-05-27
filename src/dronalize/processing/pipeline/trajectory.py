"""Direct assembly for trajectory-processing pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

import dronalize.processing.pipeline.transforms as tr
from dronalize.config.models.split import ShuffledTimeBlockAssign, TimeBlockAssign
from dronalize.core.functional import ResampleMethod, ResampleSpec
from dronalize.core.functional.basic import normalize_group_by
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.pipeline.pipeline import Pipeline
from dronalize.processing.screening.screen import ScreeningRuleSet

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.config.models.scenes import ScenesConfig
    from dronalize.processing.models import SplitAssignmentPlan, TrajectoryPipelinePlan


_SPLIT_PARTITION_COLUMN = "_split_partition"
_SCENE_ID_COLUMN = "_scene_id"
_WINDOW_INDEX_COLUMN = "window_index"

_LANE_CHANGE_EVENT_COLUMN = "valid_lane_change"
_SCENE_LANE_CHANGE_COUNT_COLUMN = "_scene_lane_change_count"

_RAW_SPLIT_ASSIGNMENT_COLUMN = "_split_assignment"
_RAW_SPLIT_PARTITION_COLUMN = "_split_partition_raw"
_RAW_SPLIT_SEGMENT_COLUMN = "_split_segment"


def build_trajectory_pipeline(
    plan: TrajectoryPipelinePlan,
    *,
    columns: TrajectoryColumns | None = None,
    window_by: str | Sequence[str] | None = None,
    lane_id_column: str = "lane_id",
) -> Pipeline:
    """Build the trajectory-processing pipeline for one execution plan."""
    state = _compile_state(
        plan,
        columns=TrajectoryColumns() if columns is None else columns,
        window_by=window_by,
        lane_id_column=lane_id_column,
    )

    return (
        Pipeline()
        >> _build_split_stage(state)
        >> _build_lane_change_pre_window_stage(state)
        >> _build_window_stage(state)
        >> _build_scene_id_stage(state)
        >> _build_screening_stage(state)
        >> _build_lane_change_post_screening_stage(state)
        >> _build_output_stage(state)
    )


@dataclass(frozen=True, slots=True)
class _TrajectoryPipelineState:
    plan: TrajectoryPipelinePlan
    columns: TrajectoryColumns
    window_by: tuple[str, ...]
    lane_id_column: str
    split_columns: tuple[str, ...]
    window_group_columns: tuple[str, ...]
    scene_key_columns: tuple[str, ...]
    scene_id_column: str | None
    drop_columns: tuple[str, ...]

    @property
    def frame_column(self) -> str:
        return self.columns.frame

    @property
    def agent_id_column(self) -> str:
        return self.columns.agent_id

    @property
    def scenes(self) -> ScenesConfig:
        return self.plan.scenes

    @property
    def assignment_request(self) -> SplitAssignmentPlan | None:
        return self.plan.assignment


def _compile_state(
    plan: TrajectoryPipelinePlan,
    *,
    columns: TrajectoryColumns,
    window_by: str | Sequence[str] | None,
    lane_id_column: str,
) -> _TrajectoryPipelineState:
    if plan.scenes.lane_change is not None and plan.scenes.window is None:
        msg = "Lane-change sampling requires window sampling to be enabled."
        raise ValueError(msg)

    window_by_columns = tuple(normalize_group_by(window_by))
    assignment_request = plan.assignment
    split_columns = (
        (_SPLIT_PARTITION_COLUMN,)
        if assignment_request is not None
        and assignment_request.strategy in {"time", "shuffled-time"}
        else ()
    )
    window_group_columns = (*window_by_columns, *split_columns)
    has_window = plan.scenes.window is not None
    scene_key_columns = (
        (*window_group_columns, _WINDOW_INDEX_COLUMN) if has_window else window_group_columns
    )
    scene_id_column = (
        _SCENE_ID_COLUMN if scene_key_columns or _uses_lane_change_sampling(plan) else None
    )
    drop_columns = (
        *((_WINDOW_INDEX_COLUMN,) if has_window else ()),
        *split_columns,
        *((scene_id_column,) if scene_id_column is not None else ()),
    )

    return _TrajectoryPipelineState(
        plan=plan,
        columns=columns,
        window_by=window_by_columns,
        lane_id_column=lane_id_column,
        split_columns=split_columns,
        window_group_columns=window_group_columns,
        scene_key_columns=scene_key_columns,
        scene_id_column=scene_id_column,
        drop_columns=drop_columns,
    )


def _uses_lane_change_sampling(plan: TrajectoryPipelinePlan) -> bool:
    config = plan.scenes.lane_change
    return config is not None and config.negative_keep_every != 1


def _build_split_stage(state: _TrajectoryPipelineState) -> Pipeline:
    if not state.split_columns or state.assignment_request is None:
        return Pipeline()
    return _split_partition_pipeline(
        state.assignment_request,
        time_column=state.frame_column,
        group_by=state.window_by or None,
        split_column="split",
        partition_column=state.split_columns[0],
    )


def _build_lane_change_pre_window_stage(state: _TrajectoryPipelineState) -> Pipeline:
    config = state.scenes.lane_change
    if config is None or not _uses_lane_change_sampling(state.plan):
        return Pipeline()

    return Pipeline().then(
        tr.valid_lane_change(
            persist=config.persist,
            margin_before=config.margin_before,
            margin_after=config.margin_after,
            frame_column=state.frame_column,
            agent_id_column=state.agent_id_column,
            lane_id_column=state.lane_id_column,
            group_by=list(state.window_group_columns) or None,
            valid_column=_LANE_CHANGE_EVENT_COLUMN,
        ),
        name="mark_lane_changes",
    )


def _build_window_stage(state: _TrajectoryPipelineState) -> Pipeline:
    window_spec = state.scenes.window
    if window_spec is None:
        return Pipeline()

    return Pipeline().then(
        tr.window(
            state.scenes.horizon_frames,
            window_spec.step,
            policy=window_spec.policy,
            group_by=list(state.window_group_columns) or None,
            sliding_col=state.frame_column,
        )
    )


def _build_scene_id_stage(state: _TrajectoryPipelineState) -> Pipeline:
    scene_id_column = state.scene_id_column
    if scene_id_column is None:
        return Pipeline()

    if not state.scene_key_columns:
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
                .over(list(state.scene_key_columns))
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


def _build_screening_stage(state: _TrajectoryPipelineState) -> Pipeline:
    screening_spec = (
        ScreeningRuleSet.from_config(state.plan.screening)
        if state.plan.screening is not None
        else None
    )
    if screening_spec is None:
        return Pipeline()

    return Pipeline().then(
        tr.screen_scene(
            screening_spec,
            columns=state.columns,
            group_by=state.scene_id_column,
            mark_passed_agents=True,
            retain_scene_passes=True,
        )
    )


def _build_lane_change_post_screening_stage(state: _TrajectoryPipelineState) -> Pipeline:
    config = state.scenes.lane_change
    if config is None or not _uses_lane_change_sampling(state.plan):
        return Pipeline()

    scene_id_column = state.scene_id_column
    if scene_id_column is None:
        msg = "Lane-change sampling requires scene IDs."
        raise ValueError(msg)

    def _label_scenes(df: pl.LazyFrame) -> pl.LazyFrame:
        return df.with_columns(
            pl
            .col(_LANE_CHANGE_EVENT_COLUMN)
            .sum()
            .over(scene_id_column)
            .alias(_SCENE_LANE_CHANGE_COUNT_COLUMN)
        )

    def _sample_scenes(df: pl.LazyFrame) -> pl.LazyFrame:
        is_positive = (
            pl.col(_SCENE_LANE_CHANGE_COUNT_COLUMN).fill_null(value=0)
            >= config.required_lane_changes
        )
        keep_negative = (pl.col(scene_id_column) % config.negative_keep_every) == 0
        return df.filter(is_positive | keep_negative).select(
            pl.all().exclude(_LANE_CHANGE_EVENT_COLUMN, _SCENE_LANE_CHANGE_COUNT_COLUMN)
        )

    return (
        Pipeline()
        .then(_label_scenes, name="label_lane_change_scenes")
        .then(_sample_scenes, name="sample_lane_change_scenes")
    )


def _build_output_stage(state: _TrajectoryPipelineState) -> Pipeline:
    group_by = [state.agent_id_column]
    if state.scene_id_column is not None:
        group_by.insert(0, state.scene_id_column)

    pipeline = Pipeline().then(
        tr.resample(
            spec=_compile_resample_config(state), frame_column=state.frame_column, group_by=group_by
        )
    )

    if state.scene_id_column is not None:
        pipeline = pipeline.then_flat_map(
            tr.group_by_yield(state.scene_id_column, drop_group_cols=False)
        )

    if state.drop_columns:
        pipeline = pipeline.then(tr.select(pl.all().exclude(list(state.drop_columns))))

    return pipeline


def _compile_resample_config(state: _TrajectoryPipelineState) -> ResampleSpec | None:
    resample_config = state.scenes.resample
    if resample_config is None:
        return None

    return ResampleSpec(
        up=resample_config.up,
        down=resample_config.down,
        sample_time=state.scenes.sample_time,
        coordinates=resample_config.coordinates,
        method=ResampleMethod(resample_config.method),
        max_gap=resample_config.max_gap,
        emit_velocity=resample_config.emit_velocity,
        emit_acceleration=resample_config.emit_acceleration,
    )


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
    request: SplitAssignmentPlan,
    *,
    time_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
    split_column: str = "split",
    partition_column: str | None = None,
) -> Pipeline:
    split_labels = {i: split.value for i, split in enumerate(request.active_splits())}
    group_cols = normalize_group_by(group_by)
    config = request.config
    weights = request.active_weights()

    if isinstance(config, TimeBlockAssign):
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
    elif isinstance(config, ShuffledTimeBlockAssign):
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
