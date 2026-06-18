"""Trajectory scene preparation around the explicit screening boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from dronalize.config.models import ShuffledTimeBlockAssign, TimeBlockAssign
from dronalize.core import functional as f
from dronalize.core.functional import ResampleMethod, ResampleSpec
from dronalize.core.functional.basic import normalize_group_by
from dronalize.processing.columns import TrajectoryColumns
from dronalize.processing.screening.screen import ScreeningRuleSet, screen_data

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from dronalize.config.models import ScenesConfig
    from dronalize.processing.models import SplitAssignmentPlan, TrajectoryPipelinePlan
    from dronalize.processing.screening.screen import ScreeningResult


_SPLIT_PARTITION_COLUMN = "_split_partition"
_SCENE_ID_COLUMN = "_scene_id"
_WINDOW_INDEX_COLUMN = "window_index"

_LANE_CHANGE_EVENT_COLUMN = "valid_lane_change"
_SCENE_LANE_CHANGE_COUNT_COLUMN = "_scene_lane_change_count"

_RAW_SPLIT_ASSIGNMENT_COLUMN = "_split_assignment"
_RAW_SPLIT_PARTITION_COLUMN = "_split_partition_raw"
_RAW_SPLIT_SEGMENT_COLUMN = "_split_segment"


@dataclass(frozen=True, slots=True)
class TrajectoryProcessingStages:
    """Runtime-oriented trajectory processing split around screening."""

    state: _TrajectoryProcessingState
    screening: ScreeningRuleSet | None

    @property
    def columns(self) -> TrajectoryColumns:
        """Return the trajectory column mapping used by screening."""
        return self.state.columns

    def iter_prescreen_frames(self, frame: pl.LazyFrame | pl.DataFrame) -> Iterator[pl.DataFrame]:
        """Yield candidate scene frames ready for structured screening."""
        current = _as_lazy(frame)
        current = _apply_split_partition(current, self.state)
        current = _mark_lane_changes(current, self.state)
        current = _apply_window(current, self.state)
        current = _attach_scene_id(current, self.state)
        yield from _collect_non_empty(_iter_candidate_frames(current, self.state))

    def screen(self, frame: pl.DataFrame) -> ScreeningResult:
        """Screen one candidate scene frame and return structured metadata."""
        return screen_data(frame, scene_screening=self.screening, columns=self.columns)

    def iter_output_frames(self, frame: pl.DataFrame) -> Iterator[pl.DataFrame]:
        """Yield post-screening frames ready for output routing."""
        current = frame.lazy()
        current = _filter_lane_change_scenes(current, self.state)
        current = _resample_output(current, self.state)

        frames: Iterable[pl.LazyFrame]
        if self.state.scene_id_column is None:
            frames = (current,)
        else:
            frames = _iter_grouped_lazyframes(
                current, self.state.scene_id_column, drop_group_cols=False
            )

        if self.state.drop_columns:
            frames = (
                frame.select(pl.all().exclude(list(self.state.drop_columns))) for frame in frames
            )

        yield from _collect_non_empty(frames)


def build_trajectory_processing_stages(
    plan: TrajectoryPipelinePlan,
    *,
    columns: TrajectoryColumns | None = None,
    window_by: str | Sequence[str] | None = None,
    lane_id_column: str = "lane_id",
) -> TrajectoryProcessingStages:
    """Build trajectory-processing stages with screening as a runtime boundary."""
    state = _compile_state(
        plan,
        columns=TrajectoryColumns() if columns is None else columns,
        window_by=window_by,
        lane_id_column=lane_id_column,
    )
    screening = (
        ScreeningRuleSet.from_config(state.plan.screening)
        if state.plan.screening is not None
        else None
    )
    return TrajectoryProcessingStages(state=state, screening=screening)


@dataclass(frozen=True, slots=True)
class _TrajectoryProcessingState:
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
) -> _TrajectoryProcessingState:
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

    return _TrajectoryProcessingState(
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


def _as_lazy(frame: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def _collect_non_empty(frames: Iterable[pl.LazyFrame]) -> Iterator[pl.DataFrame]:
    for frame in frames:
        collected = frame.collect()
        if not collected.is_empty():
            yield collected


def _apply_split_partition(frame: pl.LazyFrame, state: _TrajectoryProcessingState) -> pl.LazyFrame:
    if not state.split_columns or state.assignment_request is None:
        return frame

    return _split_partition_frame(
        frame,
        state.assignment_request,
        time_column=state.frame_column,
        group_by=state.window_by or None,
        split_column="split",
        partition_column=state.split_columns[0],
    )


def _mark_lane_changes(frame: pl.LazyFrame, state: _TrajectoryProcessingState) -> pl.LazyFrame:
    config = state.scenes.lane_change
    if config is None or not _uses_lane_change_sampling(state.plan):
        return frame

    return f.valid_lane_change(
        frame,
        persist=config.persist,
        margin_before=config.margin_before,
        margin_after=config.margin_after,
        frame_column=state.frame_column,
        agent_id_column=state.agent_id_column,
        lane_id_column=state.lane_id_column,
        group_by=list(state.window_group_columns) or None,
        valid_column=_LANE_CHANGE_EVENT_COLUMN,
    )


def _apply_window(frame: pl.LazyFrame, state: _TrajectoryProcessingState) -> pl.LazyFrame:
    window_spec = state.scenes.window
    if window_spec is None:
        return frame

    return f.sliding_window(
        frame,
        window_size=state.scenes.horizon_frames,
        step_size=window_spec.step,
        policy=window_spec.policy,
        group_by=list(state.window_group_columns) or None,
        sliding_col=state.frame_column,
    )


def _attach_scene_id(frame: pl.LazyFrame, state: _TrajectoryProcessingState) -> pl.LazyFrame:
    scene_id_column = state.scene_id_column
    if scene_id_column is None:
        return frame

    if not state.scene_key_columns:
        return frame.with_columns(pl.lit(0).cast(pl.UInt32).alias(scene_id_column))

    return (
        frame
        .with_row_index("_scene_row_order")
        .with_columns(
            pl
            .col("_scene_row_order")
            .min()
            .over(list(state.scene_key_columns))
            .alias("_scene_first_row")
        )
        .with_columns(
            (pl.col("_scene_first_row").rank("dense") - 1).cast(pl.UInt32).alias(scene_id_column)
        )
        .drop("_scene_row_order", "_scene_first_row")
    )


def _iter_candidate_frames(
    frame: pl.LazyFrame, state: _TrajectoryProcessingState
) -> Iterable[pl.LazyFrame]:
    if state.scene_id_column is None:
        return (frame,)
    return _iter_grouped_lazyframes(frame, state.scene_id_column, drop_group_cols=False)


def _filter_lane_change_scenes(
    frame: pl.LazyFrame, state: _TrajectoryProcessingState
) -> pl.LazyFrame:
    config = state.scenes.lane_change
    if config is None or not _uses_lane_change_sampling(state.plan):
        return frame

    scene_id_column = state.scene_id_column
    if scene_id_column is None:
        msg = "Lane-change sampling requires scene IDs."
        raise ValueError(msg)

    is_positive = (
        pl.col(_SCENE_LANE_CHANGE_COUNT_COLUMN).fill_null(value=0) >= config.required_lane_changes
    )
    keep_negative = (pl.col(scene_id_column) % config.negative_keep_every) == 0

    return (
        frame
        .with_columns(
            pl
            .col(_LANE_CHANGE_EVENT_COLUMN)
            .sum()
            .over(scene_id_column)
            .alias(_SCENE_LANE_CHANGE_COUNT_COLUMN)
        )
        .filter(is_positive | keep_negative)
        .select(pl.all().exclude(_LANE_CHANGE_EVENT_COLUMN, _SCENE_LANE_CHANGE_COUNT_COLUMN))
    )


def _resample_output(frame: pl.LazyFrame, state: _TrajectoryProcessingState) -> pl.LazyFrame:
    group_by = [state.agent_id_column]
    if state.scene_id_column is not None:
        group_by.insert(0, state.scene_id_column)

    return f.resample(
        frame,
        _compile_resample_config(state) or ResampleSpec(),
        frame_column=state.frame_column,
        group_by=group_by,
        time_origin_by=(state.scene_id_column if state.scene_id_column is not None else ()),
    )


def _compile_resample_config(state: _TrajectoryProcessingState) -> ResampleSpec | None:
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


def _iter_grouped_lazyframes(
    frame: pl.LazyFrame, *by: str, drop_group_cols: bool = True, batch_size: int = 100_000
) -> Iterator[pl.LazyFrame]:
    by_tuple = tuple(by)
    if not by_tuple:
        yield frame
        return

    pending: pl.DataFrame | None = None
    for collected_batch in frame.collect_batches(chunk_size=batch_size, maintain_order=True):
        batch = collected_batch
        if pending is not None:
            batch = pl.concat([pending, batch], how="vertical")
            pending = None
        if batch.is_empty():
            continue

        last_key = batch.select(*by_tuple).tail(1).row(0, named=True)
        last_key_mask = pl.all_horizontal(
            *(pl.col(column).eq_missing(value) for column, value in last_key.items())
        )
        complete = batch.filter(~last_key_mask)
        pending = batch.filter(last_key_mask)

        if not complete.is_empty():
            yield from _yield_partitions(
                complete, by_tuple=by_tuple, drop_group_cols=drop_group_cols
            )

    if pending is not None and not pending.is_empty():
        yield from _yield_partitions(pending, by_tuple=by_tuple, drop_group_cols=drop_group_cols)


def _yield_partitions(
    frame: pl.DataFrame, *, by_tuple: tuple[str, ...], drop_group_cols: bool
) -> Iterator[pl.LazyFrame]:
    for part in frame.partition_by(
        *by_tuple, maintain_order=True, as_dict=False, include_key=not drop_group_cols
    ):
        yield part.lazy()


def _split_partition_frame(
    frame: pl.LazyFrame,
    request: SplitAssignmentPlan,
    *,
    time_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
    split_column: str = "split",
    partition_column: str | None = None,
) -> pl.LazyFrame:
    split_labels = {i: split.value for i, split in enumerate(request.active_splits())}
    group_cols = normalize_group_by(group_by)
    config = request.config
    weights = request.active_weights()

    if isinstance(config, TimeBlockAssign):
        split_source_column = _RAW_SPLIT_PARTITION_COLUMN if partition_column else split_column
        partitioned = f.cumulative_blocks(
            frame,
            weights=weights,
            time_column=time_column,
            group_by=group_by,
            gap=config.gap,
            partition_column=split_source_column,
        )
        partition_source_column = split_source_column if partition_column else None
    elif isinstance(config, ShuffledTimeBlockAssign):
        split_source_column = _RAW_SPLIT_ASSIGNMENT_COLUMN
        partitioned = f.shuffled_blocks(
            frame,
            weights=weights,
            n_segments=config.segments,
            time_column=time_column,
            group_by=group_by,
            gap=config.gap,
            assignment_column=split_source_column,
            segment_column=_RAW_SPLIT_SEGMENT_COLUMN,
            seed=request.seed,
        )
        partition_source_column = _RAW_SPLIT_SEGMENT_COLUMN
    else:
        return frame

    return _finalize_split_frame(
        partitioned,
        split_labels=split_labels,
        split_source_column=split_source_column,
        split_column=split_column,
        partition_source_column=partition_source_column,
        partition_column=partition_column,
        group_columns=group_cols,
    )


def _finalize_split_frame(
    frame: pl.LazyFrame,
    *,
    split_labels: dict[int, str],
    split_source_column: str,
    split_column: str,
    partition_source_column: str | None = None,
    partition_column: str | None = None,
    group_columns: Sequence[str] | None = None,
) -> pl.LazyFrame:
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

    finalized = frame.with_columns(*new_cols)
    if drop_cols:
        return finalized.select(pl.all().exclude(list(drop_cols)))
    return finalized
