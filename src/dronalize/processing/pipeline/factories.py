"""Reusable composite pipeline factories for common trajectory processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import dronalize.processing.pipeline.transforms as tr
from dronalize.processing.ingest.splits import ShuffledTimeBlockSplit, SplitRequest, TimeBlockSplit
from dronalize.processing.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.processing.ingest.config import LoaderConfig

_SPLIT_PARTITION_COLUMN = "_split_partition"
_RAW_SPLIT_ASSIGNMENT_COLUMN = "_split_assignment"
_RAW_SPLIT_PARTITION_COLUMN = "_split_partition_raw"
_RAW_SPLIT_SEGMENT_COLUMN = "_split_segment"
_DEBUG_ORIGINAL_FRAME_COLUMN = "_debug_original_frame"
_DEBUG_BLOCK_SPLIT_CSV_PATH = "test.csv"


def _normalize_group_by(group_by: str | Sequence[str] | None) -> list[str]:
    return [group_by] if isinstance(group_by, str) else list(group_by or [])


def _finalize_split_pipeline(
    pipeline: Pipeline,
    *,
    split_labels: dict[int, str],
    split_source_column: str,
    split_column: str,
    partition_source_column: str | None = None,
    partition_column: str | None = None,
    group_columns: list[str] | None = None,
) -> Pipeline:
    new_cols = [pl.col(split_source_column).replace_strict(split_labels).alias(split_column)]
    drop_cols = {split_source_column} - {split_column}

    if partition_source_column and partition_column:
        group_cols = group_columns or []
        if group_cols:
            new_cols.append(
                pl
                .struct(*[pl.col(c) for c in (*group_cols, partition_source_column)])
                .hash()
                .alias(partition_column)
            )
        else:
            new_cols.append(pl.col(partition_source_column).alias(partition_column))
        drop_cols.add(partition_source_column)

    finalized = pipeline.with_columns(*new_cols)
    if drop_cols:
        finalized = finalized.then(tr.select(pl.all().exclude(list(drop_cols))))
    return finalized


def split_partition_pipeline(
    request: SplitRequest,
    *,
    time_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
    split_column: str = "split",
    partition_column: str | None = None,
) -> Pipeline:
    """Build a pipeline for partition-based splitting.

    If `request` uses a block-based split strategy, the returned pipeline writes
    named split labels to `split_column`. When `partition_column` is provided,
    each contiguous temporal partition also receives a stable partition id for
    downstream grouping.
    """
    split_labels = {i: split.value for i, split in enumerate(request.active_splits())}
    group_cols = _normalize_group_by(group_by)

    match request.strategy:
        case TimeBlockSplit(gap=gap):
            src_col = _RAW_SPLIT_PARTITION_COLUMN if partition_column else split_column
            return _finalize_split_pipeline(
                Pipeline().then(
                    tr.block_partition_cumulative(
                        weights=request.active_weights(),
                        time_column=time_column,
                        group_by=group_by,
                        gap=gap,
                        partition_column=src_col,
                    )
                ),
                split_labels=split_labels,
                split_source_column=src_col,
                split_column=split_column,
                partition_source_column=src_col if partition_column else None,
                partition_column=partition_column,
                group_columns=group_cols,
            )

        case ShuffledTimeBlockSplit(gap=gap, segments=segments):
            return _finalize_split_pipeline(
                Pipeline().then(
                    tr.block_partition_shuffle(
                        weights=request.active_weights(),
                        segments=segments,
                        time_column=time_column,
                        group_by=group_by,
                        gap=gap,
                        assignment_column=_RAW_SPLIT_ASSIGNMENT_COLUMN,
                        segment_column=_RAW_SPLIT_SEGMENT_COLUMN,
                        seed=request.seed,
                    )
                ),
                split_labels=split_labels,
                split_source_column=_RAW_SPLIT_ASSIGNMENT_COLUMN,
                split_column=split_column,
                partition_source_column=_RAW_SPLIT_SEGMENT_COLUMN,
                partition_column=partition_column,
                group_columns=group_cols,
            )

        case _:
            return Pipeline()


def trajectory_pipeline(
    config: LoaderConfig,
    *,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = "agent_category",
    split_request: SplitRequest | None = None,
    window_group_by: str | Sequence[str] | None = None,
) -> Pipeline:
    """Build the shared trajectory-processing pipeline used by most loaders."""
    pipeline = Pipeline()
    uses_block_split = split_request is not None and split_request.uses_block_split
    base_group_columns = _normalize_group_by(window_group_by)
    window_group_columns: list[str] = [
        *base_group_columns,
        *([_SPLIT_PARTITION_COLUMN] if uses_block_split else []),
    ]
    scene_group_columns: list[str] = [
        *([_SPLIT_PARTITION_COLUMN] if uses_block_split else []),
        *(["window_index"] if config.window is not None else []),
    ]

    if uses_block_split and split_request is not None:
        pipeline = (
            pipeline
            .with_columns(pl.col(frame_column).alias(_DEBUG_ORIGINAL_FRAME_COLUMN))
            .compose(
                split_partition_pipeline(
                    split_request,
                    time_column=frame_column,
                    group_by=window_group_by,
                    split_column="split",
                    partition_column=_SPLIT_PARTITION_COLUMN,
                )
            )
            .then(tr.select(pl.all().exclude(_DEBUG_ORIGINAL_FRAME_COLUMN)))
        )

    if config.window is not None:
        pipeline = pipeline.then(
            tr.window(
                config.window.window_size,
                config.window.step_size,
                group_by=window_group_columns or None,
            )
        )

    if config.filters is not None:
        pipeline = pipeline.then(
            tr.filter_scene(
                config.filters,
                group_by=scene_group_columns or None,
                agent_id=agent_id,
                frame_column=frame_column,
                category_column=category_column,
            ),
        )

    pipeline = pipeline.then(
        tr.resample(
            spec=config.resampling,
            frame_column=frame_column,
            group_by=[*scene_group_columns, agent_id],
        ),
    )

    if scene_group_columns:
        pipeline = pipeline.then_flat_map(tr.group_by_yield(*scene_group_columns))

    return pipeline


def highway_trajectory_pipeline(
    config: LoaderConfig,
    *,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = "agent_category",
    window_group_by: str | tuple[str, ...] | None = None,
) -> Pipeline:
    """Build a trajectory pipeline specialized for highway datasets."""
    return trajectory_pipeline(
        config,
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
        window_group_by=window_group_by,
    )
