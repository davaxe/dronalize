"""Split-specific helpers for trajectory processing pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import dronalize.processing.pipeline.transforms as tr
from dronalize._internal._polars_ops import normalize_group_by
from dronalize.processing.ingest.splits import ShuffledTimeBlockSplit, SplitRequest, TimeBlockSplit
from dronalize.processing.pipeline._internal import SPLIT_PARTITION_COLUMN
from dronalize.processing.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    """Build a pipeline for partition-based splitting."""
    split_labels = {i: split.value for i, split in enumerate(request.active_splits())}
    group_cols = normalize_group_by(group_by)

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


def apply_split_partition(
    pipeline: Pipeline,
    *,
    split_request: SplitRequest | None,
    time_column: str,
    group_by: str | Sequence[str] | None,
) -> Pipeline:
    """Attach block-based split partitioning to the pipeline when requested."""
    if split_request and split_request.uses_block_split:
        return pipeline.compose(
            split_partition_pipeline(
                split_request,
                time_column=time_column,
                group_by=group_by,
                split_column="split",
                partition_column=SPLIT_PARTITION_COLUMN,
            )
        )
    return pipeline
