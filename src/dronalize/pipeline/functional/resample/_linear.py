from __future__ import annotations

from typing import TYPE_CHECKING, cast

import polars as pl
import polars.selectors as cs

from dronalize.pipeline.functional.resample._common import (
    SEGMENT_COLUMN,
    ResamplePlan,
    build_plan,
    segment_data,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize._internal._typing import DataFrameT
    from dronalize.pipeline.functional.resample._common import ResampleSpec


def linear_resample(
    data: DataFrameT,
    spec: ResampleSpec,
    *,
    frame_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
) -> DataFrameT:
    plan = build_plan(
        spec,
        frame_column=frame_column,
        group_by=group_by,
    )

    if spec.no_resampling:
        return data
    if spec.sort:
        data = data.sort([*plan.group_by, frame_column])

    data = segment_data(
        data,
        frame_column=frame_column,
        group_by=plan.group_by,
        max_gap=spec.max_gap,
        sort=False,
    )

    if spec.up > 1:
        data = _upsample_dataframe(data, factor=spec.up, plan=plan)
    if spec.down > 1:
        data = _downsample_dataframe(data, factor=spec.down, frame_column=frame_column)

    return data.drop(SEGMENT_COLUMN, strict=False)


def _downsample_dataframe(
    data: DataFrameT,
    *,
    factor: int,
    frame_column: str,
) -> DataFrameT:
    return data.filter(pl.col(frame_column) % factor == 0).with_columns(
        (pl.col(frame_column) // factor).alias(frame_column),
    )


def _upsample_dataframe(
    data: DataFrameT,
    *,
    factor: int,
    plan: ResamplePlan,
) -> DataFrameT:
    scaled = data.with_columns((pl.col(plan.frame_column) * factor).alias(plan.frame_column))
    frame_range = pl.int_range(
        pl.col(plan.frame_column).min(),
        pl.col(plan.frame_column).max() + 1,
        step=1,
        dtype=pl.Int64,
    ).alias(plan.frame_column)

    if plan.segment_keys:
        upsampled = (
            scaled
            .group_by(plan.segment_keys, maintain_order=True)
            .agg(frame_range)
            .explode(plan.frame_column)
        )
    else:
        upsampled = scaled.select(frame_range)

    on = [*plan.segment_keys, plan.frame_column]
    exprs: list[pl.Expr] = []

    if plan.position_columns:
        interpolated = cs.by_name(*plan.position_columns, require_all=False).interpolate()
        exprs.append(interpolated.over(plan.segment_keys) if plan.segment_keys else interpolated)

    carried = pl.exclude(*on, *plan.position_columns).forward_fill()
    exprs.append(carried.over(plan.segment_keys) if plan.segment_keys else carried)
    return (
        upsampled.join(cast("DataFrameT", scaled), on=on, how="left").sort(on).with_columns(exprs)
    )
