from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.pipeline.functional.resample._common import ResampleMethod, ResampleSpec
from dronalize.pipeline.functional.resample._linear import linear_resample
from dronalize.pipeline.functional.resample._spline import (
    cubic_hermite_interpolator_factory,
    cubic_spline_interpolator_factory,
    pchip_interpolator_factory,
    spline_resample,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize._internal._typing import DataFrameT


def resample(
    data: DataFrameT,
    spec: ResampleSpec | None = None,
    *,
    frame_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
) -> DataFrameT:
    """Resample trajectory data according to an explicit resampling spec."""
    resample_spec = spec or ResampleSpec()

    match resample_spec.method:
        case ResampleMethod.LINEAR:
            return linear_resample(
                data=data,
                spec=resample_spec,
                frame_column=frame_column,
                group_by=group_by,
            )
        case ResampleMethod.CUBIC:
            return spline_resample(
                data=data,
                spec=resample_spec,
                frame_column=frame_column,
                group_by=group_by,
                interpolator_factory=cubic_spline_interpolator_factory,
            )
        case ResampleMethod.PCHIP:
            return spline_resample(
                data=data,
                spec=resample_spec,
                frame_column=frame_column,
                group_by=group_by,
                interpolator_factory=pchip_interpolator_factory,
            )
        case ResampleMethod.HERMITE:
            return spline_resample(
                data=data,
                spec=resample_spec,
                frame_column=frame_column,
                group_by=group_by,
                interpolator_factory=cubic_hermite_interpolator_factory,
            )
