"""Public entry point for temporal resampling of trajectory tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dronalize.processing.pipeline.functional.resample._common import (
    ResampleMethod,
    ResampleSpec,
    resolve_request,
)
from dronalize.processing.pipeline.functional.resample._linear import linear_resample
from dronalize.processing.pipeline.functional.resample._spline import (
    cubic_spline_interpolator_factory,
    pchip_interpolator_factory,
    spline_resample,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.core.typing import DataFrameT


def resample(
    data: DataFrameT,
    spec: ResampleSpec | None = None,
    *,
    frame_column: str = "frame",
    group_by: str | Sequence[str] | None = None,
) -> DataFrameT:
    """Resample trajectory data according to an explicit resampling spec.

    Parameters
    ----------
    data : DataFrameT
        Polars `DataFrame` or `LazyFrame` containing the trajectory data to
        resample.
    spec : ResampleSpec | None, optional
        Resampling specification. When omitted, the default
        :class:`ResampleSpec` is used.
    frame_column : str, optional
        Column containing monotonically increasing frame indices within each
        trajectory group.
    group_by : str or sequence of str or None, optional
        Column or columns that define independent trajectories. When `None`,
        the full table is treated as one trajectory.

    Returns
    -------
    DataFrameT
        Resampled table of the same eager/lazy type as `data`.
    """
    resample_spec = spec or ResampleSpec()
    plan = resolve_request(resample_spec, frame_column=frame_column, group_by=group_by)
    match resample_spec.method:
        case ResampleMethod.LINEAR:
            return linear_resample(data=data, spec=resample_spec, plan=plan)
        case ResampleMethod.CUBIC:
            return spline_resample(
                data=data,
                spec=resample_spec,
                plan=plan,
                interpolator_factory=cubic_spline_interpolator_factory,
            )
        case ResampleMethod.PCHIP:
            return spline_resample(
                data=data,
                spec=resample_spec,
                plan=plan,
                interpolator_factory=pchip_interpolator_factory,
            )
