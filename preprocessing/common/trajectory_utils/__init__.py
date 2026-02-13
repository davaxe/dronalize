from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from preprocessing.common.trajectory_utils.basic import (
    collect,
    yaw_from_pos,
    yaw_from_vel,
)
from preprocessing.common.trajectory_utils.convert import (
    convert_to_agent_data_dict as convert_to_agent_data_dict,
)
from preprocessing.common.trajectory_utils.derivative import derivative as derivative
from preprocessing.common.trajectory_utils.filter import (
    filter_scene as filter_scene,
)
from preprocessing.common.trajectory_utils.filter import (
    filter_scene_expr as filter_scene_expr,
)
from preprocessing.common.trajectory_utils.resample import (
    resample_tracks as resample_tracks,
)
from preprocessing.common.trajectory_utils.window import (
    sliding_window as sliding_window,
)

if TYPE_CHECKING:
    import polars as pl

    T_DataFrame = TypeVar("T_DataFrame", pl.DataFrame, pl.LazyFrame)
