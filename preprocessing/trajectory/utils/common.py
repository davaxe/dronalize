from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, TypeVar

import polars as pl

if TYPE_CHECKING:
    import torch


class AgentData(TypedDict):
    """TypedDict for a single pedestrian data sample."""

    num_nodes: int
    """Number of agents/nodes in the scene."""
    ta_index: int
    """Primary target agent index (integer in range [0, num_nodes-1])."""
    type: torch.Tensor
    """Integer tensor of shape (num_nodes,) indicating the category/type of each agent."""
    inp_pos: torch.Tensor
    """Position in meters, shape (num_nodes, input_len, 2)."""
    inp_vel: torch.Tensor
    """Velocity in m/s, shape (num_nodes, input_len, 2)."""
    inp_acc: torch.Tensor
    """Acceleration in m/s^2, shape (num_nodes, input_len, 2)."""
    inp_yaw: torch.Tensor
    """Orientation in radians, shape (num_nodes, input_len)."""
    trg_pos: torch.Tensor
    """Position in meters, shape (num_nodes, output_len, 2)."""
    trg_vel: torch.Tensor
    """Velocity in m/s, shape (num_nodes, output_len, 2)."""
    trg_acc: torch.Tensor
    """Acceleration in m/s^2, shape (num_nodes, output_len, 2)."""
    trg_yaw: torch.Tensor
    """Orientation in radians, shape (num_nodes, output_len)."""
    input_mask: torch.Tensor
    """Boolean mask indicating valid input data, shape (num_nodes, input_len)."""
    valid_mask: torch.Tensor
    """Boolean mask indicating valid data across output, shape (num_nodes, output_len)."""
    ma_mask: torch.Tensor
    sa_mask: torch.Tensor


T_DataFame = TypeVar("T_DataFame", pl.DataFrame, pl.LazyFrame)


def lazy(data: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Convert a DataFrame to a LazyFrame if necessary."""
    if isinstance(data, pl.DataFrame):
        return data.lazy()
    return data


def collect(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Resolve a LazyFrame to a DataFrame if necessary."""
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    return data


def yaw_from_vel(
    data: T_DataFame,
    vx_col: str = "vx",
    vy_col: str = "vy",
    yaw_col: str = "yaw",
) -> T_DataFame:
    """Estimate yaw from velocity vector.

    Args:
        data: Input data frame or lazy frame.
        vx_col: Name of the column containing x velocity. Defaults to "vx".
        vy_col: Name of the column containing y velocity. Defaults to "vy".
        yaw_col: Name of the column to store the estimated yaw. Defaults to "yaw".

    Returns:
        Data frame or lazy frame with the estimated yaw column added.

    """
    return data.with_columns(
        pl.arctan2(pl.col(vy_col), pl.col(vx_col)).alias(yaw_col)
    )


def yaw_from_pos(
    data: T_DataFame,
    x_col: str = "x",
    y_col: str = "y",
    yaw_col: str = "yaw",
) -> T_DataFame:
    """Estimate yaw from position differences.

    Args:
        data: Input data frame or lazy frame.
        x_col: Name of the column containing x position. Defaults to "x".
        y_col: Name of the column containing y position. Defaults to "y".
        yaw_col: Name of the column to store the estimated yaw. Defaults to "yaw".

    Returns:
        Data frame or lazy frame with the estimated yaw column added.

    """
    return data.with_columns(
        pl.arctan2(
            pl.col(y_col).diff(),
            pl.col(x_col).diff(),
        ).alias(yaw_col)
    )
