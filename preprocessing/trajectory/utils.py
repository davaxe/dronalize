from typing import TypeVar

import polars as pl

T_DataFame = TypeVar("T_DataFame", pl.DataFrame, pl.LazyFrame)


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
