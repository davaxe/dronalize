from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING, Literal, TypedDict, TypeVar, overload

import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as sl
import torch
from scipy.signal import step

if TYPE_CHECKING:
    from collections.abc import Iterable

    from preprocessing.trajectory.interface import ProcessorConfig

T_DataFame = TypeVar("T_DataFame", pl.DataFrame, pl.LazyFrame)


class Category(IntEnum):
    """Enumeration of categories of agents / objects."""

    CAR = auto()
    VAN = auto()
    TRAILER = auto()
    TRUCK = auto()
    TRAM = auto()
    BUS = auto()
    MOTORCYCLE = auto()
    BICYCLE = auto()
    PEDESTRIAN = auto()
    TRICYCLE = auto()
    ANIMAL = auto()
    STATIC_OBJECT = auto()
    MOVEABLE_OBJECT = auto()
    UNKNOWN = auto()
    EMERGENCY_VEHICLE = auto()


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


def derivative(
    data: T_DataFame,
    *x: str,
    dt: float = 1.0,
    n: int = 1,
    include_intermediate: bool = False,
    group_by: str | list[str] | None = None,
    derivative_rename: dict[int, list[str]] | None = None,
) -> T_DataFame:
    """Compute the n-th order derivative for specified columns using `np.gradient`.

    Calculates numerical derivatives for one or multiple columns. If multiple
    orders are requested via `include_intermediate`, all steps from $1$ to $n$
    are returned.

    Note:
        Raises an exception if the dataset contains fewer than two datapoints
        per group, as `np.gradient` requires sufficient padding.

    Args:
        data: Input DataFrame or LazyFrame.
        *x: Names of columns to differentiate.
        dt: Constant time step between samples. Defaults to 1.0.
        n: The maximum order of the derivative. Defaults to 1.
        include_intermediate: If True, retains all derivatives from 1 to n-1.
        group_by: Column(s) used to partition data before calculation.
        derivative_rename: Mapping of order to a list of new column names.
            Format: `{order: [name_x1, name_x2, ...]}`.

    Returns:
        The original data structure with new derivative columns appended.

    """
    derivative_rename = {} if derivative_rename is None else derivative_rename

    def calc_grad(s: pl.Series) -> np.ndarray:
        # np.gradient returns a numpy array of the same shape
        return np.gradient(s.to_numpy(), dt)

    current_col_names: list[str] = list(x)
    intermediate_cols: list[str] = []

    for i in range(1, n + 1):
        next_step_exprs = []
        next_col_names = []

        rename_list = derivative_rename.get(
            i, [f"d{i}_{original_root}" for original_root in x]
        )

        for j, col_name in enumerate(current_col_names):
            new_name = rename_list[j]
            grad_expr = pl.col(col_name).map_batches(
                calc_grad, return_dtype=pl.Float64
            )
            if group_by is not None:
                grad_expr = grad_expr.over(group_by)

            next_step_exprs.append(grad_expr.alias(new_name))
            next_col_names.append(new_name)

            if i < n:
                intermediate_cols.append(new_name)

        data = data.with_columns(next_step_exprs)
        current_col_names = next_col_names

    # 4. Cleanup
    if not include_intermediate:
        data = data.drop(intermediate_cols)

    return data


def filter_scene(
    data: T_DataFame,
    config: ProcessorConfig,
    group_by: str | None = None,
) -> T_DataFame:
    """Filter scenes based on configuration.

    Args:
        data: input data.
        config: ProcessorConfig object with filtering criteria.
        group_by: Optional column name to group by.

    Returns:
        DataFrame or LazyFrame with filtered scene(s).

    """
    if config.scene_filtering is None:
        return data

    filtering = config.scene_filtering

    # 1. Filter Invalid Agents (Rows)
    # Equivalent to: df.filter(pl.col("frame").n_unique().over("id") == total_len)
    # We must ensure we group by [scene, agent] if a scene grouper exists.
    if filtering.require_all_valid:
        total_len = config.input_len + config.output_len

        # If we have a scene grouper, we must partition the agent check by it
        # to avoid ID collisions across scenes (unless IDs are globally unique).
        agent_window = [group_by, "id"] if group_by else "id"

        data = data.filter(
            pl.col("frame").n_unique().over(agent_window) == total_len
        )

    # 2. Filter Invalid Scenes (Groups)
    # We define a window to calculate statistics per scene.
    # If no group_by is provided, we treat the whole frame as one group (or no-op).
    scene_window = group_by or pl.lit(1)

    # Criteria A: Minimum Agents
    # We check the count of unique IDs remaining in the scene/group.
    # Note: This runs AFTER the agent filter above, matching your original logic.
    has_enough_agents = (
        pl.col("id").n_unique().over(scene_window) >= filtering.min_agents
    )

    # Criteria B: Prediction Frame Existence
    # Logic: min(frame) + input_len - 1 must exist in the group.
    if filtering.require_prediction_frame:
        start_frame = pl.col("frame").min().over(scene_window)
        target_frame = start_frame + config.input_len - 1

        # Check if any row in the group matches the target frame
        has_pred_frame = (pl.col("frame") == target_frame).any().over(scene_window)

        # Combine the boolean masks
        scene_mask = has_enough_agents & has_pred_frame
    else:
        scene_mask = has_enough_agents

    # Apply the scene-level mask
    return data.filter(scene_mask)


@overload
def sliding_window(
    data: T_DataFame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    is_sorted: bool = False,
    include_boundaries: bool = False,
    return_iterable: Literal[True] = True,
) -> Iterable[pl.DataFrame]: ...


@overload
def sliding_window(
    data: T_DataFame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    is_sorted: bool = False,
    include_boundaries: bool = False,
    return_iterable: Literal[False],
) -> T_DataFame: ...


def sliding_window(
    data: T_DataFame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    is_sorted: bool = False,
    include_boundaries: bool = False,
    return_iterable: bool = True,
) -> Iterable[pl.DataFrame] | T_DataFame:
    """Generate sliding windows from a DataFrame.

    When returning as an iterable, the function yields DataFrames for each
    window. This means that if the input was a `pl.LazyFrame` it will be
    collected.

    Args:
        data: Input DataFrame to generate windows from.
        window_size: Number of rows in each window.
        step_size: Number of rows to move the window at each step.
        sliding_col: Column name to use for determining the window boundaries.
            Defaults to "frame".
        is_sorted: Whether the input DataFrame is already sorted by the sliding_col.
            If False, the function will sort the DataFrame by sliding_col before
            generating windows. Defaults to False.
        include_boundaries: Passed to `group_by_dynamic` to include window
            boundaries in the output. Defaults to False.
        return_iterable: Whether to return an iterable of DataFrames or a single
            DataFrame containing all windows. Defaults to True.

    Yields:
        DataFrames corresponding to each sliding window.

    """
    if not is_sorted:
        data = data.sort(sliding_col)

    if return_iterable:
        return _sliding_window_iterable(
            collect(data),
            window_size,
            step_size,
            sliding_col,
            include_boundaries=include_boundaries,
        )

    return _sliding_window(
        data,
        window_size,
        step_size,
        sliding_col,
        include_boundaries=include_boundaries,
    )


def _sliding_window_iterable(
    data: pl.DataFrame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    include_boundaries: bool = False,
) -> Iterable[pl.DataFrame]:
    for _, window in data.group_by_dynamic(
        sliding_col,
        every=f"{step_size}i",
        period=f"{window_size}i",
        include_boundaries=include_boundaries,
    ):
        if not window.is_empty():
            yield window


def _sliding_window(
    data: T_DataFame,
    window_size: int,
    step_size: int,
    sliding_col: str = "frame",
    *,
    include_boundaries: bool = False,
) -> T_DataFame:
    return (
        data
        .group_by_dynamic(
            sliding_col,
            every=f"{step_size}i",
            period=f"{window_size}i",
            include_boundaries=include_boundaries,
        )
        .agg(
            pl.col(sliding_col).alias(f"{sliding_col}_actual"),
            pl.all().exclude(sliding_col),
        )
        .with_row_index("window_index")
        .explode(sl.all().exclude("window_index", sliding_col))
        .drop(sliding_col)
        .rename({f"{sliding_col}_actual": sliding_col})
    )


def convert_to_agent_data_dict(
    data: pl.DataFrame,
    input_len: int,
    output_len: int,
    target_agent: int | None = None,
    *,
    category_mapping: dict[Category, int] | None = None,
) -> AgentData:
    """Convert `Scene` into a agent dictionary.

    The dictionary is in format that is later compatible with pytorch
    geometric HeteroData.

    Args:
        data: DataFrame containing the scene data.
        input_len: Number of observed frames.
        output_len: Number of frames to predict.
        target_agent: Optional track ID to use as the target node. If None, the
            first valid track will be used as the target.
        category_mapping: Optional mapping from Category enum to integer type for
            customized type encoding. If None, the integer value from the Enum will
            be used directly.

    Returns:
        Dictionary containing the agent data according to the
        AgentData TypedDict.

    """
    target_agent_id = _extract_target_agent(
        data, input_len, output_len, target_agent
    )
    time_steps = input_len + output_len
    start_frame = data["frame"].min()
    unique_ids = data["id"].unique().to_list()
    if target_agent_id in unique_ids:
        unique_ids.remove(target_agent_id)
    sorted_ids = [target_agent_id, *sorted(unique_ids)]

    num_agents = len(sorted_ids)
    # Create a mapping dict: {track_id: tensor_row_index}
    id_to_idx_map = {uid: i for i, uid in enumerate(sorted_ids)}

    # We add 'row_idx' (agent) and 'col_idx' (time) columns to the dataframe
    # Casting to proper types ensures numpy compatibility
    df_indexed = data.with_columns([
        pl
        .col("id")
        .replace(id_to_idx_map, default=None)
        .cast(pl.Int32)
        .alias("row_idx"),
        (pl.col("frame") - start_frame).cast(pl.Int32).alias("col_idx"),
    ])

    # Extract coordinate arrays (N_samples,)
    # We extract all data points in one go
    row_indices = df_indexed["row_idx"].to_numpy()
    col_indices = df_indexed["col_idx"].to_numpy()

    pos, vel, acc = _full_zeros((num_agents, time_steps, 2), n=3)
    yaw = np.zeros((num_agents, time_steps), dtype=np.float32)
    mask = np.zeros((num_agents, time_steps), dtype=bool)

    # Fill the arrays using the row and column indices
    pos[row_indices, col_indices] = df_indexed.select(["x", "y"]).to_numpy()
    vel[row_indices, col_indices] = df_indexed.select(["vx", "vy"]).to_numpy()
    acc[row_indices, col_indices] = df_indexed.select(["ax", "ay"]).to_numpy()
    yaw[row_indices, col_indices] = df_indexed["yaw"].to_numpy()
    mask[row_indices, col_indices] = True
    type_df = df_indexed.unique(subset="row_idx", keep="first").sort("row_idx")

    # Ensure the type array is aligned with row indices 0..N
    # (The sort above guarantees this because row_idx is 0..N)
    raw_categories = type_df["agent_class"].to_numpy()
    if category_mapping:
        type_array = np.array(
            [category_mapping.get(Category(c), -1) for c in raw_categories],
            dtype=np.int32,
        )
    else:
        type_array = raw_categories.astype(np.int32)

    return AgentData(
        num_nodes=num_agents,
        ta_index=0,  # Target agent is forced to index 0
        type=torch.from_numpy(type_array).int(),
        inp_pos=torch.from_numpy(pos[:, :input_len]),
        inp_vel=torch.from_numpy(vel[:, :input_len]),
        inp_acc=torch.from_numpy(acc[:, :input_len]),
        inp_yaw=torch.from_numpy(yaw[:, :input_len]),
        trg_pos=torch.from_numpy(pos[:, input_len:]),
        trg_vel=torch.from_numpy(vel[:, input_len:]),
        trg_acc=torch.from_numpy(acc[:, input_len:]),
        trg_yaw=torch.from_numpy(yaw[:, input_len:]),
        input_mask=torch.from_numpy(mask[:, :input_len]),
        valid_mask=torch.from_numpy(mask[:, input_len:]),
        # TODO: Correctly implement these masks
        ma_mask=torch.empty(1),
        sa_mask=torch.empty(1),
    )


def _extract_target_agent(
    data: pl.DataFrame,
    input_len: int,
    output_len: int,
    target_agent: int | None = None,
) -> int:
    if target_agent is None:
        # Target agent needs to have valid data for the entire sequence (input +
        # output).
        candidates = (
            data
            .group_by("id")
            .agg(
                valid_frames=pl.col("frame").n_unique(),
            )
            .filter(pl.col("valid_frames") >= input_len + output_len)
            .select(pl.col("id"))
        )

        if candidates.is_empty():
            msg = "No valid target agent found with sufficient valid frames."
            raise ValueError(msg)

        # Use a different variable name to avoid shadowing the argument
        return int(candidates.select(pl.col("id").first()).item())

    target_agent_frames = (
        data
        .filter(pl.col("id") == target_agent)
        .select(pl.col("frame").n_unique())
        .item()
    )
    if target_agent_frames < input_len + output_len:
        msg = (
            f"Specified target agent {target_agent} does not have enough "
            f"valid frames ({target_agent_frames}) for the required "
            f"input ({input_len}) and output ({output_len}) length."
        )
        raise ValueError(msg)

    return target_agent


def _full_zeros(
    shape: tuple[int, ...],
    n: int = 2,
    dtype: npt.DTypeLike = np.float32,
) -> tuple[npt.NDArray[np.float32], ...]:
    return tuple(np.zeros(shape, dtype=dtype) for _ in range(n))


def _full_nan(
    shape: tuple[int, ...],
    n: int = 2,
    dtype: npt.DTypeLike = np.float32,
) -> tuple[npt.NDArray[np.float32], ...]:
    return tuple(np.full(shape, np.nan, dtype=dtype) for _ in range(n))


if __name__ == "__main__":
    # Example usage of the utility functions
    df = pl.DataFrame({
        "x": [1, 1, 1, 1],
        "y": [1, 2, 3, 4],
    })
    rename_map = {
        1: ["vel_x", "vel_y"],
    }
    df = derivative(
        df,
        "x",
        "y",
        derivative_rename=rename_map,
        n=1,
    )
    print(df)
