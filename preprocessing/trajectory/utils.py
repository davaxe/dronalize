from enum import IntEnum, auto
from typing import TypedDict, TypeVar

import numpy as np
import numpy.typing as npt
import polars as pl
import torch

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
    target_agent = _extract_target_agent(
        data,
        input_len=input_len,
        output_len=output_len,
        target_agent=target_agent,
    )

    agent_id_to_index: dict[int, int] = {target_agent: 0}
    time_steps = input_len + output_len
    start: int = int(data.select(pl.col("frame").min()).item())
    num_agents: int = int(data.select(pl.col("id").n_unique()).item())
    agent_id_to_index.update({
        agent_id: index
        for index, agent_id in enumerate(
            data.select(pl.col("id").unique()).to_series().to_list()
        )
    })
    # TODO: Acceleration is not currently available:
    # - Is it needed? If so, add it soon.
    # - If not, it can be removed from `AgentData` and rest of codebase.
    base_shape = (num_agents, time_steps, 2)
    pos, vel, acc, yaw = _full_nan(base_shape, n=4)
    mask = np.zeros((num_agents, time_steps), dtype=bool)
    type_array = np.full((num_agents,), fill_value=-1, dtype=np.int32)
    for (agent_id, *_), group in data.group_by("id"):
        index = agent_id_to_index[agent_id]
        frames = group.select(pl.col("frame")).to_series().to_numpy()
        pos_array = group.select(pl.col(["x", "y"])).to_numpy()
        vel_array = group.select(pl.col(["vx", "vy"])).to_numpy()
        yaw_array = group.select(pl.col("yaw")).to_numpy()
        category = group.select(pl.col("agent_class").first()).item()

        pos[index, frames - start] = pos_array
        vel[index, frames - start] = vel_array
        yaw[index, frames - start] = yaw_array
        mask[index, frames - start] = True

        if category_mapping is not None:
            type_array[index] = category_mapping.get(Category(category), -1)
        else:
            type_array[index] = category

    # Set invalid values to 0.0 to avoid NaN
    for arr in (pos, vel, acc, yaw):
        arr[~mask] = 0.0

    return AgentData(
        num_nodes=num_agents,
        ta_index=target_agent if target_agent is not None else 0,
        # Explicitly convert to expected datatype to ensure compatibility with
        # downstream code, even if the input data may already be in the correct
        # format.
        type=torch.from_numpy(type_array).int(),
        inp_pos=torch.from_numpy(pos[:, :input_len, :]).float(),
        inp_vel=torch.from_numpy(vel[:, :input_len, :]).float(),
        inp_acc=torch.from_numpy(acc[:, :input_len, :]).float(),
        inp_yaw=torch.from_numpy(yaw[:, :input_len]).float(),
        trg_pos=torch.from_numpy(pos[:, input_len:, :]).float(),
        trg_vel=torch.from_numpy(vel[:, input_len:, :]).float(),
        trg_acc=torch.from_numpy(acc[:, input_len:, :]).float(),
        trg_yaw=torch.from_numpy(yaw[:, input_len:]).float(),
        input_mask=torch.from_numpy(mask[:, :input_len]).bool(),
        valid_mask=torch.from_numpy(mask[:, input_len:]).bool(),
        # TODO: Update to use actual masks for MA and SA instead of dummy masks.
        # Possibly use the function `get_masks` from `..common` to compute
        # these two masks below.
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


def _full_nan(
    shape: tuple[int, ...],
    n: int = 2,
    dtype: npt.DTypeLike = np.float32,
) -> tuple[npt.NDArray[np.float32], ...]:
    return tuple(np.full(shape, np.nan, dtype=dtype) for _ in range(n))
