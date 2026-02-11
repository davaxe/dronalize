from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

from preprocessing.common.agent_data import AgentData
from preprocessing.core.categories import AgentCategory


def convert_to_agent_data_dict(
    data: pl.DataFrame,
    input_len: int,
    output_len: int,
    target_agent: int | None = None,
    *,
    category_mapping: dict[AgentCategory, int] | None = None,
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
    target_agent_id = _extract_target_agent(data, input_len, output_len, target_agent)
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
    df_indexed = data.with_columns(
        [
            pl.col("id")
            .replace(id_to_idx_map, default=None)
            .cast(pl.Int32)
            .alias("row_idx"),
            (pl.col("frame") - start_frame).cast(pl.Int32).alias("col_idx"),
        ]
    )

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
    raw_categories = type_df["agent_category"].to_numpy()
    if category_mapping:
        type_array = np.array(
            [category_mapping.get(AgentCategory(c), -1) for c in raw_categories],
            dtype=np.int32,
        )
    else:
        type_array = raw_categories.astype(np.int32)

    return AgentData(
        num_nodes=num_agents,
        ta_index=0,  # Target agent is forced to index 0
        type=type_array,
        inp_pos=pos[:, :input_len],
        inp_vel=vel[:, :input_len],
        inp_acc=acc[:, :input_len],
        inp_yaw=yaw[:, :input_len],
        trg_pos=pos[:, input_len:],
        trg_vel=vel[:, input_len:],
        trg_acc=acc[:, input_len:],
        trg_yaw=yaw[:, input_len:],
        input_mask=mask[:, :input_len],
        valid_mask=mask[:, input_len:],
        # TODO: Correctly implement these masks
        ma_mask=np.empty(1, dtype=bool),
        sa_mask=np.empty(1, dtype=bool),
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
            data.group_by("id")
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
        data.filter(pl.col("id") == target_agent)
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
