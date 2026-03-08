from __future__ import annotations

from typing import TypedDict

import numpy as np
import numpy.typing as npt
import polars as pl

from dronalize.core.datatypes.categories import AgentCategory

# TODO: Update this when ready


class NumpySceneDict(TypedDict):
    num_nodes: int
    ta_index: int
    type: npt.NDArray[np.int32]
    inp_pos: npt.NDArray[np.float32]
    inp_vel: npt.NDArray[np.float32]
    inp_acc: npt.NDArray[np.float32]
    inp_yaw: npt.NDArray[np.float32]
    trg_pos: npt.NDArray[np.float32]
    trg_vel: npt.NDArray[np.float32]
    trg_acc: npt.NDArray[np.float32]
    trg_yaw: npt.NDArray[np.float32]
    input_mask: npt.NDArray[np.bool]
    valid_mask: npt.NDArray[np.bool]
    ma_mask: npt.NDArray[np.bool]
    sa_mask: npt.NDArray[np.bool]


def convert_to_numpy_dict(
    data: pl.DataFrame,
    input_len: int,
    output_len: int,
    target_agent: int,
    *,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> NumpySceneDict:
    """Convert the dataframe for the scene into a dictionary of numpy arrays.

    The dictionary is in a format that is later compatible with PyTorch
    Geometric HeteroData when converted to Tensors.

    Parameters
    ----------
    data : pl.DataFrame
        DataFrame containing the scene data.
    input_len : int
        Number of observed frames.
    output_len : int
        Number of frames to predict.
    target_agent : int, optional
        Track ID to use as the target node. If None, the first valid track
        will be used as the target.
    category_mapping : dict[AgentCategory, int], optional
        Mapping from Category enum to integer type for customized type
        encoding. If None, the integer value from the Enum will be used
        directly.

    Returns
    -------
    dict
        Dictionary containing the agent data according to the AgentData
        TypedDict.

    """
    time_steps = input_len + output_len
    start_frame = data["frame"].min()
    unique_ids = data["id"].unique().to_list()
    if target_agent in unique_ids:
        unique_ids.remove(target_agent)
    sorted_ids = [target_agent, *sorted(unique_ids)]

    num_agents = len(sorted_ids)
    # Create a mapping dict: {track_id: tensor_row_index}
    id_to_idx_map = {uid: i for i, uid in enumerate(sorted_ids)}

    # We add 'row_idx' (agent) and 'col_idx' (time) columns to the dataframe
    # Casting to proper types ensures numpy compatibility
    df_indexed = data.with_columns([
        pl.col("id").replace_strict(id_to_idx_map, default=None).cast(pl.Int32).alias("row_idx"),
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
    raw_categories = type_df["agent_category"].to_numpy()
    if category_mapping:
        type_array = np.array(
            [category_mapping.get(AgentCategory(c), -1) for c in raw_categories],
            dtype=np.int32,
        )
    else:
        type_array = raw_categories.astype(np.int32)

    return {
        "num_nodes": num_agents,
        "ta_index": 0,  # Target agent is forced to index 0
        "type": type_array,
        "inp_pos": pos[:, :input_len],
        "inp_vel": vel[:, :input_len],
        "inp_acc": acc[:, :input_len],
        "inp_yaw": yaw[:, :input_len],
        "trg_pos": pos[:, input_len:],
        "trg_vel": vel[:, input_len:],
        "trg_acc": acc[:, input_len:],
        "trg_yaw": yaw[:, input_len:],
        "input_mask": mask[:, :input_len],
        "valid_mask": mask[:, input_len:],
        # TODO: Correctly implement these masks
        "ma_mask": np.empty(1, dtype=bool),
        "sa_mask": np.empty(1, dtype=bool),
    }


def target_candidates(
    data: pl.DataFrame,
    input_len: int,
    output_len: int,
    *,
    min_input_frames: int = 1,
    min_output_frames: int = 1,
) -> list[int]:
    """Return candidate target agents that have sufficient valid frames.

    An agent qualifies if it has at least `min_input_frames` observations
    in the input window and at least `min_output_frames` in the output window.
    """
    start_frame = data["frame"].min()
    return (
        data
        .with_columns((pl.col("frame") - start_frame).alias("t"))
        .group_by("id")
        .agg(
            input_frames=pl.col("t").filter(pl.col("t") < input_len).n_unique(),
            output_frames=pl.col("t").filter(pl.col("t") >= input_len).n_unique(),
        )
        .filter(
            (pl.col("input_frames") >= min_input_frames)
            & (pl.col("output_frames") >= min_output_frames)
        )
        .select("id")
        .to_series()
        .to_list()
    )


def _full_zeros(
    shape: tuple[int, ...],
    n: int = 2,
    dtype: npt.DTypeLike = np.float32,
) -> tuple[npt.NDArray[np.float32], ...]:
    return tuple(np.zeros(shape, dtype=dtype) for _ in range(n))
