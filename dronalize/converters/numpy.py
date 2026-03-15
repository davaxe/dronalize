from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import numpy.typing as npt
import polars as pl

from dronalize.categories import AgentCategory
from dronalize.pipeline.functional.targets import target_candidates

if TYPE_CHECKING:
    from dronalize.maps.graph import MapGraph
    from dronalize.scene import Scene


class NumpySceneDict(TypedDict):
    """TypedDict for the output of `convert_to_numpy_dict`."""

    num_nodes: int
    ta_index: int
    type: npt.NDArray[np.int32]
    inp_pos: npt.NDArray[np.float64]
    inp_vel: npt.NDArray[np.float64]
    inp_acc: npt.NDArray[np.float64]
    inp_yaw: npt.NDArray[np.float64]
    trg_pos: npt.NDArray[np.float64]
    trg_vel: npt.NDArray[np.float64]
    trg_acc: npt.NDArray[np.float64]
    trg_yaw: npt.NDArray[np.float64]
    input_mask: npt.NDArray[np.bool_]
    valid_mask: npt.NDArray[np.bool_]
    ma_mask: npt.NDArray[np.bool_]
    sa_mask: npt.NDArray[np.bool_]


def scene_to_numpy_dict(
    scene: Scene,
    *,
    multiple_targets: bool | int = False,
    target_agent: int | None = None,
) -> dict[int, NumpySceneDict]:
    """Convert a Scene to a numpy representation compatible with Pytorch.

    Parameters
    ----------
    scene: Scene
        The scene to convert.
    multiple_targets : bool or int, optional
        Whether to return multile "samples" from each scene by changing the
        target angent. If `True` as many as possible samples will be
        returned, if an integer is given, at most that many samples will be
        returned.
    target_agent : int, optional
        If `multiple_targets` is False, this specifies the track ID to use as
        the target agent. If None, the first valid track will be used as the
        target.

    Returns
    -------
    dict[int, NumpySceneDict]
        A dictionary mapping target agent track IDs to their corresponding
        numpy representations of the scene data.

    """
    if not multiple_targets and target_agent is not None:
        return {
            target_agent: _convert_to_numpy_dict(
                scene.inner, scene.input_len, scene.output_len, target_agent
            )
        }

    candidates = target_candidates(scene.inner, scene.input_len)
    candidates = candidates[:1] if multiple_targets is False else candidates[:multiple_targets]

    return {
        target: _convert_to_numpy_dict(
            scene.inner,
            scene.input_len,
            scene.output_len,
            target,
        )
        for target in candidates
    }


def _convert_to_numpy_dict(
    data: pl.DataFrame,
    input_len: int,
    output_len: int,
    target_agent: int,
    *,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> NumpySceneDict:
    """Convert the dataframe for the scene into a dictionary of numpy arrays."""
    time_steps = input_len + output_len
    start_frame = data["frame"].min()
    unique_ids = data["id"].unique().to_list()
    if target_agent in unique_ids:
        unique_ids.remove(target_agent)
    sorted_ids = [target_agent, *sorted(unique_ids)]

    num_agents = len(sorted_ids)
    id_to_idx_map = {uid: i for i, uid in enumerate(sorted_ids)}

    df_indexed = data.with_columns([
        pl.col("id").replace_strict(id_to_idx_map, default=None).cast(pl.Int32).alias("row_idx"),
        (pl.col("frame") - start_frame).cast(pl.Int32).alias("col_idx"),
    ])

    row_indices = df_indexed["row_idx"].to_numpy()
    col_indices = df_indexed["col_idx"].to_numpy()

    pos, vel, acc = _full_zeros((num_agents, time_steps, 2), n=3)
    yaw = np.zeros((num_agents, time_steps), dtype=np.float64)
    mask = np.zeros((num_agents, time_steps), dtype=bool)

    pos[row_indices, col_indices] = df_indexed.select(["x", "y"]).to_numpy()
    vel[row_indices, col_indices] = df_indexed.select(["vx", "vy"]).to_numpy()
    acc[row_indices, col_indices] = df_indexed.select(["ax", "ay"]).to_numpy()
    yaw[row_indices, col_indices] = df_indexed["yaw"].to_numpy()
    mask[row_indices, col_indices] = True
    type_df = df_indexed.unique(subset="row_idx", keep="first").sort("row_idx")

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
        "ta_index": 0,
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
        "ma_mask": np.empty(1, dtype=bool),
        "sa_mask": np.empty(1, dtype=bool),
    }


class NumpyMapGraphDict(TypedDict):
    """TypedDict for the output of `map_graph_to_numpy`."""

    map_num_nodes: int
    map_num_edges: int
    map_node_positions: npt.NDArray[np.float64]
    map_edge_indices: npt.NDArray[np.int32]
    map_node_types: npt.NDArray[np.int32]
    map_edge_types: npt.NDArray[np.int32]


def _full_zeros(
    shape: tuple[int, ...],
    n: int = 2,
    dtype: npt.DTypeLike = np.float64,
) -> tuple[npt.NDArray[np.float64], ...]:
    return tuple(np.zeros(shape, dtype=dtype) for _ in range(n))


def map_graph_to_numpy(graph: MapGraph) -> NumpyMapGraphDict:
    """Convert the `MapGraph` to a dictionary of NumPy arrays.

    Returns
    -------
    dict
        Dictionary containing the graph data as NumPy arrays.

    """
    return {
        "map_num_nodes": graph.num_nodes,
        "map_num_edges": graph.num_edges,
        "map_node_positions": graph.node_positions,
        "map_edge_indices": graph.edge_indices,
        "map_node_types": graph.node_types,
        "map_edge_types": graph.edge_types,
    }
