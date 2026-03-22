from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.typing as npt
import polars as pl

from dronalize.categories import AgentCategory

if TYPE_CHECKING:
    from dronalize._internal._types import FloatDType
    from dronalize.maps.graph import MapGraph
    from dronalize.scene import Scene, SceneSchema
    from dronalize.storage.spec import (
        StorageMapSample,
        StorageMapSampleF32,
        StorageMapSampleF64,
        StorageSceneSample,
        StorageSceneSampleF32,
        StorageSceneSampleF64,
    )

PLACEHOLDER_FLOAT64 = np.zeros((1,), dtype=np.float64)
PLACEHOLDER_FLOAT32 = np.zeros((1,), dtype=np.float32)
PLACEHOLDER_I32 = np.zeros((1,), dtype=np.int32)


@overload
def scene_to_numpy_dict(
    scene: Scene,
    *,
    dtype: type[np.float32],
    offset_position: bool = True,
    scene_schema: SceneSchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> StorageSceneSampleF32: ...


@overload
def scene_to_numpy_dict(
    scene: Scene,
    *,
    dtype: type[np.float64],
    offset_position: bool = True,
    scene_schema: SceneSchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> StorageSceneSampleF64: ...


def scene_to_numpy_dict(
    scene: Scene,
    *,
    dtype: FloatDType,
    offset_position: bool = True,
    scene_schema: SceneSchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> StorageSceneSampleF32 | StorageSceneSampleF64:
    """Convert a Scene to a persisted tensor representation."""
    if scene_schema is not None:
        scene = scene.as_schema(scene_schema)

    data = scene.inner
    input_len = scene.input_len
    output_len = scene.output_len
    feature_columns = scene.schema.feature_columns()

    time_steps = input_len + output_len
    start_frame = data["frame"].min()
    unique_ids: list[int] = data["id"].unique().to_list()
    num_agents = len(unique_ids)
    id_to_idx_map = {uid: i for i, uid in enumerate(sorted(unique_ids))}
    df_indexed = data.with_columns([
        (pl.col("frame") - start_frame).alias("col_idx"),
        pl.col("id").replace_strict(id_to_idx_map).alias("row_idx"),
    ])

    if offset_position:
        mean_x = data["x"].mean()
        mean_y = data["y"].mean()
        offset = np.array([mean_x, mean_y], dtype=np.float64)
    else:
        offset = np.zeros((2,), dtype=np.float64)

    row_indices = df_indexed["row_idx"].to_numpy()
    col_indices = df_indexed["col_idx"].to_numpy()

    features = np.zeros((num_agents, time_steps, len(feature_columns)), dtype=dtype)
    mask = np.zeros((num_agents, time_steps), dtype=bool)

    for feature_idx, column in enumerate(feature_columns):
        values = df_indexed[column].to_numpy(writable=column in {"x", "y"})
        if column == "x":
            values -= offset[0]
        elif column == "y":
            values -= offset[1]
        features[row_indices, col_indices, feature_idx] = values.astype(dtype, copy=False)

    mask[row_indices, col_indices] = True
    type_df = df_indexed.unique(subset="row_idx", keep="first").sort("row_idx")

    raw_categories = type_df["agent_category"].to_numpy()
    if category_mapping:
        agent_types = np.array(
            [category_mapping.get(AgentCategory(c), -1) for c in raw_categories],
            dtype=np.int32,
        )
    else:
        agent_types = raw_categories.astype(np.int32)

    scene_dict: StorageSceneSample[Any] = {
        "scene_number": scene.number,
        "global_origin": offset,
        "num_agents": num_agents,
        "agent_types": agent_types,
        "input_features": features[:, :input_len, :],
        "target_features": features[:, input_len:, :],
        "input_mask": mask[:, :input_len],
        "target_mask": mask[:, input_len:],
    }
    return scene_dict


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float32],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[True],
) -> StorageMapSampleF32: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[True],
) -> StorageMapSampleF64: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float32],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[False] = False,
) -> StorageMapSampleF32 | None: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[False] = False,
) -> StorageMapSampleF64 | None: ...


def encode_map_from_scene(
    scene: Scene,
    dtype: FloatDType,
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: bool = False,
) -> StorageMapSampleF32 | StorageMapSampleF64 | None:
    """Resolve the map graph from the scene and convert it to a persisted layout."""
    graph = scene.resolve_map()
    if graph is None:
        if not return_empty:
            return None
        map_dict: StorageMapSample[Any] = {
            "map_num_nodes": 0,
            "map_num_edges": 0,
            "map_node_positions": np.zeros((0, 2), dtype=dtype),
            "map_edge_indices": np.zeros((0, 2), dtype=np.int32),
            "map_node_types": np.zeros((0,), dtype=np.int32),
            "map_edge_types": np.zeros((0,), dtype=np.int32),
        }
        return map_dict

    return map_graph_to_numpy(graph, dtype=dtype, offset=offset)


@overload
def map_graph_to_numpy(
    graph: MapGraph,
    dtype: type[np.float32],
    offset: npt.NDArray[np.float64] | None = None,
) -> StorageMapSampleF32: ...


@overload
def map_graph_to_numpy(
    graph: MapGraph,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None = None,
) -> StorageMapSampleF64: ...


def map_graph_to_numpy(
    graph: MapGraph,
    dtype: FloatDType,
    offset: npt.NDArray[np.float64] | None = None,
) -> StorageMapSampleF32 | StorageMapSampleF64:
    """Convert a MapGraph to a persisted dictionary of NumPy arrays."""
    node_positions = graph.node_positions - offset if offset is not None else graph.node_positions
    map_dict: StorageMapSample[Any] = {
        "map_num_nodes": graph.num_nodes,
        "map_num_edges": graph.num_edges,
        "map_node_positions": node_positions.astype(dtype, copy=False),
        "map_edge_indices": np.ascontiguousarray(graph.edge_indices.T, dtype=np.int32),
        "map_node_types": graph.node_types,
        "map_edge_types": graph.edge_types,
    }
    return map_dict
