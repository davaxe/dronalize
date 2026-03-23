from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.typing as npt
import polars as pl

from dronalize.categories import AgentCategory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize._internal._typing import FloatDType
    from dronalize.maps.graph import MapGraph
    from dronalize.scene import Scene, SceneSchema
    from dronalize.storage.spec import (
        MapSample,
        MapSampleF32,
        MapSampleF64,
        SceneSample,
        SceneSampleF32,
        SceneSampleF64,
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
) -> SceneSampleF32: ...


@overload
def scene_to_numpy_dict(
    scene: Scene,
    *,
    dtype: type[np.float64],
    offset_position: bool = True,
    scene_schema: SceneSchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> SceneSampleF64: ...


def scene_to_numpy_dict(
    scene: Scene,
    *,
    dtype: FloatDType,
    offset_position: bool = True,
    scene_schema: SceneSchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> SceneSampleF32 | SceneSampleF64:
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

    scene_dict: SceneSample[Any] = {
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
) -> MapSampleF32: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[True],
) -> MapSampleF64: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float32],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[False] = False,
) -> MapSampleF32 | None: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[False] = False,
) -> MapSampleF64 | None: ...


def encode_map_from_scene(
    scene: Scene,
    dtype: FloatDType,
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: bool = False,
) -> MapSampleF32 | MapSampleF64 | None:
    """Resolve the map graph from the scene and convert it to a persisted layout."""
    graph = scene.resolve_map()
    if graph is None:
        if not return_empty:
            return None
        map_dict: MapSample[Any] = {
            "map_num_nodes": 0,
            "map_num_edges": 0,
            "map_node_positions": np.zeros((1,), dtype=dtype),
            "map_edge_indices": np.zeros((1,), dtype=np.int32),
            "map_node_types": np.zeros((1,), dtype=np.int32),
            "map_edge_types": np.zeros((1,), dtype=np.int32),
        }
        return map_dict

    return _map_graph_to_numpy(graph, dtype=dtype, offset=offset)


@overload
def _map_graph_to_numpy(
    graph: MapGraph,
    dtype: type[np.float32],
    offset: npt.NDArray[np.float64] | None = None,
) -> MapSampleF32: ...


@overload
def _map_graph_to_numpy(
    graph: MapGraph,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None = None,
) -> MapSampleF64: ...


def _map_graph_to_numpy(
    graph: MapGraph,
    dtype: FloatDType,
    offset: npt.NDArray[np.float64] | None = None,
) -> MapSampleF32 | MapSampleF64:
    """Convert a MapGraph to a persisted dictionary of NumPy arrays."""
    node_positions = graph.node_positions - offset if offset is not None else graph.node_positions
    map_dict: MapSample[Any] = {
        "map_num_nodes": graph.num_nodes,
        "map_num_edges": graph.num_edges,
        "map_node_positions": node_positions.astype(dtype, copy=False),
        "map_edge_indices": np.ascontiguousarray(graph.edge_indices.T, dtype=np.int32),
        "map_node_types": graph.node_types,
        "map_edge_types": graph.edge_types,
    }
    return map_dict


def scene_sample_to_parts(
    sample: SceneSample[Any],
    *,
    feature_columns: Sequence[str],
    start_frame: int = 0,
) -> tuple[pl.DataFrame, int, int]:
    """Convert a persisted scene sample back into a Polars DataFrame.

    Parameters
    ----------
    sample : SceneSample[Any]
        The persisted scene data from `scene_to_numpy_dict`.
    feature_columns : Sequence[str]
        The list of feature column names in the order they appear in the input
        and target features.
    start_frame : int, optional
        The frame number corresponding to the first column of the input features,
        by default 0.

    Returns
    -------
    tuple[pl.DataFrame, int, int]
        Reconstructed DataFrame of agent states, along with input and output
        lengths.

    """
    input_features = sample["input_features"]
    target_features = sample["target_features"]
    input_mask = sample["input_mask"]
    target_mask = sample["target_mask"]
    agent_types = sample["agent_types"]
    offset = sample["global_origin"]

    input_len = input_features.shape[1]
    output_len = target_features.shape[1]

    features = np.concatenate([input_features, target_features], axis=1)
    mask = np.concatenate([input_mask, target_mask], axis=1)

    rows: list[dict[str, int | float]] = []
    for i in range(features.shape[0]):
        for t in range(features.shape[1]):
            if not mask[i, t]:
                continue

            row: dict[str, int | float] = {
                "frame": start_frame + t,
                "id": i,
                "agent_category": int(agent_types[i]),
            }

            for j, name in enumerate(feature_columns):
                v = features[i, t, j]
                if name == "x":
                    v += offset[0]
                elif name == "y":
                    v += offset[1]
                row[name] = v.item() if isinstance(v, np.generic) else v

            rows.append(row)

    return (
        pl.DataFrame(rows, schema_overrides={"agent_category": pl.Int32}).sort(["frame", "id"]),
        input_len,
        output_len,
    )
