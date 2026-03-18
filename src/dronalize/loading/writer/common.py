from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, TypeVar, overload

import numpy as np
import numpy.typing as npt
import polars as pl

from dronalize.categories import AgentCategory

if TYPE_CHECKING:
    from dronalize.maps.graph import MapGraph
    from dronalize.scene import Scene, SceneSchema

FloatScalar = TypeVar("FloatScalar", np.float32, np.float64)
FloatDType = type[np.float32] | type[np.float64]


PLACEHOLDER_FLOAT = np.zeros((1,), dtype=np.float64)
PLACEHOLDER_I32 = np.zeros((1,), dtype=np.int32)


class NumpyMapGraphDict(TypedDict, Generic[FloatScalar]):
    """TypedDict for the output of `map_graph_to_numpy`."""

    map_num_nodes: int
    map_num_edges: int
    map_node_positions: npt.NDArray[FloatScalar]
    map_edge_indices: npt.NDArray[np.int32]
    map_node_types: npt.NDArray[np.int32]
    map_edge_types: npt.NDArray[np.int32]


class NumpySceneDict(TypedDict, Generic[FloatScalar]):
    """TypedDict for the output of `convert_to_numpy_dict`."""

    scene_number: int
    global_origin: npt.NDArray[np.float64]
    num_nodes: int
    type: npt.NDArray[np.int32]
    input_features: npt.NDArray[FloatScalar]
    target_features: npt.NDArray[FloatScalar]
    input_mask: npt.NDArray[np.bool_]
    target_mask: npt.NDArray[np.bool_]


NumpySceneDictF32 = NumpySceneDict[np.float32]
NumpySceneDictF64 = NumpySceneDict[np.float64]
NumpyMapGraphDictF32 = NumpyMapGraphDict[np.float32]
NumpyMapGraphDictF64 = NumpyMapGraphDict[np.float64]


@overload
def scene_to_numpy_dict(
    scene: Scene,
    *,
    dtype: type[np.float32],
    offset_position: bool = True,
    scene_schema: SceneSchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> NumpySceneDictF32: ...


@overload
def scene_to_numpy_dict(
    scene: Scene,
    *,
    dtype: type[np.float64],
    offset_position: bool = True,
    scene_schema: SceneSchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> NumpySceneDictF64: ...


def scene_to_numpy_dict(
    scene: Scene,
    *,
    dtype: FloatDType,
    offset_position: bool = True,
    scene_schema: SceneSchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> NumpySceneDictF32 | NumpySceneDictF64:
    """Convert a Scene to a numpy representation compatible with Pytorch."""
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
        type_array = np.array(
            [category_mapping.get(AgentCategory(c), -1) for c in raw_categories],
            dtype=np.int32,
        )
    else:
        type_array = raw_categories.astype(np.int32)

    scene_dict: NumpySceneDict[Any] = {
        "scene_number": scene.number,
        "global_origin": offset,
        "num_nodes": num_agents,
        "type": type_array,
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
) -> NumpyMapGraphDictF32: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[True],
) -> NumpyMapGraphDictF64: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float32],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[False] = False,
) -> NumpyMapGraphDictF32 | None: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[False] = False,
) -> NumpyMapGraphDictF64 | None: ...


def encode_map_from_scene(
    scene: Scene,
    dtype: FloatDType,
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: bool = False,
) -> NumpyMapGraphDictF32 | NumpyMapGraphDictF64 | None:
    """Resolve the map graph from the scene and convert to numpy.

    Parameters
    ----------
    scene : Scene
        The scene to resolve the map from.
    dtype : FloatDType
        The floating point dtype to use for the node positions.
    offset : npt.NDArray[np.float64] | None
        The offset to apply to the node positions, or `None` to keep original
        positions.
    return_empty : bool, optional
        If `True`, return an empty map graph dictionary when the scene has no
        map, instead of returning `None`.

    Returns
    -------
    NumpyMapGraphDict[np.float32] | NumpyMapGraphDict[np.float64] | None
        The converted map graph as a dictionary of NumPy arrays, or `None` if
        the scene has no map.
    """
    graph = scene.resolve_map()
    if graph is None:
        if not return_empty:
            return None
        map_dict: NumpyMapGraphDict[Any] = {
            "map_num_nodes": 0,
            "map_num_edges": 0,
            "map_node_positions": np.zeros((0, 2), dtype=dtype),
            "map_edge_indices": np.zeros((2, 0), dtype=np.int32),
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
) -> NumpyMapGraphDictF32: ...


@overload
def map_graph_to_numpy(
    graph: MapGraph,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None = None,
) -> NumpyMapGraphDictF64: ...


def map_graph_to_numpy(
    graph: MapGraph,
    dtype: FloatDType,
    offset: npt.NDArray[np.float64] | None = None,
) -> NumpyMapGraphDictF32 | NumpyMapGraphDictF64:
    """Convert the `MapGraph` to a dictionary of NumPy arrays."""
    node_positions = graph.node_positions - offset if offset is not None else graph.node_positions
    map_dict: NumpyMapGraphDict[Any] = {
        "map_num_nodes": graph.num_nodes,
        "map_num_edges": graph.num_edges,
        "map_node_positions": node_positions.astype(dtype, copy=False),
        "map_edge_indices": graph.edge_indices,
        "map_node_types": graph.node_types,
        "map_edge_types": graph.edge_types,
    }
    return map_dict
