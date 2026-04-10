"""Conversion helpers between scene objects and persisted tensor layouts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.typing as npt
import polars as pl

from dronalize.core.categories import AgentCategory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dronalize.core.map_graph import MapGraph
    from dronalize.core.scene import Scene, TrajectorySchema
    from dronalize.core.typing import FloatDType
    from dronalize.io.manifest import (
        MapRecord,
        MapRecordF32,
        MapRecordF64,
        SceneRecord,
        SceneRecordF32,
        SceneRecordF64,
    )


@overload
def encode_scene_record(
    scene: Scene,
    *,
    dtype: type[np.float32],
    recenter_position: bool = True,
    trajectory_schema: TrajectorySchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> SceneRecordF32: ...


@overload
def encode_scene_record(
    scene: Scene,
    *,
    dtype: type[np.float64],
    recenter_position: bool = True,
    trajectory_schema: TrajectorySchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> SceneRecordF64: ...


def encode_scene_record(
    scene: Scene,
    *,
    dtype: FloatDType,
    recenter_position: bool = True,
    trajectory_schema: TrajectorySchema | None = None,
    category_mapping: dict[AgentCategory, int] | None = None,
) -> SceneRecordF32 | SceneRecordF64:
    """Convert a scene to the persisted tensor representation used by writers.

    Parameters
    ----------
    scene : Scene
        Scene to encode.
    dtype : type[np.float32] or type[np.float64]
        Floating-point dtype used for the output feature tensors.
    recenter_position : bool, optional
        Whether to subtract the mean `x` and `y` position of the scene
        before storing the feature tensor. The applied offset is returned in
        `position_offset`.
    trajectory_schema : TrajectorySchema | None, optional
        Target schema to persist. When provided, `scene` is converted before
        encoding.
    category_mapping : dict[AgentCategory, int] | None, optional
        Optional mapping from internal agent categories to persisted integer
        labels. Unmapped categories are encoded as `-1`.

    Returns
    -------
    SceneRecordF32 or SceneRecordF64
        Dictionary of NumPy arrays containing the encoded scene tensors.
    """
    if trajectory_schema is not None:
        scene = scene.as_schema(trajectory_schema)

    data = scene.frame
    history_frames = scene.history_frames
    future_frames = scene.future_frames
    feature_columns = scene.schema.feature_columns()

    time_steps = history_frames + future_frames
    start_frame = data["frame"].min()
    unique_ids: list[int] = data["id"].unique().to_list()
    num_agents = len(unique_ids)
    id_to_idx_map = {uid: i for i, uid in enumerate(sorted(unique_ids))}
    df_indexed = data.with_columns([
        (pl.col("frame") - start_frame).alias("col_idx"),
        pl.col("id").replace_strict(id_to_idx_map).alias("row_idx"),
    ])

    if recenter_position:
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
            [category_mapping.get(AgentCategory(c), -1) for c in raw_categories], dtype=np.int32
        )
    else:
        agent_types = raw_categories.astype(np.int32)

    scene_dict: SceneRecord[Any] = {
        "scene_number": scene.scene_number,
        "position_offset": offset,
        "agent_types": agent_types,
        "features": features,
        "mask": mask,
    }
    return scene_dict


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float32],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[True],
) -> MapRecordF32: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[True],
) -> MapRecordF64: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float32],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[False] = False,
) -> MapRecordF32 | None: ...


@overload
def encode_map_from_scene(
    scene: Scene,
    dtype: type[np.float64],
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: Literal[False] = False,
) -> MapRecordF64 | None: ...


def encode_map_from_scene(
    scene: Scene,
    dtype: FloatDType,
    offset: npt.NDArray[np.float64] | None,
    *,
    return_empty: bool = False,
) -> MapRecordF32 | MapRecordF64 | None:
    """Resolve the scene map and convert it to the persisted map layout.

    Parameters
    ----------
    scene : Scene
        Scene whose attached map should be encoded.
    dtype : type[np.float32] or type[np.float64]
        Floating-point dtype used for persisted node coordinates.
    offset : ndarray of float64 or None
        Position offset to subtract from map node coordinates before
        persistence. This typically matches the `position_offset` returned by
        :func:`encode_scene_record`.
    return_empty : bool, optional
        Whether to emit the sentinel empty-map payload when `scene` has no
        map. When `False`, `None` is returned instead.

    Returns
    -------
    MapRecordF32 or MapRecordF64 or None
        Encoded map payload, sentinel empty payload, or `None` when the
        scene has no map and `return_empty` is `False`.
    """
    graph = scene.resolve_map()
    if graph is None:
        if not return_empty:
            return None
        map_dict: MapRecord[Any] = {
            "map_node_positions": np.full((1, 2), dtype=dtype, fill_value=np.nan),
            "map_edge_indices": np.full((2, 1), dtype=np.int32, fill_value=-1),
            "map_node_types": np.full((1,), dtype=np.int32, fill_value=-1),
            "map_edge_types": np.full((1,), dtype=np.int32, fill_value=-1),
        }
        return map_dict

    return _map_graph_to_numpy(graph, dtype=dtype, offset=offset)


@overload
def _map_graph_to_numpy(
    graph: MapGraph, dtype: type[np.float32], offset: npt.NDArray[np.float64] | None = None
) -> MapRecordF32: ...


@overload
def _map_graph_to_numpy(
    graph: MapGraph, dtype: type[np.float64], offset: npt.NDArray[np.float64] | None = None
) -> MapRecordF64: ...


def _map_graph_to_numpy(
    graph: MapGraph, dtype: FloatDType, offset: npt.NDArray[np.float64] | None = None
) -> MapRecordF32 | MapRecordF64:
    """Convert a `MapGraph` to a persisted dictionary of NumPy arrays."""
    node_positions = graph.node_positions - offset if offset is not None else graph.node_positions
    map_dict: MapRecord[Any] = {
        "map_node_positions": node_positions.astype(dtype, copy=False),
        "map_edge_indices": np.ascontiguousarray(graph.edge_indices, dtype=np.int32),
        "map_node_types": graph.node_types,
        "map_edge_types": graph.edge_types,
    }
    return map_dict


def scene_record_to_frame(
    sample: SceneRecord[Any], *, feature_columns: Sequence[str], start_frame: int = 0
) -> pl.DataFrame:
    """Convert a persisted scene sample back into a Polars DataFrame.

    Parameters
    ----------
    sample : SceneRecord[Any]
        The persisted scene data from `encode_scene_record`.
    feature_columns : Sequence[str]
        The list of feature column names in the order they appear in the input
        and target features.
    start_frame : int, optional
        The frame number corresponding to the first column of the input features,
        by default 0.

    Returns
    -------
    pl.DataFrame
        Reconstructed agent-state table with `frame`, `id`,
        `agent_category`, and the requested feature columns.

    """
    features = sample["features"]
    mask = sample["mask"]
    agent_types = sample["agent_types"]
    offset = sample["position_offset"]

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

    return pl.DataFrame(rows, schema_overrides={"agent_category": pl.Int32}).sort(["frame", "id"])
