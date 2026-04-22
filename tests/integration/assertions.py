from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.core.maps import MapGraph
    from dronalize.core.scene import Scene
    from dronalize.io import SceneRecord
    from tests.integration.catalog import DatasetCase


def _assert_shape(array: npt.NDArray[Any], expected: tuple[int, ...], name: str) -> None:
    assert array.shape == expected, f"{name} must have shape {expected}, got {array.shape}"


def _assert_ndim(array: npt.NDArray[Any], expected: int, name: str) -> None:
    assert array.ndim == expected, f"{name} must be {expected}D, got {array.ndim}D"


def _assert_dtype(array: npt.NDArray[Any], expected: npt.DTypeLike, name: str) -> None:
    assert array.dtype == np.dtype(expected), (
        f"{name} must have dtype {expected}, got {array.dtype}"
    )


def _assert_finite(array: npt.NDArray[np.floating[Any]], name: str) -> None:
    assert np.isfinite(array).all(), f"{name} contains NaN/Inf"


def _assert_edge_index_bounds(
    edge_indices: npt.NDArray[np.int32], num_nodes: int, *, name: str
) -> None:
    if edge_indices.shape[1] == 0:
        return
    assert num_nodes > 0, f"{name} has edges but no nodes"
    assert int(edge_indices.min()) >= 0, f"{name} contains negative node index"
    assert int(edge_indices.max()) < num_nodes, f"{name} contains out-of-bounds node index"


def assert_basic_scene_sanity(scene: Scene) -> None:
    assert scene.scene_number >= 0, "scene_number must be non-negative"
    assert scene.history_frames >= 0, "history_frames must be >= 0"
    assert scene.future_frames >= 0, "future_frames must be >= 0"
    assert scene.history_frames + scene.future_frames > 0, "expected positive total horizon"

    frame = scene.frame
    assert frame.height > 0, "scene frame is empty"

    required_columns = ("frame", "id", "x", "y", "agent_category")
    missing = [col for col in required_columns if col not in frame.columns]
    assert not missing, f"missing required columns: {missing}"

    null_counts = frame.select([
        pl.col(col).is_null().sum().alias(col) for col in required_columns
    ]).row(0, named=True)
    assert all(count == 0 for count in null_counts.values()), (
        f"nulls found in required columns: {null_counts}"
    )

    invalid_xy = frame.filter(~pl.col("x").is_finite() | ~pl.col("y").is_finite())
    assert invalid_xy.is_empty(), "found non-finite coordinates"

    assert int(frame["id"].n_unique()) > 0, "no agents found"
    assert int(frame["frame"].n_unique()) > 0, "no frames found"
    if frame.height > 1:
        x_span, y_span = frame.select(
            (pl.col("x").max() - pl.col("x").min()).alias("x_span"),
            (pl.col("y").max() - pl.col("y").min()).alias("y_span"),
        ).row(0)
        assert float(x_span) > 0.0 or float(y_span) > 0.0, (
            "degenerate scene with zero coordinate span"
        )

    duplicate_frames = frame.group_by("id", "frame").agg(count=pl.len()).filter(pl.col("count") > 1)
    assert duplicate_frames.is_empty(), "found duplicate frames for some agents"


def assert_basic_map_sanity(graph: MapGraph | None, *, expect_map: bool) -> None:
    if not expect_map:
        if graph is None:
            return
        assert graph.num_nodes == 0, "expected no map nodes for non-map scene"
        assert graph.num_edges == 0, "expected no map edges for non-map scene"
        return

    assert graph is not None, "scene advertises map support but resolved map is None"

    _assert_ndim(graph.node_positions, 2, "map node_positions")
    _assert_ndim(graph.edge_indices, 2, "map edge_indices")
    _assert_ndim(graph.node_types, 1, "map node_types")
    _assert_ndim(graph.edge_types, 1, "map edge_types")

    assert graph.node_positions.shape[1] == 2, "map node_positions must have shape (N, 2)"
    assert graph.edge_indices.shape[0] == 2, "map edge_indices must have shape (2, E)"

    assert graph.node_positions.shape[0] == graph.node_types.shape[0], (
        "map node count mismatch between node_positions and node_types"
    )
    assert graph.edge_indices.shape[1] == graph.edge_types.shape[0], (
        "map edge count mismatch between edge_indices and edge_types"
    )

    _assert_dtype(graph.node_positions, np.float64, "map node_positions")
    _assert_dtype(graph.edge_indices, np.int32, "map edge_indices")
    _assert_dtype(graph.node_types, np.int32, "map node_types")
    _assert_dtype(graph.edge_types, np.int32, "map edge_types")

    if graph.num_nodes > 0:
        _assert_finite(graph.node_positions, "map node_positions")

    _assert_edge_index_bounds(graph.edge_indices, graph.num_nodes, name="map edge_indices")


def assert_record_sanity(record: SceneRecord, scene: Scene) -> None:
    assert record.scene_number == scene.scene_number, "scene_number mismatch"

    _assert_shape(record.position_offset, (2,), "position_offset")
    _assert_finite(record.position_offset, "position_offset")

    n_agents = int(record.agent_types.shape[0])
    history_frames = scene.history_frames
    future_frames = scene.future_frames
    input_feature_dim = int(record.input_features.shape[2])
    output_feature_dim = int(record.output_features.shape[2])

    _assert_shape(record.passed_agent_mask, (n_agents,), "passed_agent_mask")
    _assert_dtype(record.passed_agent_mask, np.bool_, "passed_agent_mask")

    _assert_ndim(record.input_features, 3, "input_features")
    _assert_ndim(record.output_features, 3, "output_features")
    _assert_ndim(record.input_mask, 2, "input_mask")
    _assert_ndim(record.output_mask, 2, "output_mask")

    assert record.input_features.shape[0] == n_agents, "input_features agent dim mismatch"
    assert record.output_features.shape[0] == n_agents, "output_features agent dim mismatch"
    assert record.input_mask.shape[0] == n_agents, "input_mask agent dim mismatch"
    assert record.output_mask.shape[0] == n_agents, "output_mask agent dim mismatch"

    assert record.input_features.shape[1] == history_frames, (
        "input_features time dim must match scene.history_frames"
    )
    assert record.output_features.shape[1] == future_frames, (
        "output_features time dim must match scene.future_frames"
    )
    assert record.input_mask.shape[1] == history_frames, (
        "input_mask time dim must match scene.history_frames"
    )
    assert record.output_mask.shape[1] == future_frames, (
        "output_mask time dim must match scene.future_frames"
    )

    assert input_feature_dim > 0, "feature dimension must be > 0"
    assert input_feature_dim == output_feature_dim, (
        "input_features and output_features feature dims must match"
    )

    _assert_dtype(record.input_mask, np.bool_, "input_mask")
    _assert_dtype(record.output_mask, np.bool_, "output_mask")

    if record.input_mask.any():
        assert np.isfinite(record.input_features[record.input_mask]).all(), (
            "input_features contain NaN/Inf at valid mask positions"
        )
    if record.output_mask.any():
        assert np.isfinite(record.output_features[record.output_mask]).all(), (
            "output_features contain NaN/Inf at valid mask positions"
        )

    if n_agents > 0:
        assert record.passed_agent_mask.any(), "no passing agents in encoded record"

    map_num_nodes = record.map_node_positions.shape[0]
    map_num_edges = record.map_edge_indices.shape[1]

    _assert_ndim(record.map_node_positions, 2, "map_node_positions")
    _assert_ndim(record.map_edge_indices, 2, "map_edge_indices")
    assert record.map_node_positions.shape[1] == 2, "map_node_positions must have shape (N, 2)"
    assert record.map_edge_indices.shape[0] == 2, "map_edge_indices must have shape (2, E)"
    _assert_shape(record.map_node_types, (map_num_nodes,), "map_node_types")
    _assert_shape(record.map_edge_types, (map_num_edges,), "map_edge_types")

    _assert_finite(record.map_node_positions, "map_node_positions")
    _assert_edge_index_bounds(record.map_edge_indices, map_num_nodes, name="map_edge_indices")

    has_record_map = map_num_nodes > 0 or map_num_edges > 0
    if not scene.has_map():
        assert not has_record_map, "encoded map payload present for scene without map"


def save_scene_artifacts(
    scene: Scene, graph: MapGraph | None, out_dir: Path, case: DatasetCase
) -> None:
    """Save scene artifacts like trajectories and maps for debugging."""
    try:
        from dronalize.plot import plot_scene  # noqa: PLC0415
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    frame = scene.frame.select("frame", "id", "x", "y")
    _ = plot_scene(scene, show_map=False, save_path=out_dir / "trajectories.html", aspect="equal")

    if graph is not None and (graph.num_nodes > 0 or graph.num_edges > 0):
        _ = plot_scene(scene, save_path=out_dir / "overlay.html", aspect="equal")

    summary_frame = frame.select(
        pl.col("x").min().alias("x_min"),
        pl.col("x").max().alias("x_max"),
        pl.col("y").min().alias("y_min"),
        pl.col("y").max().alias("y_max"),
        pl.col("frame").min().alias("frame_min"),
        pl.col("frame").max().alias("frame_max"),
    )
    x_min, x_max, y_min, y_max, frame_min, frame_max = summary_frame.row(0)

    summary: dict[str, object] = {
        "dataset": case.dataset,
        "scene_number": scene.scene_number,
        "rows": frame.height,
        "agents": int(frame["id"].n_unique()),
        "frame_min": int(frame_min),
        "frame_max": int(frame_max),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "has_map": graph is not None,
        "map_num_nodes": int(graph.num_nodes) if graph is not None else 0,
        "map_num_edges": int(graph.num_edges) if graph is not None else 0,
    }

    if graph is not None and graph.num_nodes > 0:
        summary["map_x_min"] = float(np.min(graph.node_positions[:, 0]))
        summary["map_x_max"] = float(np.max(graph.node_positions[:, 0]))
        summary["map_y_min"] = float(np.min(graph.node_positions[:, 1]))
        summary["map_y_max"] = float(np.max(graph.node_positions[:, 1]))

    with (out_dir / "summary.json").open("w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2, sort_keys=True)
