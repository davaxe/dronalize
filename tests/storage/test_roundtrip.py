from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import zarr
from streaming.base.local import LocalDataset

from dronalize._internal._zarr_typing import read_typed_array, require_array  # noqa: PLC2701
from dronalize.config import LoaderConfig, WriterConfig
from dronalize.maps.graph import MapGraph
from dronalize.scene import CANONICAL_V1, Scene
from dronalize.storage.encoding import encode_map_from_scene, scene_to_numpy_dict
from dronalize.storage.writers.mds import MDSSceneWriter
from dronalize.storage.writers.zarr import ZarrSceneWriter


def _scene() -> Scene:
    graph = MapGraph(
        node_positions=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        edge_indices=np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32),
        node_types=np.array([1, 2, 3], dtype=np.int32),
        edge_types=np.array([4, 5, 6], dtype=np.int32),
    )
    return Scene(
        inner=pl.DataFrame({
            "frame": [0, 1, 2, 0, 1, 2],
            "id": [10, 10, 10, 20, 20, 20],
            "x": [0.0, 1.0, 2.0, 0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            "vx": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "vy": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "ax": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "ay": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0, 1.57, 1.57, 1.57],
            "agent_category": [1, 1, 1, 2, 2, 2],
        }),
        number=7,
        input_len=2,
        output_len=1,
        schema=CANONICAL_V1,
        sample_time=1.0,
        map_key="toy-map",
        map_resolver=lambda _scene, graph=graph: graph,
    )


def _loader_config() -> LoaderConfig:
    return LoaderConfig(input_len=2, output_len=1, sample_time=1.0)


def _writer_config() -> WriterConfig:
    return WriterConfig.create("canonical", precision="float32", offset_positions=True)


def _expected_sample() -> dict[str, Any]:
    scene = _scene()
    writer_config = _writer_config()
    scene_sample = scene_to_numpy_dict(
        scene,
        dtype=writer_config.float_dtype,
        offset_position=writer_config.offset_positions,
        scene_schema=writer_config.scene_schema,
    )
    map_sample = encode_map_from_scene(
        scene,
        dtype=writer_config.float_dtype,
        offset=scene_sample["global_origin"],
        return_empty=True,
    )
    return {
        "scene_number": int(scene_sample["scene_number"]),
        "num_agents": int(scene_sample["num_agents"]),
        "global_origin": scene_sample["global_origin"],
        "agent_types": scene_sample["agent_types"],
        "features": scene_sample["features"],
        "mask": scene_sample["mask"],
        "map_num_nodes": int(map_sample["map_num_nodes"]),
        "map_num_edges": int(map_sample["map_num_edges"]),
        "map_node_positions": map_sample["map_node_positions"],
        "map_edge_indices": map_sample["map_edge_indices"],
        "map_node_types": map_sample["map_node_types"],
        "map_edge_types": map_sample["map_edge_types"],
    }


def _assert_sample_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for key, actual_value in actual.items():
        expected_value = expected[key]
        if isinstance(actual_value, np.ndarray):
            assert isinstance(expected_value, np.ndarray)
            if np.issubdtype(actual_value.dtype, np.floating):
                np.testing.assert_allclose(actual_value, expected_value)
            else:
                np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def _read_mds_sample(output_dir: Path) -> dict[str, Any]:
    dataset = LocalDataset(local=str(output_dir), split="all")
    assert len(dataset) == 1
    sample = dataset.get_item(0)
    return {
        "scene_number": int(sample["scene_number"]),
        "num_agents": int(sample["num_agents"]),
        "global_origin": np.asarray(sample["global_origin"], dtype=np.float64),
        "agent_types": np.asarray(sample["agent_types"], dtype=np.int32),
        "features": np.asarray(sample["features"], dtype=np.float32),
        "mask": np.asarray(sample["mask"], dtype=bool),
        "map_num_nodes": int(sample["map_num_nodes"]),
        "map_num_edges": int(sample["map_num_edges"]),
        "map_node_positions": np.asarray(sample["map_node_positions"], dtype=np.float32),
        "map_edge_indices": np.asarray(sample["map_edge_indices"], dtype=np.int32),
        "map_node_types": np.asarray(sample["map_node_types"], dtype=np.int32),
        "map_edge_types": np.asarray(sample["map_edge_types"], dtype=np.int32),
    }


def _read_zarr_sample(output_dir: Path) -> dict[str, Any]:
    root = zarr.open_group(output_dir / "all.zarr", mode="r")
    pointers = read_typed_array(root, "meta/pointers", dtype=np.int64)
    counts = read_typed_array(root, "meta/counts", dtype=np.int32)
    origins = read_typed_array(root, "meta/global_origin", dtype=np.float64)

    assert pointers.shape == (1, 4)
    assert counts.shape == (1, 3)
    assert origins.shape == (1, 2)

    scene_number, agent_offset, node_offset, edge_offset = pointers[0]
    num_agents, num_nodes, num_edges = counts[0]

    agent_types_array = require_array(root, "agent/agent_types")
    features_array = require_array(root, "agent/features")
    mask_array = require_array(root, "agent/mask")
    map_node_positions_array = require_array(root, "map/node_positions")
    map_edge_indices_array = require_array(root, "map/edge_indices")
    map_node_types_array = require_array(root, "map/node_types")
    map_edge_types_array = require_array(root, "map/edge_types")

    return {
        "scene_number": scene_number,
        "num_agents": num_agents,
        "global_origin": origins[0],
        "agent_types": agent_types_array[agent_offset : agent_offset + num_agents],
        "features": features_array[agent_offset : agent_offset + num_agents],
        "mask": mask_array[agent_offset : agent_offset + num_agents],
        "map_num_nodes": num_nodes,
        "map_num_edges": num_edges,
        "map_node_positions": map_node_positions_array[node_offset : node_offset + num_nodes],
        "map_edge_indices": map_edge_indices_array[edge_offset : edge_offset + num_edges],
        "map_node_types": map_node_types_array[node_offset : node_offset + num_nodes],
        "map_edge_types": map_edge_types_array[edge_offset : edge_offset + num_edges],
    }


def test_zarr_roundtrip(tmp_path: Path) -> None:
    """Test consitent data between writing and reading a scene through Zarr format."""
    writer = ZarrSceneWriter(
        Path(tmp_path),
        config=_writer_config(),
        loader_config=_loader_config(),
        splits=None,
        parallel=False,
        has_map=True,
    )
    assert writer.write(_scene())
    writer.finish_local()
    writer.finish_final()
    _assert_sample_equal(_read_zarr_sample(Path(tmp_path)), _expected_sample())


def test_mds_roundtrip(tmp_path: Path) -> None:
    """Test consitent data between writing and reading a scene through MDS format."""
    """MDS writer should roundtrip one scene through the produced shard files."""
    output_dir = tmp_path / "mds"
    writer = MDSSceneWriter(
        output_dir,
        config=_writer_config(),
        loader_config=_loader_config(),
        splits=None,
        parallel=False,
        has_map=True,
    )

    assert writer.write(_scene())
    writer.finish_local()
    writer.finish_final()
    _assert_sample_equal(_read_mds_sample(output_dir), _expected_sample())
