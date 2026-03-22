# pyright: standard

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import zarr
from streaming import StreamingDataset

from dronalize.categories import DatasetSplit
from dronalize.config import LoaderConfig, WriterConfig
from dronalize.maps.graph import MapGraph
from dronalize.scene import CANONICAL_V1, Scene
from dronalize.storage.encoding import encode_map_from_scene, scene_to_numpy_dict
from dronalize.storage.spec import (
    FORMAT_VERSION,
    StorageMapSampleF32,
    StorageMapSampleF64,
    StorageSceneSampleF32,
    StorageSceneSampleF64,
    read_manifest,
)
from dronalize.storage.writers.mds import MDSSceneWriter
from dronalize.storage.writers.zarr import ZarrSceneWriter


def _scene(
    *,
    scene_number: int = 11,
    position_shift: tuple[float, float] = (0.0, 0.0),
    with_map: bool = True,
    split_assignment: DatasetSplit | None = DatasetSplit.TRAIN,
) -> Scene:
    shift_x, shift_y = position_shift
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
            "x": [
                0.0 + shift_x,
                1.0 + shift_x,
                2.0 + shift_x,
                0.0 + shift_x,
                0.0 + shift_x,
                0.0 + shift_x,
            ],
            "y": [
                0.0 + shift_y,
                0.0 + shift_y,
                0.0 + shift_y,
                0.0 + shift_y,
                1.0 + shift_y,
                2.0 + shift_y,
            ],
            "vx": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "vy": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            "ax": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "ay": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0, 1.57, 1.57, 1.57],
            "agent_category": [1, 1, 1, 2, 2, 2],
        }),
        number=scene_number,
        input_len=2,
        output_len=1,
        schema=CANONICAL_V1,
        sample_time=1.0,
        map_key="toy-map" if with_map else None,
        map_resolver=(lambda _scene, graph=graph: graph) if with_map else None,
        split_assignment=split_assignment,
    )


def _writer_config() -> WriterConfig:
    return WriterConfig.create("canonical", precision="float32", offset_positions=False)


def _loader_config() -> LoaderConfig:
    return LoaderConfig(input_len=2, output_len=1, sample_time=1.0)


def _expected_scene_payload(scene: Scene) -> StorageSceneSampleF32 | StorageSceneSampleF64:
    config = _writer_config()
    return scene_to_numpy_dict(
        scene,
        dtype=config.float_dtype,
        offset_position=config.offset_positions,
        scene_schema=config.scene_schema,
    )


def _expected_map_payload(scene: Scene) -> StorageMapSampleF32 | StorageMapSampleF64:
    config = _writer_config()
    scene_payload = _expected_scene_payload(scene)
    return encode_map_from_scene(
        scene,
        dtype=config.float_dtype,
        offset=scene_payload["global_origin"] if config.offset_positions else None,
        return_empty=True,
    )


def _assert_manifest(manifest: dict[str, object]) -> None:
    assert manifest == {
        "format_version": FORMAT_VERSION,
        "scene_schema": "canonical",
        "feature_columns": ["x", "y", "vx", "vy", "ax", "ay", "yaw"],
        "input_len": 2,
        "output_len": 1,
        "precision": "float32",
        "offset_positions": False,
        "has_map": True,
    }


def _assert_sample_matches_expected(
    actual_scene: dict[str, np.ndarray | int],
    actual_map: dict[str, np.ndarray | int],
    expected_scene: dict[str, np.ndarray | int],
    expected_map: dict[str, np.ndarray | int],
) -> None:
    assert int(actual_scene["scene_number"]) == int(expected_scene["scene_number"])
    assert int(actual_scene["num_agents"]) == int(expected_scene["num_agents"])
    np.testing.assert_array_equal(actual_scene["global_origin"], expected_scene["global_origin"])
    np.testing.assert_array_equal(actual_scene["agent_types"], expected_scene["agent_types"])
    np.testing.assert_array_equal(actual_scene["input_features"], expected_scene["input_features"])
    np.testing.assert_array_equal(
        actual_scene["target_features"],
        expected_scene["target_features"],
    )
    np.testing.assert_array_equal(actual_scene["input_mask"], expected_scene["input_mask"])
    np.testing.assert_array_equal(actual_scene["target_mask"], expected_scene["target_mask"])

    assert int(actual_map["map_num_nodes"]) == int(expected_map["map_num_nodes"])
    assert int(actual_map["map_num_edges"]) == int(expected_map["map_num_edges"])
    np.testing.assert_array_equal(
        actual_map["map_node_positions"],
        expected_map["map_node_positions"],
    )
    np.testing.assert_array_equal(actual_map["map_edge_indices"], expected_map["map_edge_indices"])
    np.testing.assert_array_equal(actual_map["map_node_types"], expected_map["map_node_types"])
    np.testing.assert_array_equal(actual_map["map_edge_types"], expected_map["map_edge_types"])


def _read_zarr_scene(root: zarr.Group, index: int) -> dict[str, Any]:
    agent_start = int(root["meta/agent_ptr"][index])
    num_agents = int(root["meta/num_agents"][index])
    agent_end = agent_start + num_agents

    return {
        "scene_number": int(root["meta/scene_number"][index]),
        "global_origin": root["meta/global_origin"][index],
        "num_agents": num_agents,
        "agent_types": root["agent/agent_types"][agent_start:agent_end],
        "input_features": root["agent/input_features"][agent_start:agent_end],
        "target_features": root["agent/target_features"][agent_start:agent_end],
        "input_mask": root["agent/input_mask"][agent_start:agent_end],
        "target_mask": root["agent/target_mask"][agent_start:agent_end],
    }


def _read_zarr_map(root: zarr.Group, index: int) -> dict[str, Any]:
    node_start = int(root["meta/map_node_ptr"][index])
    edge_start = int(root["meta/map_edge_ptr"][index])
    num_nodes = int(root["meta/num_map_nodes"][index])
    num_edges = int(root["meta/num_map_edges"][index])
    node_end = node_start + num_nodes
    edge_end = edge_start + num_edges

    return {
        "map_num_nodes": num_nodes,
        "map_num_edges": num_edges,
        "map_node_positions": root["map/node_positions"][node_start:node_end],
        "map_edge_indices": root["map/edge_indices"][edge_start:edge_end],
        "map_node_types": root["map/node_types"][node_start:node_end],
        "map_edge_types": root["map/edge_types"][edge_start:edge_end],
    }


def _read_mds_scene(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_number": int(sample["scene_number"]),
        "global_origin": np.asarray(sample["global_origin"]),
        "num_agents": int(sample["num_agents"]),
        "agent_types": np.asarray(sample["agent_types"]),
        "input_features": np.asarray(sample["input_features"]),
        "target_features": np.asarray(sample["target_features"]),
        "input_mask": np.asarray(sample["input_mask"]).astype(bool),
        "target_mask": np.asarray(sample["target_mask"]).astype(bool),
    }


def _read_mds_map(sample: dict[str, Any]) -> dict[str, np.ndarray | int]:
    return {
        "map_num_nodes": int(sample["map_num_nodes"]),
        "map_num_edges": int(sample["map_num_edges"]),
        "map_node_positions": np.asarray(sample["map_node_positions"]),
        "map_edge_indices": np.asarray(sample["map_edge_indices"]),
        "map_node_types": np.asarray(sample["map_node_types"]),
        "map_edge_types": np.asarray(sample["map_edge_types"]),
    }


def test_zarr_roundtrip_persists_and_reads_back_multiple_scenes(tmp_path: Path) -> None:
    """Zarr exports should round-trip manifests, pointers, and per-scene arrays."""
    scenes = [
        _scene(scene_number=11),
        _scene(scene_number=12, position_shift=(10.0, -2.0)),
    ]
    writer = ZarrSceneWriter(
        tmp_path,
        config=_writer_config(),
        loader_config=_loader_config(),
        splits=(DatasetSplit.TRAIN,),
        parallel=False,
        has_map=True,
    )

    for scene in scenes:
        assert writer.write(scene, split=DatasetSplit.TRAIN) is True
    writer.finish_local()
    writer.finish_final()

    manifest = read_manifest(tmp_path / "train.zarr")
    _assert_manifest(manifest)

    root = zarr.open_group(tmp_path / "train.zarr", mode="r")
    np.testing.assert_array_equal(root["meta/scene_number"][:], np.array([11, 12], dtype=np.int32))
    np.testing.assert_array_equal(root["meta/num_agents"][:], np.array([2, 2], dtype=np.int32))
    np.testing.assert_array_equal(root["meta/num_map_nodes"][:], np.array([3, 3], dtype=np.int32))
    np.testing.assert_array_equal(root["meta/num_map_edges"][:], np.array([3, 3], dtype=np.int32))
    np.testing.assert_array_equal(root["meta/agent_ptr"][:], np.array([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(root["meta/map_node_ptr"][:], np.array([0, 3], dtype=np.int64))
    np.testing.assert_array_equal(root["meta/map_edge_ptr"][:], np.array([0, 3], dtype=np.int64))

    for index, scene in enumerate(scenes):
        expected_scene = _expected_scene_payload(scene)
        expected_map = _expected_map_payload(scene)
        actual_scene = _read_zarr_scene(root, index)
        actual_map = _read_zarr_map(root, index)
        _assert_sample_matches_expected(actual_scene, actual_map, expected_scene, expected_map)


def test_mds_roundtrip_persists_and_reads_back_multiple_scenes(tmp_path: Path) -> None:
    """MDS exports should round-trip manifests and per-sample payloads."""
    scenes = [
        _scene(scene_number=11),
        _scene(scene_number=12, position_shift=(10.0, -2.0), split_assignment=None),
    ]
    writer = MDSSceneWriter(
        tmp_path,
        config=_writer_config(),
        loader_config=_loader_config(),
        splits=None,
        parallel=False,
        has_map=True,
    )

    for scene in scenes:
        assert writer.write(scene) is True
    writer.finish_local()
    writer.finish_final()

    manifest = read_manifest(tmp_path)
    _assert_manifest(manifest)

    dataset = StreamingDataset(local=str(tmp_path), shuffle=False)
    assert len(dataset) == 2

    for index, scene in enumerate(scenes):
        sample = dataset[index]
        actual_scene = _read_mds_scene(sample)
        actual_map = _read_mds_map(sample)
        expected_scene = _expected_scene_payload(scene)
        expected_map = _expected_map_payload(scene)
        _assert_sample_matches_expected(actual_scene, actual_map, expected_scene, expected_map)
        assert int(sample["input_len"]) == 2
        assert int(sample["output_len"]) == 1
