# pyright: standard
# ruff: noqa: E402

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from dronalize.core.maps import MapGraph
from dronalize.core.scene import CANONICAL, Scene
from dronalize.io import ExportConfig

pytest.importorskip("torch")
pytest.importorskip("streaming")

from dronalize.io.adapters import MDSTorchDataset
from dronalize.io.backends.mds import MDSDatasetWriter
from dronalize.io.encoding import encode_map_from_scene, encode_scene_record
from dronalize.io.readers.mds import MDSReader
from dronalize.processing import LoaderConfig

if TYPE_CHECKING:
    from pathlib import Path


def _scene() -> Scene:
    graph = MapGraph(
        node_positions=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        edge_indices=np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32),
        node_types=np.array([1, 2, 3], dtype=np.int32),
        edge_types=np.array([4, 5, 6], dtype=np.int32),
    )
    return Scene(
        frame=pl.DataFrame({
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
        scene_number=7,
        input_len=2,
        output_len=1,
        schema=CANONICAL,
        sample_time=1.0,
        map_key="toy-map",
        map_resolver=lambda _scene, graph=graph: graph,
    )


def _loader_config() -> LoaderConfig:
    return LoaderConfig(input_len=2, output_len=1, sample_time=1.0)


def _writer_config() -> ExportConfig:
    return ExportConfig.create("canonical", precision="float32", recenter_positions=True)


def _expected_sample() -> dict[str, Any]:
    scene = _scene()
    writer_config = _writer_config()
    scene_sample = encode_scene_record(
        scene,
        dtype=writer_config.float_dtype,
        recenter_position=writer_config.recenter_positions,
        trajectory_schema=writer_config.trajectory_schema,
    )
    map_sample = encode_map_from_scene(
        scene,
        dtype=writer_config.float_dtype,
        offset=scene_sample["position_offset"],
        return_empty=True,
    )
    return {
        "scene_number": int(scene_sample["scene_number"]),
        "position_offset": scene_sample["position_offset"],
        "agent_types": scene_sample["agent_types"],
        "features": scene_sample["features"],
        "mask": scene_sample["mask"],
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
    reader = MDSReader(path=output_dir, split="all")
    assert len(reader) == 1
    sample = reader[0]
    return {
        "scene_number": int(sample.scene_number),
        "position_offset": sample.position_offset.astype(np.float64, copy=False),
        "agent_types": sample.agent_types.astype(np.int32, copy=False),
        "features": np.concatenate((sample.input_features, sample.output_features), axis=1),
        "mask": np.concatenate((sample.input_mask, sample.output_mask), axis=1),
        "map_node_positions": sample.map_node_positions.astype(np.float32, copy=False),
        "map_edge_indices": sample.map_edge_indices.astype(np.int32, copy=False),
        "map_node_types": sample.map_node_types.astype(np.int32, copy=False),
        "map_edge_types": sample.map_edge_types.astype(np.int32, copy=False),
    }


def _read_mds_torch_sample(output_dir: Path) -> dict[str, Any]:
    dataset = MDSTorchDataset(path=output_dir, split="all")
    assert len(dataset) == 1
    sample = dataset[0]
    features = np.concatenate(
        (
            sample.input_features.numpy().astype(np.float32, copy=False),
            sample.output_features.numpy().astype(np.float32, copy=False),
        ),
        axis=1,
    )
    mask = np.concatenate(
        (
            sample.input_mask.numpy().astype(bool, copy=False),
            sample.output_mask.numpy().astype(bool, copy=False),
        ),
        axis=1,
    )
    return {
        "scene_number": int(sample.scene_number),
        "position_offset": sample.position_offset.numpy().astype(np.float64, copy=False),
        "agent_types": sample.agent_types.numpy().astype(np.int32, copy=False),
        "features": features,
        "mask": mask,
        "map_node_positions": sample.map_node_positions.numpy().astype(np.float32, copy=False),
        "map_edge_indices": sample.map_edge_indices.numpy().astype(np.int32, copy=False),
        "map_node_types": sample.map_node_types.numpy().astype(np.int32, copy=False),
        "map_edge_types": sample.map_edge_types.numpy().astype(np.int32, copy=False),
    }


def _read_mds_torch_iterable_sample(output_dir: Path) -> dict[str, Any]:
    dataset = MDSTorchDataset(path=output_dir, split="all", batch_size=1)
    assert len(dataset) == 1
    sample = next(iter(dataset))
    features = np.concatenate(
        (
            sample.input_features.numpy().astype(np.float32, copy=False),
            sample.output_features.numpy().astype(np.float32, copy=False),
        ),
        axis=1,
    )
    mask = np.concatenate(
        (
            sample.input_mask.numpy().astype(bool, copy=False),
            sample.output_mask.numpy().astype(bool, copy=False),
        ),
        axis=1,
    )
    return {
        "scene_number": int(sample.scene_number),
        "position_offset": sample.position_offset.numpy().astype(np.float64, copy=False),
        "agent_types": sample.agent_types.numpy().astype(np.int32, copy=False),
        "features": features,
        "mask": mask,
        "map_node_positions": sample.map_node_positions.numpy().astype(np.float32, copy=False),
        "map_edge_indices": sample.map_edge_indices.numpy().astype(np.int32, copy=False),
        "map_node_types": sample.map_node_types.numpy().astype(np.int32, copy=False),
        "map_edge_types": sample.map_edge_types.numpy().astype(np.int32, copy=False),
    }


def test_mds_roundtrip(tmp_path: Path) -> None:
    """MDS writer should roundtrip one scene through the framework-neutral reader."""
    output_dir = tmp_path / "mds"
    writer = MDSDatasetWriter(
        output_dir,
        config=_writer_config(),
        loader_config=_loader_config(),
        source_trajectory_schema=CANONICAL,
        splits=None,
        parallel=False,
        has_map=True,
    )

    assert writer.write(_scene())
    writer.finish_local()
    writer.finish_final()
    _assert_sample_equal(_read_mds_sample(output_dir), _expected_sample())


def test_mds_torch_dataset_roundtrip(tmp_path: Path) -> None:
    """MDS Torch adapter should roundtrip one scene through the produced shard files."""
    output_dir = tmp_path / "mds"
    writer = MDSDatasetWriter(
        output_dir,
        config=_writer_config(),
        loader_config=_loader_config(),
        source_trajectory_schema=CANONICAL,
        splits=None,
        parallel=False,
        has_map=True,
    )

    assert writer.write(_scene())
    writer.finish_local()
    writer.finish_final()
    _assert_sample_equal(_read_mds_torch_sample(output_dir), _expected_sample())


def test_mds_torch_iterable_dataset_roundtrip(tmp_path: Path) -> None:
    """MDS Torch adapter should also roundtrip correctly through iterable access."""
    output_dir = tmp_path / "mds"
    writer = MDSDatasetWriter(
        output_dir,
        config=_writer_config(),
        loader_config=_loader_config(),
        source_trajectory_schema=CANONICAL,
        splits=None,
        parallel=False,
        has_map=True,
    )

    assert writer.write(_scene())
    writer.finish_local()
    writer.finish_final()
    _assert_sample_equal(_read_mds_torch_iterable_sample(output_dir), _expected_sample())
