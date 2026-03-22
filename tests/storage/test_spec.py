import numpy as np
import polars as pl

from dronalize.config import LoaderConfig, WriterConfig
from dronalize.maps.graph import MapGraph
from dronalize.scene import CANONICAL_V1, Scene
from dronalize.storage.encoding import encode_map_from_scene, scene_to_numpy_dict
from dronalize.storage.spec import FORMAT_VERSION, StorageManifest


def _scene(*, with_map: bool = True) -> Scene:
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
        map_key="toy-map" if with_map else None,
        map_resolver=(lambda _scene, graph=graph: graph) if with_map else None,
    )


def test_storage_manifest_reflects_resolved_writer_contract() -> None:
    """Manifest generation should mirror the resolved writer contract."""
    loader_config = LoaderConfig(input_len=2, output_len=1, sample_time=0.5)
    writer_config = WriterConfig.create("canonical", precision="float64", offset_positions=False)

    manifest = StorageManifest.from_configs(
        loader_config=loader_config,
        writer_config=writer_config,
        has_map=True,
    )

    assert manifest.format_version == FORMAT_VERSION
    assert manifest.scene_schema == "canonical"
    assert manifest.feature_columns == ("x", "y", "vx", "vy", "ax", "ay", "yaw")
    assert manifest.input_len == 2
    assert manifest.output_len == 1
    assert manifest.precision == "float64"
    assert manifest.offset_positions is False
    assert manifest.has_map is True


def test_scene_encoding_uses_new_logical_field_names() -> None:
    """Encoded scenes should use the new persisted field names."""
    sample = scene_to_numpy_dict(_scene(), dtype=np.float32)

    assert sample["num_agents"] == 2
    assert sample["agent_types"].tolist() == [1, 2]
    assert sample["input_features"].shape == (2, 2, 7)
    assert sample["target_features"].shape == (2, 1, 7)


def test_map_encoding_persists_edge_indices_in_append_safe_layout() -> None:
    """Encoded maps should transpose edge indices into an append-safe layout."""
    map_sample = encode_map_from_scene(_scene(), np.float32, None, return_empty=True)

    assert map_sample["map_num_nodes"] == 3
    assert map_sample["map_num_edges"] == 3
    assert map_sample["map_edge_indices"].shape == (3, 2)
    np.testing.assert_array_equal(
        map_sample["map_edge_indices"],
        np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
    )


def test_map_encoding_can_return_empty_payloads() -> None:
    """Map encoding should produce zero-sized arrays when no map is attached."""
    map_sample = encode_map_from_scene(_scene(with_map=False), np.float32, None, return_empty=True)

    assert map_sample["map_num_nodes"] == 0
    assert map_sample["map_num_edges"] == 0
    assert map_sample["map_node_positions"].shape == (0, 2)
    assert map_sample["map_edge_indices"].shape == (0, 2)
