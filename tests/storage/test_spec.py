import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from dronalize.config import LoaderConfig, WriterConfig
from dronalize.maps.graph import MapGraph
from dronalize.scene import CANONICAL_V1, Scene
from dronalize.storage.encoding import (
    encode_map_from_scene,
    scene_sample_to_parts,
    scene_to_numpy_dict,
)
from dronalize.storage.spec import FORMAT_VERSION, StorageManifest


def _scene(*, with_map: bool = True) -> Scene:
    graph = MapGraph(
        node_positions=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        edge_indices=np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32),
        node_types=np.array([1, 2, 3], dtype=np.int32),
        edge_types=np.array([4, 5, 6], dtype=np.int32),
    )
    return Scene(
        inner=pl.DataFrame(
            {
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
            },
        ),
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


def test_map_encoding_can_return_empty_payloads() -> None:
    """Map encoding should produce zero-sized arrays when no map is attached."""
    map_sample = encode_map_from_scene(_scene(with_map=False), np.float32, None, return_empty=True)

    # MDS format cannot encode empty arrays, instead they are represented as
    # arrays with a single element.
    assert map_sample["map_num_nodes"] in {0, 1}
    assert map_sample["map_num_edges"] in {0, 1}
    assert map_sample["map_node_positions"].shape in {(0, 2), (1,)}
    assert map_sample["map_edge_indices"].shape in {(0, 2), (1,)}


def test_inverse() -> None:
    """Test that the inverse of encoding a scene sample is the original scene data."""
    scene = _scene()
    sample = scene_to_numpy_dict(scene, dtype=np.float32)
    reconstructed_scene, input_len, output_len = scene_sample_to_parts(
        sample,
        feature_columns=scene.schema.feature_columns(),
    )
    inner = scene.inner.sort(by=["frame", "id"])
    assert input_len == scene.input_len
    assert output_len == scene.output_len
    for feature in scene.schema.feature_columns():
        assert feature in reconstructed_scene.columns
        assert_series_equal(inner[feature], reconstructed_scene[feature])

    assert_series_equal(inner["agent_category"], reconstructed_scene["agent_category"])
