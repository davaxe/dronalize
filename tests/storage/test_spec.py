import numpy as np
import polars as pl
from polars.testing import assert_series_equal

from dronalize.core.maps import MapGraph
from dronalize.core.scene import CANONICAL_V1, POSITIONS_ONLY_V1, Scene
from dronalize.io import WriterConfig
from dronalize.io.encoding import (
    encode_map_from_scene,
    scene_sample_to_parts,
    scene_to_numpy_dict,
)
from dronalize.io.manifest import FORMAT_VERSION, StorageManifest
from dronalize.processing.ingest import LoaderConfig


def _scene(*, with_map: bool = True) -> Scene:
    graph = MapGraph(
        node_positions=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        edge_indices=np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int32),
        node_types=np.array([1, 2, 3], dtype=np.int32),
        edge_types=np.array([4, 5, 6], dtype=np.int32),
    )
    return Scene(
        frame=pl.DataFrame(
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
        scene_number=7,
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
        source_scene_schema=CANONICAL_V1,
        writer_config=writer_config,
        has_map=True,
    )

    assert manifest.format_version == FORMAT_VERSION
    assert manifest.source_scene_schema == "canonical"
    assert manifest.scene_schema == "canonical"
    assert manifest.derived_features == ()
    assert manifest.feature_columns == ("x", "y", "vx", "vy", "ax", "ay", "yaw")
    assert manifest.input_len == 2
    assert manifest.output_len == 1
    assert manifest.precision == "float64"
    assert manifest.offset_positions is False
    assert manifest.has_map is True


def test_storage_manifest_lists_exported_derived_fields() -> None:
    """Manifest should record which exported schema fields are derived."""
    loader_config = LoaderConfig(input_len=2, output_len=1, sample_time=0.5)
    writer_config = WriterConfig.create("canonical", precision="float64", offset_positions=False)

    manifest = StorageManifest.from_configs(
        loader_config=loader_config,
        source_scene_schema=POSITIONS_ONLY_V1,
        writer_config=writer_config,
        has_map=False,
    )

    assert manifest.source_scene_schema == "positions_only"
    assert manifest.scene_schema == "canonical"
    assert manifest.derived_features == ("vx", "vy", "ax", "ay", "yaw")


def test_storage_manifest_reads_older_payloads_without_derivation_metadata() -> None:
    """Older manifests should default missing provenance fields sensibly."""
    manifest = StorageManifest.from_json_dict({
        "format_version": 1,
        "scene_schema": "canonical",
        "feature_columns": ["x", "y", "vx", "vy", "ax", "ay", "yaw"],
        "input_len": 2,
        "output_len": 1,
        "precision": "float32",
        "offset_positions": True,
        "has_map": False,
        "sample_time": 1.0,
        "original_sample_time": 1.0,
    })

    assert manifest.source_scene_schema == "canonical"
    assert manifest.derived_features == ()


def test_scene_encoding_uses_new_logical_field_names() -> None:
    """Encoded scenes should use the new persisted field names."""
    sample = scene_to_numpy_dict(_scene(), dtype=np.float32)

    assert sample["scene_number"] == 7
    assert sample["agent_types"].tolist() == [1, 2]
    assert sample["features"].shape == (2, 3, 7)
    assert sample["mask"].shape == (2, 3)


def test_map_encoding_can_return_empty_payloads() -> None:
    """Map encoding should produce zero-sized arrays when no map is attached."""
    map_sample = encode_map_from_scene(_scene(with_map=False), np.float32, None, return_empty=True)
    map_node_positions = map_sample["map_node_positions"]
    map_edge_indices = map_sample["map_edge_indices"]
    map_node_types = map_sample["map_node_types"]
    map_edge_types = map_sample["map_edge_types"]
    is_empty_map = (
        map_node_positions.shape == (1, 2)
        and np.isnan(map_node_positions).all()
        and map_edge_indices.shape == (2, 1)
        and (map_edge_indices == -1).all()
        and map_node_types.shape == (1,)
        and (map_node_types == -1).all()
        and map_edge_types.shape == (1,)
        and (map_edge_types == -1).all()
    )
    assert is_empty_map, "Expected empty map encoding, but got non-empty arrays"


def test_inverse() -> None:
    """Test that the inverse of encoding a scene sample is the original scene data."""
    scene = _scene()
    sample = scene_to_numpy_dict(scene, dtype=np.float32)
    reconstructed_scene = scene_sample_to_parts(
        sample,
        feature_columns=scene.schema.feature_columns(),
    )
    frame = scene.frame.sort(by=["frame", "id"])
    for feature in scene.schema.feature_columns():
        assert feature in reconstructed_scene.columns
        assert_series_equal(frame[feature], reconstructed_scene[feature])

    assert_series_equal(frame["agent_category"], reconstructed_scene["agent_category"])
