# ruff: noqa: PLC0415
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from dronalize.io import DatasetManifest, read_manifest
from dronalize.io.backends.pickle import PickleWriter
from dronalize.io.encoding import encode_scene_record
from dronalize.io.encoding.mds import decode_mds_row, encode_mds_row
from dronalize.io.manifest import write_manifest
from dronalize.io.readers import PickleReader
from tests.support import assert_scene_record_equal, output_config

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from dronalize.core.scene import Scene
    from dronalize.io.records import SceneRecord

NDArrayAny = npt.NDArray[Any]


@dataclass(slots=True)
class CustomPickleRecord:
    scene_number: int
    dataset: object
    values: npt.NDArray[Any]
    source: str


def test_split_scene_record_rejects_bad_length(scene: Scene) -> None:
    record = encode_scene_record(scene, dtype=np.float64)

    with pytest.raises(ValueError, match="observation_length"):
        _ = record.split(observation_length=record.horizon_frames + 1)


def test_encode_scene_record_uses_passed_ids(scene: Scene) -> None:
    scene = replace(scene, passed_agent_ids=frozenset({10}))
    record = encode_scene_record(scene, dtype=np.float32)

    np.testing.assert_array_equal(record.screened_agent_mask, np.array([True, False]))
    np.testing.assert_array_equal(record.agent_ids, np.array([10, 20], dtype=np.int64))


def test_pickle_writer_roundtrip(tmp_path: Path, scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"
    writer = PickleWriter(
        output_dir=output_dir, config=output_config(), default_observation_length=2, splits=None
    )

    expected = encode_scene_record(scene, dtype=np.float32, default_observation_length=2)
    writer.write(scene)
    writer.finish_local()

    reader = PickleReader(output_dir)
    assert len(reader) == 1
    assert_scene_record_equal(reader[0], expected)


def test_pickle_writer_accepts_record_transform(tmp_path: Path, scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"

    def transform(record: SceneRecord) -> CustomPickleRecord:
        return CustomPickleRecord(
            scene_number=record.scene_number,
            dataset=record.dataset_id,
            values=record.features[:, :1, 0],
            source="record",
        )

    writer = PickleWriter(
        output_dir=output_dir, config=output_config(), splits=None, record_transform=transform
    )
    writer.write(scene)
    writer.finish_local()

    reader = PickleReader(output_dir, record_type=CustomPickleRecord)
    record = reader[0]

    assert record.scene_number == scene.scene_number
    assert record.dataset is None
    assert record.source == "record"
    assert record.values.shape == (2, 1)


def test_pickle_writer_accepts_scene_transform(tmp_path: Path, scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"

    def transform(scene: Scene) -> CustomPickleRecord:
        return CustomPickleRecord(
            scene_number=scene.scene_number,
            dataset=scene.dataset,
            values=np.array([scene.horizon_frames], dtype=np.int32),
            source="scene",
        )

    writer = PickleWriter(
        output_dir=output_dir, config=output_config(), splits=None, scene_transform=transform
    )
    writer.write(scene)
    writer.finish_local()

    record = PickleReader(output_dir, record_type=CustomPickleRecord)[0]

    assert record.scene_number == scene.scene_number
    assert record.dataset == "demo"
    assert record.source == "scene"
    np.testing.assert_array_equal(record.values, np.array([scene.horizon_frames], dtype=np.int32))


def test_pickle_writer_rejects_multiple_transforms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="record_transform"):
        _ = PickleWriter(
            output_dir=tmp_path,
            config=output_config(),
            splits=None,
            record_transform=lambda record: record,
            scene_transform=lambda scene: scene,
        )


def test_mds_writer_roundtrip(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("streaming")
    scene = replace(scene, dataset="demo")

    from dronalize.io.backends.mds import MDSDatasetWriter
    from dronalize.io.readers import MDSReader

    output_dir = tmp_path / "mds"
    writer = MDSDatasetWriter(
        output_dir=output_dir, config=output_config(), splits=None, parallel=False
    )

    expected = encode_scene_record(scene, dtype=np.float32)
    writer.write(scene)
    writer.finish_local()
    writer.finish_final()

    reader = MDSReader(path=output_dir)
    assert len(reader) == 1
    assert reader[0].ego_agent_id == 10
    assert_scene_record_equal(reader[0], expected)


def test_mds_writer_accepts_transform_with_columns(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("streaming")
    scene = replace(scene, dataset="demo")

    from dronalize.io.backends.mds import MDSDatasetWriter
    from dronalize.io.readers import MDSReader

    def transform(record: SceneRecord) -> dict[str, Any]:
        observation_length = 1
        return {
            "scene_number": int(record.scene_number),
            "history": record.features[:, :observation_length],
            "future": record.features[:, observation_length:],
            "history_mask": record.mask[:, :observation_length].astype(np.uint8, copy=False),
            "future_mask": record.mask[:, observation_length:].astype(np.uint8, copy=False),
        }

    columns = {
        "scene_number": "int",
        "history": "ndarray:float32",
        "future": "ndarray:float32",
        "history_mask": "ndarray:uint8",
        "future_mask": "ndarray:uint8",
    }
    output_dir = tmp_path / "mds"
    writer = MDSDatasetWriter(
        output_dir=output_dir,
        config=output_config(),
        splits=None,
        parallel=False,
        record_transform=transform,
        mds_columns=columns,
    )
    writer.write(scene)
    writer.finish_local()
    writer.finish_final()

    raw = MDSReader(path=output_dir, convert_raw=lambda record: record)[0]
    expected = encode_scene_record(scene, dtype=np.float32)

    assert int(raw["scene_number"]) == scene.scene_number
    np.testing.assert_allclose(raw["history"], expected.features[:, :1])
    np.testing.assert_allclose(raw["future"], expected.features[:, 1:])
    np.testing.assert_array_equal(raw["history_mask"], expected.mask[:, :1].astype(np.uint8))
    np.testing.assert_array_equal(raw["future_mask"], expected.mask[:, 1:].astype(np.uint8))


def test_mds_writer_requires_columns_for_custom_transform(tmp_path: Path) -> None:
    pytest.importorskip("streaming")

    from dronalize.io.backends.mds import MDSDatasetWriter

    with pytest.raises(ValueError, match="mds_columns"):
        _ = MDSDatasetWriter(
            output_dir=tmp_path,
            config=output_config(),
            splits=None,
            parallel=False,
            record_transform=lambda record: {"scene_number": record.scene_number},
        )


def test_mds_encoder_decoder_roundtrip(scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    expected = encode_scene_record(scene, dtype=np.float32, default_observation_length=2)
    record = encode_mds_row(expected)
    decoded = decode_mds_row(record)

    assert_scene_record_equal(decoded, expected)


def test_split_scene_record_preserves_identity(scene: Scene) -> None:
    record = encode_scene_record(scene, dtype=np.float32)

    split = record.split(2)

    np.testing.assert_array_equal(split.agent_ids, record.agent_ids)
    assert split.ego_agent_id == record.ego_agent_id


def test_manifest_write_and_read_roundtrip(tmp_path: Path) -> None:
    manifest = DatasetManifest(
        dataset="test_dataset",
        storage_backend="pickle",
        dronalize_version="2.0.0",
        source_trajectory_schema="positions_only",
        source_trajectory_schema_fields=("frame", "id", "x", "y", "agent_category"),
        trajectory_schema="canonical",
        trajectory_schema_fields=(
            "frame",
            "id",
            "x",
            "y",
            "vx",
            "vy",
            "ax",
            "ay",
            "yaw",
            "agent_category",
        ),
        derived_features=("vx", "vy", "yaw"),
        feature_columns=("x", "y", "vx", "vy", "ax", "ay", "yaw"),
        horizon_frames=10,
        default_observation_length=4,
        precision="float32",
        recenter_positions=True,
        has_map=True,
        sample_time=0.1,
        original_sample_time=0.1,
    )

    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)

    assert loaded == manifest
    assert loaded.dataset_names == ("test_dataset",)


def test_manifest_rejects_bad_default_obs_length() -> None:
    with pytest.raises(ValueError, match="default_observation_length"):
        _ = DatasetManifest(
            dataset="test_dataset",
            storage_backend="pickle",
            dronalize_version="2.0.0",
            source_trajectory_schema="positions_only",
            source_trajectory_schema_fields=("frame", "id", "x", "y", "agent_category"),
            trajectory_schema="canonical",
            trajectory_schema_fields=(
                "frame",
                "id",
                "x",
                "y",
                "vx",
                "vy",
                "ax",
                "ay",
                "yaw",
                "agent_category",
            ),
            derived_features=("vx", "vy", "yaw"),
            feature_columns=("x", "y", "vx", "vy", "ax", "ay", "yaw"),
            horizon_frames=10,
            default_observation_length=11,
            precision="float32",
            recenter_positions=True,
            has_map=True,
            sample_time=0.1,
            original_sample_time=0.1,
        )


def _build_pickle_reader(tmp_path: Path, scene: Scene) -> tuple[PickleReader, SceneRecord]:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"
    writer = PickleWriter(output_dir=output_dir, config=output_config(), splits=None)

    expected = encode_scene_record(scene, dtype=np.float32)
    writer.write(scene)
    writer.finish_local()

    return PickleReader(output_dir), expected


def _to_numpy(tensor: torch.Tensor) -> NDArrayAny:
    return cast("NDArrayAny", tensor.detach().cpu().numpy())


def _assert_tensor_allclose(tensor: torch.Tensor, expected: NDArrayAny) -> None:
    np.testing.assert_allclose(_to_numpy(tensor), expected)


def _assert_tensor_array_equal(tensor: torch.Tensor, expected: NDArrayAny) -> None:
    np.testing.assert_array_equal(_to_numpy(tensor), expected)


def test_torch_dataset_roundtrip(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch")
    from dronalize.io.adapters.torch import TorchSceneDataset

    reader, expected = _build_pickle_reader(tmp_path, scene)
    record = TorchSceneDataset(reader)[0]

    assert record.scene_number == expected.scene_number
    assert record.dataset_id == expected.dataset_id
    _assert_tensor_allclose(record.position_offset, expected.position_offset)
    _assert_tensor_array_equal(record.agent_ids, expected.agent_ids)
    _assert_tensor_array_equal(record.agent_types, expected.agent_types)
    assert record.ego_agent_id == expected.ego_agent_id
    _assert_tensor_array_equal(record.screened_agent_mask, expected.screened_agent_mask)
    _assert_tensor_allclose(record.features, expected.features)
    _assert_tensor_array_equal(record.agent_time_mask, expected.mask)
    _assert_tensor_allclose(record.map_node_positions, expected.map_node_positions)
    _assert_tensor_array_equal(record.map_edge_indices, expected.map_edge_indices)
    _assert_tensor_array_equal(record.map_node_types, expected.map_node_types)
    _assert_tensor_array_equal(record.map_edge_types, expected.map_edge_types)


def test_torch_scene_record_splits_features(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch")
    from dronalize.io.adapters.torch import TorchSceneDataset

    reader, expected = _build_pickle_reader(tmp_path, scene)
    record = TorchSceneDataset(reader)[0]
    split = record.split(2)

    assert split.scene_number == expected.scene_number
    assert split.dataset_id == expected.dataset_id
    _assert_tensor_allclose(split.position_offset, expected.position_offset)
    _assert_tensor_array_equal(split.agent_ids, expected.agent_ids)
    assert split.ego_agent_id == expected.ego_agent_id
    _assert_tensor_allclose(split.history_features, expected.features[:, :2])
    _assert_tensor_array_equal(split.history_mask, expected.mask[:, :2])
    _assert_tensor_allclose(split.future_features, expected.features[:, 2:])
    _assert_tensor_array_equal(split.future_mask, expected.mask[:, 2:])
    _assert_tensor_array_equal(split.map_edge_indices, expected.map_edge_indices)


def test_pyg_dataset_roundtrip(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from dronalize.io.adapters.pyg import HeteroSceneDataset

    reader, expected = _build_pickle_reader(tmp_path, scene)
    record = HeteroSceneDataset(reader).get(0)

    assert record.scene_number == expected.scene_number
    assert record.dataset_id == (-1 if expected.dataset_id is None else expected.dataset_id)
    _assert_tensor_allclose(record.position_offset, expected.position_offset)
    _assert_tensor_allclose(record["agent"].features, expected.features)
    _assert_tensor_array_equal(record["agent"].agent_id, expected.agent_ids)
    _assert_tensor_array_equal(record["agent"].agent_time_mask, expected.mask)
    _assert_tensor_array_equal(record["agent"].agent_type, expected.agent_types)
    _assert_tensor_array_equal(record["agent"].screened_agent_mask, expected.screened_agent_mask)
    _assert_tensor_allclose(record["map"].x, expected.map_node_positions)
    _assert_tensor_array_equal(record["map"].node_type, expected.map_node_types)
    _assert_tensor_array_equal(
        record["map", "connects", "map"].edge_index, expected.map_edge_indices
    )
    _assert_tensor_array_equal(record["map", "connects", "map"].edge_type, expected.map_edge_types)
    assert record.ego_agent_id == expected.ego_agent_id


def test_pyg_collate_pads_full_horizon(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from dronalize.io.adapters.pyg import HeteroSceneDataset, collate_hetero_with_time_padding

    reader, _ = _build_pickle_reader(tmp_path, scene)
    record = HeteroSceneDataset(reader).get(0)

    shorter = record.clone()
    shorter["agent"].features = shorter["agent"].features[:, :1, :]
    shorter["agent"].agent_time_mask = shorter["agent"].agent_time_mask[:, :1]

    batch = collate_hetero_with_time_padding([shorter, record])

    assert int(batch["agent"].features.size(1)) == int(record["agent"].features.size(1))
