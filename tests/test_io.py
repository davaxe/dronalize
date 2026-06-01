# ruff: noqa: PLC0415
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from dronalize.io import DatasetManifest, read_manifest
from dronalize.io.backends.pickle import PickleWriter
from dronalize.io.encoding import encode_scene_record, encode_split_scene_record
from dronalize.io.encoding.mds import decode_mds_sample, encode_mds_sample
from dronalize.io.manifest import write_manifest
from dronalize.io.readers import PickleReader
from dronalize.io.records import join_split_scene_record, split_scene_record
from tests.support import assert_scene_record_equal, output_plan

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from dronalize.core.scene import Scene
    from dronalize.io.records import SceneRecord

NDArrayAny = npt.NDArray[Any]


@dataclass(slots=True)
class CustomPickleSample:
    scene_number: int
    dataset: str | None
    values: npt.NDArray[Any]
    source: str


def test_split_helpers_roundtrip(scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    record = encode_scene_record(scene, dtype=np.float64)
    split = encode_split_scene_record(scene, dtype=np.float64, observation_length=2)

    rebuilt = split_scene_record(record, observation_length=2)
    rejoined = join_split_scene_record(split)

    np.testing.assert_allclose(record.features, rejoined.features)
    np.testing.assert_array_equal(record.mask, rejoined.mask)
    np.testing.assert_allclose(split.history_features, rebuilt.history_features)
    np.testing.assert_allclose(split.future_features, rebuilt.future_features)
    assert rebuilt.dataset == "demo"
    assert rejoined.dataset == "demo"
    assert record.dataset == "demo"


def test_split_scene_record_rejects_bad_length(scene: Scene) -> None:
    record = encode_scene_record(scene, dtype=np.float64)

    with pytest.raises(ValueError, match="observation_length"):
        _ = split_scene_record(record, observation_length=record.horizon_frames + 1)


def test_encode_scene_record_uses_passed_ids(scene: Scene) -> None:
    scene = replace(scene, passed_agent_ids=frozenset({10}))
    record = encode_scene_record(scene, dtype=np.float32)

    np.testing.assert_array_equal(record.screened_agent_mask, np.array([True, False]))


def test_pickle_writer_roundtrip(tmp_path: Path, scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"
    writer = PickleWriter(output_dir=output_dir, config=output_plan(), splits=None)

    expected = encode_scene_record(scene, dtype=np.float32)
    writer.write(scene)
    writer.finish_local()

    reader = PickleReader(output_dir)
    assert len(reader) == 1
    assert_scene_record_equal(reader[0], expected)


def test_pickle_writer_accepts_record_transform(tmp_path: Path, scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"

    def transform(record: SceneRecord) -> CustomPickleSample:
        return CustomPickleSample(
            scene_number=record.scene_number,
            dataset=record.dataset,
            values=record.features[:, :1, 0],
            source="record",
        )

    writer = PickleWriter(
        output_dir=output_dir, config=output_plan(), splits=None, record_transform=transform
    )
    writer.write(scene)
    writer.finish_local()

    reader = PickleReader(output_dir, sample_type=CustomPickleSample)
    sample = reader[0]

    assert sample.scene_number == scene.scene_number
    assert sample.dataset == "demo"
    assert sample.source == "record"
    assert sample.values.shape == (2, 1)


def test_pickle_writer_accepts_scene_transform(tmp_path: Path, scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"

    def transform(scene: Scene) -> CustomPickleSample:
        return CustomPickleSample(
            scene_number=scene.scene_number,
            dataset=scene.dataset,
            values=np.array([scene.horizon_frames], dtype=np.int32),
            source="scene",
        )

    writer = PickleWriter(
        output_dir=output_dir, config=output_plan(), splits=None, scene_transform=transform
    )
    writer.write(scene)
    writer.finish_local()

    sample = PickleReader(output_dir, sample_type=CustomPickleSample)[0]

    assert sample.scene_number == scene.scene_number
    assert sample.dataset == "demo"
    assert sample.source == "scene"
    np.testing.assert_array_equal(sample.values, np.array([scene.horizon_frames], dtype=np.int32))


def test_pickle_writer_rejects_multiple_transforms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="record_transform"):
        _ = PickleWriter(
            output_dir=tmp_path,
            config=output_plan(),
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
        output_dir=output_dir, config=output_plan(), splits=None, parallel=False
    )

    expected = encode_scene_record(scene, dtype=np.float32)
    writer.write(scene)
    writer.finish_local()
    writer.finish_final()

    reader = MDSReader(path=output_dir)
    assert len(reader) == 1
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
        config=output_plan(),
        splits=None,
        parallel=False,
        record_transform=transform,
        sample_columns=columns,
    )
    writer.write(scene)
    writer.finish_local()
    writer.finish_final()

    raw = MDSReader(path=output_dir, convert_raw=lambda sample: sample)[0]
    expected = encode_scene_record(scene, dtype=np.float32)

    assert int(raw["scene_number"]) == scene.scene_number
    np.testing.assert_allclose(raw["history"], expected.features[:, :1])
    np.testing.assert_allclose(raw["future"], expected.features[:, 1:])
    np.testing.assert_array_equal(raw["history_mask"], expected.mask[:, :1].astype(np.uint8))
    np.testing.assert_array_equal(raw["future_mask"], expected.mask[:, 1:].astype(np.uint8))


def test_mds_writer_requires_columns_for_custom_transform(tmp_path: Path) -> None:
    pytest.importorskip("streaming")

    from dronalize.io.backends.mds import MDSDatasetWriter

    with pytest.raises(ValueError, match="sample_columns"):
        _ = MDSDatasetWriter(
            output_dir=tmp_path,
            config=output_plan(),
            splits=None,
            parallel=False,
            record_transform=lambda record: {"scene_number": record.scene_number},
        )


def test_mds_encoder_decoder_roundtrip(scene: Scene) -> None:
    scene = replace(scene, dataset="demo")
    expected = encode_scene_record(scene, dtype=np.float32, default_observation_length=2)
    sample = encode_mds_sample(expected)
    decoded = decode_mds_sample(sample)

    assert_scene_record_equal(decoded, expected)


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
    writer = PickleWriter(output_dir=output_dir, config=output_plan(), splits=None)

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
    sample = TorchSceneDataset(reader)[0]

    assert sample.scene_number == expected.scene_number
    assert sample.dataset == expected.dataset
    _assert_tensor_allclose(sample.position_offset, expected.position_offset)
    _assert_tensor_array_equal(sample.agent_types, expected.agent_types)
    _assert_tensor_array_equal(sample.screened_agent_mask, expected.screened_agent_mask)
    _assert_tensor_allclose(sample.features, expected.features)
    _assert_tensor_array_equal(sample.mask, expected.mask)
    _assert_tensor_allclose(sample.map_node_positions, expected.map_node_positions)
    _assert_tensor_array_equal(sample.map_edge_indices, expected.map_edge_indices)
    _assert_tensor_array_equal(sample.map_node_types, expected.map_node_types)
    _assert_tensor_array_equal(sample.map_edge_types, expected.map_edge_types)


def test_pyg_dataset_roundtrip(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from dronalize.io.adapters.pyg import HeteroSceneDataset

    reader, expected = _build_pickle_reader(tmp_path, scene)
    sample = HeteroSceneDataset(reader).get(0)

    assert sample.scene_number == expected.scene_number
    assert sample.dataset == expected.dataset
    _assert_tensor_allclose(sample.position_offset, expected.position_offset)
    _assert_tensor_allclose(sample["agent"].features, expected.features)
    _assert_tensor_array_equal(sample["agent"].mask, expected.mask)
    _assert_tensor_array_equal(sample["agent"].agent_type, expected.agent_types)
    _assert_tensor_array_equal(sample["agent"].passed_mask, expected.screened_agent_mask)
    _assert_tensor_allclose(sample["map"].x, expected.map_node_positions)
    _assert_tensor_array_equal(sample["map"].node_type, expected.map_node_types)
    _assert_tensor_array_equal(
        sample["map", "connects", "map"].edge_index, expected.map_edge_indices
    )
    _assert_tensor_array_equal(sample["map", "connects", "map"].edge_type, expected.map_edge_types)


def test_split_pyg_dataset_roundtrip(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from dronalize.io.adapters.pyg import SplitHeteroSceneDataset

    reader, expected = _build_pickle_reader(tmp_path, scene)
    observation_length = 2
    sample = SplitHeteroSceneDataset(reader, observation_length=observation_length).get(0)
    split_expected = expected.split(observation_length)

    assert sample.scene_number == expected.scene_number
    assert sample.dataset == expected.dataset
    _assert_tensor_allclose(sample.position_offset, expected.position_offset)
    _assert_tensor_allclose(sample["agent"].x, split_expected.history_features)
    _assert_tensor_array_equal(sample["agent"].x_mask, split_expected.history_mask)
    _assert_tensor_allclose(sample["agent"].y, split_expected.future_features)
    _assert_tensor_array_equal(sample["agent"].y_mask, split_expected.future_mask)
    _assert_tensor_array_equal(sample["agent"].agent_type, expected.agent_types)
    _assert_tensor_array_equal(sample["agent"].passed_mask, expected.screened_agent_mask)
    _assert_tensor_allclose(sample["map"].x, expected.map_node_positions)
    _assert_tensor_array_equal(sample["map"].node_type, expected.map_node_types)
    _assert_tensor_array_equal(
        sample["map", "connects", "map"].edge_index, expected.map_edge_indices
    )
    _assert_tensor_array_equal(sample["map", "connects", "map"].edge_type, expected.map_edge_types)


def test_split_pyg_dataset_uses_record_default_split(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from dronalize.io.adapters.pyg import SplitHeteroSceneDataset

    scene = replace(scene, dataset="demo")
    output_dir = tmp_path / "pickle"
    writer = PickleWriter(
        output_dir=output_dir, config=output_plan(default_observation_length=1), splits=None
    )
    writer.write(scene)
    writer.finish_local()

    sample = SplitHeteroSceneDataset(PickleReader(output_dir)).get(0)

    assert sample.default_observation_length == 1
    assert sample.observation_length == 1
    assert int(sample["agent"].x.size(1)) == 1
    assert int(sample["agent"].y.size(1)) == 2


def test_split_pyg_dataset_accepts_callable_split(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from dronalize.io.adapters.pyg import SplitHeteroSceneDataset

    reader, _ = _build_pickle_reader(tmp_path, replace(scene, dataset="demo"))
    sample = SplitHeteroSceneDataset(
        reader, observation_length=lambda record: 1 if record.dataset == "demo" else 2
    ).get(0)

    assert sample.observation_length == 1
    assert int(sample["agent"].x.size(1)) == 1
    assert int(sample["agent"].y.size(1)) == 2


def test_pyg_collate_pads_full_horizon(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from dronalize.io.adapters.pyg import HeteroSceneDataset, collate_hetero_with_time_padding

    reader, _ = _build_pickle_reader(tmp_path, scene)
    sample = HeteroSceneDataset(reader).get(0)

    shorter = sample.clone()
    shorter["agent"].features = shorter["agent"].features[:, :1, :]
    shorter["agent"].mask = shorter["agent"].mask[:, :1]

    batch = collate_hetero_with_time_padding([shorter, sample])

    assert int(batch["agent"].features.size(1)) == int(sample["agent"].features.size(1))


def test_pyg_collate_pads_split(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from dronalize.io.adapters.pyg import SplitHeteroSceneDataset, collate_hetero_with_time_padding

    reader, _ = _build_pickle_reader(tmp_path, scene)
    sample = SplitHeteroSceneDataset(reader, observation_length=2).get(0)

    shorter = sample.clone()
    shorter["agent"].x = shorter["agent"].x[:, :1, :]
    shorter["agent"].x_mask = shorter["agent"].x_mask[:, :1]

    batch = collate_hetero_with_time_padding([shorter, sample])

    assert int(batch["agent"].x.size(1)) == int(sample["agent"].x.size(1))
    assert int(batch["agent"].y.size(1)) == int(sample["agent"].y.size(1))
