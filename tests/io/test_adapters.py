# ruff: noqa: PLC0415
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from dronalize.io.backends.pickle import PickleWriter
from dronalize.io.encoding import encode_scene_record
from dronalize.io.readers import PickleReader
from tests.support import output_plan

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from dronalize.core.scene import Scene
    from dronalize.io.records import SceneRecord


NDArrayAny = npt.NDArray[Any]


def _build_pickle_reader(tmp_path: Path, scene: Scene) -> tuple[PickleReader, SceneRecord]:
    output_dir = tmp_path / "pickle"
    writer = PickleWriter(output_dir=output_dir, config=output_plan(), splits=None)

    expected = encode_scene_record(scene, dtype=np.float32)
    writer.write(scene)
    writer.finish_local()
    writer.finish_final()

    return PickleReader(output_dir), expected


def _to_numpy(tensor: torch.Tensor) -> NDArrayAny:
    return cast("NDArrayAny", tensor.detach().cpu().numpy())  # pyright: ignore[reportUnknownMemberType]


def _assert_tensor_allclose(tensor: torch.Tensor, expected: NDArrayAny) -> None:
    np.testing.assert_allclose(_to_numpy(tensor), expected)


def _assert_tensor_array_equal(tensor: torch.Tensor, expected: NDArrayAny) -> None:
    np.testing.assert_array_equal(_to_numpy(tensor), expected)


def test_torch_scene_dataset_roundtrip(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch")
    from dronalize.io.adapters.torch import TorchSceneDataset

    reader, expected = _build_pickle_reader(tmp_path, scene)
    sample = TorchSceneDataset(reader)[0]

    assert sample.scene_number == expected.scene_number
    _assert_tensor_allclose(sample.position_offset, expected.position_offset)
    _assert_tensor_array_equal(sample.agent_types, expected.agent_types)
    _assert_tensor_array_equal(sample.screened_agent_mask, expected.screened_agent_mask)
    _assert_tensor_allclose(sample.history_features, expected.history_features)
    _assert_tensor_array_equal(sample.history_mask, expected.history_mask)
    _assert_tensor_allclose(sample.future_features, expected.future_features)
    _assert_tensor_array_equal(sample.future_mask, expected.future_mask)
    _assert_tensor_allclose(sample.map_node_positions, expected.map_node_positions)
    _assert_tensor_array_equal(sample.map_edge_indices, expected.map_edge_indices)
    _assert_tensor_array_equal(sample.map_node_types, expected.map_node_types)
    _assert_tensor_array_equal(sample.map_edge_types, expected.map_edge_types)


def test_pyg_scene_dataset_roundtrip(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from dronalize.io.adapters.pyg import HeteroSceneDataset

    reader, expected = _build_pickle_reader(tmp_path, scene)
    sample = HeteroSceneDataset(reader).get(0)

    assert sample.scene_number == expected.scene_number
    _assert_tensor_allclose(sample.position_offset, expected.position_offset)
    _assert_tensor_allclose(sample["agent"].x, expected.history_features)
    _assert_tensor_array_equal(sample["agent"].x_mask, expected.history_mask)
    _assert_tensor_allclose(sample["agent"].y, expected.future_features)
    _assert_tensor_array_equal(sample["agent"].y_mask, expected.future_mask)
    _assert_tensor_array_equal(sample["agent"].agent_type, expected.agent_types)
    _assert_tensor_array_equal(sample["agent"].passed_mask, expected.screened_agent_mask)
    _assert_tensor_allclose(sample["map"].x, expected.map_node_positions)
    _assert_tensor_array_equal(sample["map"].node_type, expected.map_node_types)
    _assert_tensor_array_equal(
        sample["map", "connects", "map"].edge_index, expected.map_edge_indices
    )
    _assert_tensor_array_equal(sample["map", "connects", "map"].edge_type, expected.map_edge_types)


def test_pyg_collate_hetero_with_time_padding(tmp_path: Path, scene: Scene) -> None:
    pytest.importorskip("torch_geometric")
    from dronalize.io.adapters.pyg import HeteroSceneDataset, collate_hetero_with_time_padding

    reader, _ = _build_pickle_reader(tmp_path, scene)
    sample = HeteroSceneDataset(reader).get(0)

    shorter = sample.clone()
    shorter["agent"].x = shorter["agent"].x[:, :1, :]
    shorter["agent"].x_mask = shorter["agent"].x_mask[:, :1]

    batch = collate_hetero_with_time_padding([shorter, sample])

    assert int(batch["agent"].x.size(1)) == int(sample["agent"].x.size(1))
    assert int(batch["agent"].y.size(1)) == int(sample["agent"].y.size(1))
