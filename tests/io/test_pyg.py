# ruff: noqa: E402
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import HeteroData

from dronalize.io.adapters import collate_hetero_with_time_padding


def _sample(
    *,
    input_len: int,
    output_len: int,
    agent_count: int,
    map_node_count: int,
    scene_number: int,
) -> HeteroData:
    data = HeteroData()

    data["agent"].x = torch.arange(agent_count * input_len, dtype=torch.float32).reshape(
        agent_count, input_len, 1
    )
    data["agent"].x_mask = torch.ones((agent_count, input_len), dtype=torch.bool)
    data["agent"].y = torch.arange(agent_count * output_len, dtype=torch.float32).reshape(
        agent_count, output_len, 1
    )
    data["agent"].y_mask = torch.ones((agent_count, output_len), dtype=torch.bool)
    data["agent"].agent_type = torch.arange(agent_count, dtype=torch.int64)
    data["agent"].num_nodes = agent_count

    data["map"].x = torch.arange(map_node_count * 2, dtype=torch.float32).reshape(map_node_count, 2)
    data["map"].node_type = torch.arange(map_node_count, dtype=torch.int64)
    data["map"].num_nodes = map_node_count

    if map_node_count > 1:
        data["map", "connects", "map"].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        data["map", "connects", "map"].edge_type = torch.tensor([1], dtype=torch.int64)
    else:
        data["map", "connects", "map"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["map", "connects", "map"].edge_type = torch.empty((0,), dtype=torch.int64)

    data.scene_number = scene_number
    data.position_offset = torch.tensor([float(scene_number), 0.0], dtype=torch.float32)
    return data


def test_collate_hetero_with_time_padding() -> None:
    """Validate collate function."""
    first = _sample(input_len=2, output_len=1, agent_count=2, map_node_count=1, scene_number=7)
    second = _sample(input_len=3, output_len=2, agent_count=1, map_node_count=2, scene_number=8)

    batch = collate_hetero_with_time_padding([first, second])

    assert batch["agent"].x.shape == (3, 3, 1)
    assert batch["agent"].x_mask.shape == (3, 3)
    assert batch["agent"].y.shape == (3, 2, 1)
    assert batch["agent"].y_mask.shape == (3, 2)

    assert torch.equal(batch["agent"].x[:2, 2, 0], torch.tensor([0.0, 0.0]))
    assert torch.equal(batch["agent"].x_mask[:2, 2], torch.tensor([False, False]))
    assert torch.equal(batch["agent"].y[:2, 1, 0], torch.tensor([0.0, 0.0]))
    assert torch.equal(batch["agent"].y_mask[:2, 1], torch.tensor([False, False]))

    assert batch["agent"].batch.tolist() == [0, 0, 1]
    assert batch["map"].batch.tolist() == [0, 1, 1]
    assert batch["map"].x.shape == (3, 2)
    assert batch["map", "connects", "map"].edge_index.shape == (2, 1)

    assert first["agent"].x.shape == (2, 2, 1)
    assert first["agent"].y.shape == (2, 1, 1)
