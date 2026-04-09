"""Torch dataset adapters built on top of framework-neutral storage readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset
from typing_extensions import Unpack, override

from dronalize.io.readers.mds import MDSReader, MDSReaderInitArgs, Stream

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from dronalize.io.records import RawSceneRecord


@dataclass(slots=True)
class TorchSceneRecord:
    """Torch-backed scene record returned by `MDSTorchDataset`."""

    scene_number: int
    """Unique scene number based on when the scene was processed."""
    position_offset: torch.Tensor
    """(2,1) dim per-scene position offset applied to all agent and map coordinates.

    Can be used to recover original coordinates by adding the offset back to the
    position features.
    """
    agent_types: torch.Tensor
    """Agent type IDs for each agent in the scene.

    Shaped `(num_agents,)` with `int32` dtype. Default the values correspod to
    `AgentCategory`.
    """
    input_features: torch.Tensor
    """Input features for each agent.

    Shaped `(num_agents, input_len, feature_dim)` with `float32` or `float64`
    dtype. The specific features and their order are determined by the dataset's
    trajectory schema.
    """
    input_mask: torch.Tensor
    """Boolean mask indicating valid input timesteps for each agent.

    Shaped `(num_agents, input_len)` with `bool` dtype. `True` values indicate
    valid timesteps.
    """
    output_features: torch.Tensor
    """Output features for each agent.

    Shaped `(num_agents, output_len, feature_dim)` with `float32` or `float64`
    dtype. The specific features and their order are determined by the dataset's
    trajectory schema.
    """
    output_mask: torch.Tensor
    """Boolean mask indicating valid output timesteps for each agent.

    Shaped `(num_agents, output_len)` with `bool` dtype. `True` values indicate
    valid timesteps.
    """
    map_node_positions: torch.Tensor
    """Positions of map nodes in the scene.

    Shaped `(num_map_nodes, 2)` with `float32` or `float64` dtype. The positions
    are relative to the scene's `position_offset`.
    """
    map_edge_indices: torch.Tensor
    """Edge indices for the map graph.

    Shaped `(2, num_map_edges)` with `int32` dtype. Each column represents a
    directed edge from `map_node_positions[map_edge_indices[0, i]]` to
    `map_node_positions[map_edge_indices[1, i]]`.
    """
    map_node_types: torch.Tensor
    """Type IDs for each map node in the scene.

    Shaped `(num_map_nodes,)` with `int32` dtype.
    """
    map_edge_types: torch.Tensor
    """Type IDs for each map edge in the scene.

    Shaped `(num_map_edges,)` with `int32` dtype. Values correspond to `EdgeType`.
    """


class MDSTorchDataset(IterableDataset[TorchSceneRecord]):
    """Iterable Torch dataset wrapper for raw MDS scene records.

    The dataset yields `TorchSceneRecord` instances, which contain the same data
    as the raw records returned by `MDSReader`, but with all arrays converted to
    Torch tensors.

    !!! note "Batching"
        The dataset format is not natively compatible with
        `torch.utils.data.DataLoader` batching, since the records have variable
        numbers of agents and map nodes.

        If PyG-based batching is desired, consider using `MDSHeteroDataset`
        instead, which converts the raw records into PyG `HeteroData` objects
        with appropriate batching support.

    """

    def __init__(
        self,
        *,
        path: Path | None = None,
        split: str | None = None,
        streams: Sequence[Stream] | None = None,
        **reader_args: Unpack[MDSReaderInitArgs],
    ) -> None:
        self.reader: MDSReader = MDSReader(path=path, split=split, streams=streams, **reader_args)

    def __len__(self) -> int:
        """Return the number of scene records visible through the wrapped reader."""
        return len(self.reader)

    @override
    def __iter__(self) -> Iterator[TorchSceneRecord]:
        """Iterate over Torch-backed scene records."""
        for record in self.reader:
            yield to_torch_scene_record(record)

    @override
    def __getitem__(self, idx: int) -> TorchSceneRecord:
        """Return one scene record converted to Torch tensors."""
        return to_torch_scene_record(self.reader[idx])


def to_torch_scene_record(record: RawSceneRecord, *, copy: bool = True) -> TorchSceneRecord:
    """Convert a framework-neutral scene record into Torch tensors."""
    # Streaming can expose read-only NumPy views; copy once so Torch receives
    # writable, stable buffers without emitting warnings.
    return TorchSceneRecord(
        scene_number=record.scene_number,
        position_offset=torch.asarray(record.position_offset, copy=copy),
        agent_types=torch.asarray(record.agent_types, copy=copy),
        input_features=torch.asarray(record.input_features, copy=copy),
        input_mask=torch.asarray(record.input_mask, copy=copy),
        output_features=torch.asarray(record.output_features, copy=copy),
        output_mask=torch.asarray(record.output_mask, copy=copy),
        map_node_positions=torch.asarray(record.map_node_positions, copy=copy),
        map_edge_indices=torch.asarray(record.map_edge_indices, copy=copy),
        map_node_types=torch.asarray(record.map_node_types, copy=copy),
        map_edge_types=torch.asarray(record.map_edge_types, copy=copy),
    )
