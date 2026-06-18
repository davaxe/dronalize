"""PyTorch Geometric adapters built on top of generic Torch scene datasets."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic

from typing_extensions import override

from dronalize.core.optional import raise_missing_optional_dependency
from dronalize.io.adapters.torch import IterableTorchSceneDataset

try:
    from torch.utils.data import Dataset, IterableDataset
    from torch_geometric.data import Batch, HeteroData
    from torch_geometric.data import Dataset as PyGDataset

    from dronalize.io.adapters.torch import (
        IterableReaderT,
        ReaderT,
        TorchSceneDataset,
        TorchSceneRecord,
    )
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The PyG scene dataset adapter", extra="pyg")

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import torch
    from torch_geometric.data.dataset import BaseData

HeteroDataTransform = Callable[[HeteroData], HeteroData]


class HeteroSceneDataset(PyGDataset, Dataset[HeteroData], Generic[ReaderT]):
    """PyG dataset view over full-horizon Dronalize scene records.

    Each record is a `HeteroData` object with `agent` and `map` node stores and
    a `("map", "connects", "map")` edge store. Agent trajectories are exposed
    as `agent.features` with a matching `agent.agent_time_mask`.
    """

    def __init__(
        self, reader: ReaderT, *, copy: bool = True, transform: HeteroDataTransform | None = None
    ) -> None:
        super().__init__(transform=transform)
        self._transform: HeteroDataTransform | None = transform
        self.dataset: TorchSceneDataset[ReaderT] = TorchSceneDataset(reader, copy=copy)

    @override
    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over the wrapped dataset, yielding records converted to `HeteroData`."""
        for record in self.dataset:
            hetero = _convert_full_to_hetero(record)
            if self._transform is not None:
                hetero = self._transform(hetero)
            yield hetero

    @override
    def get(self, idx: int) -> HeteroData:
        """Return one record converted to `HeteroData`."""
        return _convert_full_to_hetero(self.dataset[idx])

    @override
    def len(self) -> int:
        """Return the number of records visible through the wrapped dataset."""
        return len(self.dataset)


class IterableHeteroSceneDataset(IterableDataset[HeteroData], Generic[IterableReaderT]):
    """Iterable PyG dataset view over full-horizon Dronalize scene records."""

    @override
    def __init__(
        self,
        reader: IterableReaderT,
        *,
        copy: bool = True,
        transform: HeteroDataTransform | None = None,
    ) -> None:
        super().__init__()
        self._transform: HeteroDataTransform | None = transform
        self.dataset: IterableTorchSceneDataset[IterableReaderT] = IterableTorchSceneDataset(
            reader, copy=copy
        )

    @override
    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over the wrapped dataset, yielding records converted to `HeteroData`."""
        for record in self.dataset:
            hetero = _convert_full_to_hetero(record)
            if self._transform is not None:
                hetero = self._transform(hetero)
            yield hetero

    def __len__(self) -> int:
        """Return the number of records visible through the wrapped dataset."""
        return len(self.dataset)


def collate_hetero_with_time_padding(records: Sequence[HeteroData]) -> Batch:
    """Batch hetero scenes by padding agent time axes within the current batch."""
    if not records:
        msg = "`records` must contain at least one HeteroData object."
        raise ValueError(msg)

    max_horizon_frames = max(int(record["agent"].features.size(1)) for record in records)
    padded_records: list[BaseData] = [
        _pad_full_hetero_time_axes(record, horizon_frames=max_horizon_frames) for record in records
    ]
    return Batch.from_data_list(padded_records)


def _convert_full_to_hetero(record: TorchSceneRecord) -> HeteroData:
    data = HeteroData()

    data["agent"].features = record.features
    data["agent"].agent_time_mask = record.agent_time_mask
    data["agent"].agent_id = record.agent_ids
    data["agent"].agent_type = record.agent_types
    data["agent"].screened_agent_mask = record.screened_agent_mask
    data["agent"].num_nodes = record.features.size(0)

    _attach_map_store(
        data,
        record.map_node_positions,
        record.map_node_types,
        record.map_edge_indices,
        record.map_edge_types,
    )
    _attach_common_metadata(
        data,
        record.scene_number,
        record.dataset_id,
        record.position_offset,
        default_observation_length=record.default_observation_length,
        ego_agent_id=record.ego_agent_id,
    )
    return data


def _attach_map_store(
    data: HeteroData,
    node_positions: torch.Tensor,
    node_types: torch.Tensor,
    edge_index: torch.Tensor,
    edge_types: torch.Tensor,
) -> None:
    data["map"].x = node_positions
    data["map"].node_type = node_types
    data["map"].num_nodes = node_positions.size(0)
    data["map", "connects", "map"].edge_index = edge_index.long()
    data["map", "connects", "map"].edge_type = edge_types


def _attach_common_metadata(
    data: HeteroData,
    scene_number: int,
    dataset_id: int | None,
    position_offset: torch.Tensor,
    *,
    default_observation_length: int | None,
    ego_agent_id: int | None,
) -> None:
    data.scene_number = int(scene_number)
    data.dataset_id = -1 if dataset_id is None else int(dataset_id)
    data.position_offset = position_offset
    data.default_observation_length = default_observation_length
    data.ego_agent_id = ego_agent_id


def _pad_full_hetero_time_axes(record: HeteroData, *, horizon_frames: int) -> HeteroData:
    padded = record.clone()
    padded["agent"].features = _pad_along_dim(
        record["agent"].features, target=horizon_frames, dim=1
    )
    padded["agent"].agent_time_mask = _pad_along_dim(
        record["agent"].agent_time_mask, target=horizon_frames, dim=1
    )
    return padded


def _pad_along_dim(tensor: torch.Tensor, *, target: int, dim: int) -> torch.Tensor:
    current = int(tensor.size(dim))
    if current >= target:
        return tensor

    shape = list(tensor.shape)
    shape[dim] = target
    padded = tensor.new_zeros(shape)

    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(0, current)
    padded[tuple(slices)] = tensor
    return padded
