"""PyTorch Geometric adapters built on top of generic Torch scene datasets."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic

from typing_extensions import override

from dronalize.core.optional import raise_missing_optional_dependency

try:
    from torch.utils.data import IterableDataset
    from torch_geometric.data import Batch, HeteroData
    from torch_geometric.data import Dataset as PyGDataset

    from dronalize.io.adapters.torch import ReaderT, TorchSceneDataset, TorchSceneRecord
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The PyG scene dataset adapter", extra="pyg")

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import torch
    from torch_geometric.data.dataset import BaseData

HeteroDataTransform = Callable[[HeteroData], HeteroData]


class HeteroSceneDataset(PyGDataset, Generic[ReaderT]):
    """PyG dataset view over Dronalize scene records.

    Each sample is a `HeteroData` object with `agent` and `map` node stores and
    a `("map", "connects", "map")` edge store. Agent trajectories are exposed
    as `agent.x` and `agent.y` with matching validity masks.
    """

    def __init__(
        self, reader: ReaderT, *, copy: bool = True, transform: HeteroDataTransform | None = None
    ) -> None:
        super().__init__(transform=transform)
        self.dataset: TorchSceneDataset[ReaderT] = TorchSceneDataset(reader, copy=copy)

    @override
    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over the wrapped dataset, yielding samples converted to `HeteroData`."""
        for record in self.dataset:
            yield _convert_to_hetero(record)

    @override
    def get(self, idx: int) -> HeteroData:
        """Return one sample converted to `HeteroData`."""
        return _convert_to_hetero(self.dataset[idx])

    @override
    def len(self) -> int:
        """Return the number of samples visible through the wrapped dataset."""
        return len(self.dataset)


class IterableHeteroSceneDataset(HeteroSceneDataset[ReaderT], IterableDataset[HeteroData]):
    """Iterable PyG dataset view over Dronalize scene records."""


def collate_hetero_with_time_padding(samples: Sequence[HeteroData]) -> Batch:
    """Batch hetero scenes by padding agent time axes within the current batch."""
    if not samples:
        msg = "`samples` must contain at least one HeteroData object."
        raise ValueError(msg)

    max_history_frames = max(int(sample["agent"].x.size(1)) for sample in samples)
    max_future_frames = max(int(sample["agent"].y.size(1)) for sample in samples)
    padded_samples: list[BaseData] = [
        _pad_hetero_time_axes(
            sample, history_frames=max_history_frames, future_frames=max_future_frames
        )
        for sample in samples
    ]
    return Batch.from_data_list(padded_samples)


def _convert_to_hetero(sample: TorchSceneRecord) -> HeteroData:
    data = HeteroData()

    data["agent"].x = sample.history_features
    data["agent"].x_mask = sample.history_mask
    data["agent"].y = sample.future_features
    data["agent"].y_mask = sample.future_mask
    data["agent"].agent_type = sample.agent_types
    data["agent"].passed_mask = sample.screened_agent_mask
    data["agent"].num_nodes = sample.history_features.size(0)

    data["map"].x = sample.map_node_positions
    data["map"].node_type = sample.map_node_types
    data["map"].num_nodes = sample.map_node_positions.size(0)

    data["map", "connects", "map"].edge_index = sample.map_edge_indices.long()
    data["map", "connects", "map"].edge_type = sample.map_edge_types

    data.scene_number = int(sample.scene_number)
    data.dataset = sample.dataset
    data.position_offset = sample.position_offset
    return data


def _pad_hetero_time_axes(
    sample: HeteroData, *, history_frames: int, future_frames: int
) -> HeteroData:
    padded = sample.clone()
    padded["agent"].x = _pad_along_dim(sample["agent"].x, target=history_frames, dim=1)
    padded["agent"].x_mask = _pad_along_dim(sample["agent"].x_mask, target=history_frames, dim=1)
    padded["agent"].y = _pad_along_dim(sample["agent"].y, target=future_frames, dim=1)
    padded["agent"].y_mask = _pad_along_dim(sample["agent"].y_mask, target=future_frames, dim=1)
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
