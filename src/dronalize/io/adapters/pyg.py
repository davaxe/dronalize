"""PyTorch Geometric adapters built on top of generic Torch scene datasets."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic

from typing_extensions import override

from dronalize.core.optional import raise_missing_optional_dependency
from dronalize.io.adapters.torch import IterableTorchSceneDataset, IterableTorchSplitSceneDataset

try:
    from torch.utils.data import Dataset, IterableDataset
    from torch_geometric.data import Batch, HeteroData
    from torch_geometric.data import Dataset as PyGDataset

    from dronalize.io.adapters.torch import (
        IterableReaderT,
        ObservationLength,
        ReaderT,
        TorchSceneDataset,
        TorchSceneRecord,
        TorchSplitSceneDataset,
        TorchSplitSceneRecord,
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

    Each sample is a `HeteroData` object with `agent` and `map` node stores and
    a `("map", "connects", "map")` edge store. Agent trajectories are exposed
    as `agent.features` with a matching `agent.mask`.
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
            yield _convert_full_to_hetero(record)

    @override
    def get(self, idx: int) -> HeteroData:
        """Return one sample converted to `HeteroData`."""
        return _convert_full_to_hetero(self.dataset[idx])

    @override
    def len(self) -> int:
        """Return the number of samples visible through the wrapped dataset."""
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
        """Iterate over the wrapped dataset, yielding samples converted to `HeteroData`."""
        for record in self.dataset:
            hetero = _convert_full_to_hetero(record)
            if self._transform is not None:
                hetero = self._transform(hetero)
            yield hetero

    def __len__(self) -> int:
        """Return the number of samples visible through the wrapped dataset."""
        return len(self.dataset)


class SplitHeteroSceneDataset(PyGDataset, Dataset[HeteroData], Generic[ReaderT]):
    """PyG dataset view over Dronalize scene records split on read.

    Each sample is a `HeteroData` object with `agent.x` / `agent.x_mask` for
    the observation prefix and `agent.y` / `agent.y_mask` for the remaining
    target horizon.
    """

    def __init__(
        self,
        reader: ReaderT,
        *,
        observation_length: ObservationLength | None = None,
        copy: bool = True,
        transform: HeteroDataTransform | None = None,
    ) -> None:
        super().__init__(transform=transform)
        self.dataset: TorchSplitSceneDataset[ReaderT] = TorchSplitSceneDataset(
            reader, observation_length=observation_length, copy=copy
        )

    @override
    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over the wrapped dataset, yielding samples converted to `HeteroData`."""
        for record in self.dataset:
            yield _convert_split_to_hetero(record)

    @override
    def get(self, idx: int) -> HeteroData:
        """Return one split sample converted to `HeteroData`."""
        return _convert_split_to_hetero(self.dataset[idx])

    @override
    def len(self) -> int:
        """Return the number of samples visible through the wrapped dataset."""
        return len(self.dataset)


class IterableSplitHeteroSceneDataset(IterableDataset[HeteroData], Generic[IterableReaderT]):
    """Iterable PyG dataset view over split Dronalize scene records."""

    def __init__(
        self,
        reader: IterableReaderT,
        *,
        observation_length: ObservationLength | None = None,
        copy: bool = True,
        transform: HeteroDataTransform | None = None,
    ) -> None:
        self._transform: HeteroDataTransform | None = transform
        self.dataset: IterableTorchSplitSceneDataset[IterableReaderT] = (
            IterableTorchSplitSceneDataset(reader, observation_length=observation_length, copy=copy)
        )

    @override
    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over the wrapped dataset, yielding samples converted to `HeteroData`."""
        for record in self.dataset:
            hetero = _convert_split_to_hetero(record)
            if self._transform is not None:
                hetero = self._transform(hetero)
            yield hetero

    def __len__(self) -> int:
        """Return the number of samples visible through the wrapped dataset."""
        return len(self.dataset)


def collate_hetero_with_time_padding(samples: Sequence[HeteroData]) -> Batch:
    """Batch hetero scenes by padding agent time axes within the current batch."""
    if not samples:
        msg = "`samples` must contain at least one HeteroData object."
        raise ValueError(msg)

    if _is_split_sample(samples[0]):
        max_observation_frames = max(int(sample["agent"].x.size(1)) for sample in samples)
        max_target_frames = max(int(sample["agent"].y.size(1)) for sample in samples)
        padded_samples: list[BaseData] = [
            _pad_split_hetero_time_axes(
                sample, observation_frames=max_observation_frames, target_frames=max_target_frames
            )
            for sample in samples
        ]
        return Batch.from_data_list(padded_samples)

    max_horizon_frames = max(int(sample["agent"].features.size(1)) for sample in samples)
    padded_samples = [
        _pad_full_hetero_time_axes(sample, horizon_frames=max_horizon_frames) for sample in samples
    ]
    return Batch.from_data_list(padded_samples)


def _convert_full_to_hetero(sample: TorchSceneRecord) -> HeteroData:
    data = HeteroData()

    data["agent"].features = sample.features
    data["agent"].mask = sample.mask
    data["agent"].agent_type = sample.agent_types
    data["agent"].passed_mask = sample.screened_agent_mask
    data["agent"].num_nodes = sample.features.size(0)

    _attach_map_store(
        data,
        sample.map_node_positions,
        sample.map_node_types,
        sample.map_edge_indices,
        sample.map_edge_types,
    )
    _attach_common_metadata(
        data,
        sample.scene_number,
        sample.dataset,
        sample.position_offset,
        default_observation_length=sample.default_observation_length,
    )
    return data


def _convert_split_to_hetero(sample: TorchSplitSceneRecord) -> HeteroData:
    data = HeteroData()
    data["agent"].x = sample.history_features
    data["agent"].x_mask = sample.history_mask
    data["agent"].y = sample.future_features
    data["agent"].y_mask = sample.future_mask
    data["agent"].agent_type = sample.agent_types
    data["agent"].passed_mask = sample.screened_agent_mask
    data["agent"].num_nodes = sample.history_features.size(0)

    _attach_map_store(
        data,
        sample.map_node_positions,
        sample.map_node_types,
        sample.map_edge_indices,
        sample.map_edge_types,
    )
    _attach_common_metadata(
        data,
        sample.scene_number,
        sample.dataset,
        sample.position_offset,
        default_observation_length=sample.default_observation_length,
    )
    data.observation_length = int(sample.history_features.size(1))
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
    dataset: str | None,
    position_offset: torch.Tensor,
    *,
    default_observation_length: int | None,
) -> None:
    data.scene_number = int(scene_number)
    data.dataset = dataset
    data.position_offset = position_offset
    data.default_observation_length = default_observation_length


def _is_split_sample(sample: HeteroData) -> bool:
    return hasattr(sample["agent"], "x") and hasattr(sample["agent"], "y")


def _pad_full_hetero_time_axes(sample: HeteroData, *, horizon_frames: int) -> HeteroData:
    padded = sample.clone()
    padded["agent"].features = _pad_along_dim(
        sample["agent"].features, target=horizon_frames, dim=1
    )
    padded["agent"].mask = _pad_along_dim(sample["agent"].mask, target=horizon_frames, dim=1)
    return padded


def _pad_split_hetero_time_axes(
    sample: HeteroData, *, observation_frames: int, target_frames: int
) -> HeteroData:
    padded = sample.clone()
    padded["agent"].x = _pad_along_dim(sample["agent"].x, target=observation_frames, dim=1)
    padded["agent"].x_mask = _pad_along_dim(
        sample["agent"].x_mask, target=observation_frames, dim=1
    )
    padded["agent"].y = _pad_along_dim(sample["agent"].y, target=target_frames, dim=1)
    padded["agent"].y_mask = _pad_along_dim(sample["agent"].y_mask, target=target_frames, dim=1)
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
