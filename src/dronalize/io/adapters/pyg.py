"""PyTorch Geometric adapters and batching helpers for Dronalize MDS exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import IterableDataset
from typing_extensions import Unpack, override

from dronalize.core.optional import raise_missing_optional_dependency

try:
    from torch_geometric.data import Batch, Dataset, HeteroData

    from dronalize.io.adapters.torch import MDSTorchDataset, TorchSceneRecord
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The MDS PyG dataset adapter", extra="pyg")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    import torch
    from torch_geometric.data.dataset import BaseData

    from dronalize.io.readers.mds import MDSReaderInitArgs, Stream


class MDSHeteroDataset(Dataset, IterableDataset[HeteroData]):
    """Iterable PyG dataset for trajectory (scene) data."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        split: str | None = None,
        streams: Sequence[Stream] | None = None,
        transform: Callable[[HeteroData], HeteroData] | None = None,
        **reader_args: Unpack[MDSReaderInitArgs],
    ) -> None:
        super().__init__(transform=transform)  # pyright: ignore[reportUnknownMemberType]
        self.dataset: MDSTorchDataset = MDSTorchDataset(
            path=path, split=split, streams=streams, **reader_args
        )

    @override
    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over samples converted to `torch_geometric` hetero graphs."""
        for sample in self.dataset:
            yield _convert_to_hetero(sample)

    @override
    def len(self) -> int:
        """Return the number of samples visible through the wrapped dataset."""
        return len(self.dataset)

    @override
    def get(self, idx: int) -> HeteroData:
        """Return one sample converted to `HeteroData`."""
        return _convert_to_hetero(self.dataset[idx])


def collate_hetero_with_time_padding(samples: Sequence[HeteroData]) -> Batch:
    """Batch hetero scenes by padding time axes within the current batch.

    Agent and map counts remain dynamic across scenes. Only the agent time axes are
    padded so that `Batch.from_data_list` can concatenate the per-agent tensors.

    Parameters
    ----------
    samples : Sequence[HeteroData]
        A sequence of hetero graph samples to batch together. Each sample is
        expected to be on the format yielded by `MDSHeteroDataset`.

    Returns
    -------
    Batch
        A PyG `Batch` object containing the collated samples, with time axes
        padded to the maximum lengths within the batch.

    Notes
    -----
    This or similar collate function is necessary when using `MDSHeteroDataset`
    with a PyTorch `DataLoader` and the time dimension is dynamic across samples.
    This is generally the case when multiple different datasets are combined,
    e.g., by using `streams` when constructing `MDSHeteroDataset`.

    If the time dimensions are already consistent across samples, the PyG
    default collate function can be used directly without padding. Using
    `torch_geometric.loader.DataLoader` is valid and recommended in that case.

    """
    if not samples:
        msg = "`samples` must contain at least one HeteroData object."
        raise ValueError(msg)

    max_history_frames = max(int(sample["agent"].x.size(1)) for sample in samples)
    max_future_frames = max(int(sample["agent"].y.size(1)) for sample in samples)
    padded_samples: list[BaseData] = [
        _pad_hetero_time_axes(
            sample,
            history_frames=max_history_frames,
            future_frames=max_future_frames,
        )
        for sample in samples
    ]
    return Batch.from_data_list(padded_samples)


def _convert_to_hetero(sample: TorchSceneRecord) -> HeteroData:
    data = HeteroData()

    # Agent node store
    data["agent"].x = sample.input_features
    data["agent"].x_mask = sample.input_mask
    data["agent"].y = sample.output_features
    data["agent"].y_mask = sample.output_mask
    data["agent"].agent_type = sample.agent_types
    data["agent"].num_nodes = sample.input_features.size(0)

    # Map node store
    data["map"].x = sample.map_node_positions
    data["map"].node_type = sample.map_node_types
    data["map"].num_nodes = sample.map_node_positions.size(0)

    # Map edges
    data["map", "connects", "map"].edge_index = sample.map_edge_indices.long()
    data["map", "connects", "map"].edge_type = sample.map_edge_types

    # Scene-level metadata
    data.scene_number = int(sample.scene_number)
    data.position_offset = sample.position_offset
    return data


def _pad_hetero_time_axes(
    sample: HeteroData, *, history_frames: int, future_frames: int
) -> HeteroData:
    padded = sample.clone()
    padded["agent"].x = _pad_along_dim(sample["agent"].x, target=history_frames, dim=1)
    padded["agent"].x_mask = _pad_along_dim(
        sample["agent"].x_mask, target=history_frames, dim=1
    )
    padded["agent"].y = _pad_along_dim(sample["agent"].y, target=future_frames, dim=1)
    padded["agent"].y_mask = _pad_along_dim(
        sample["agent"].y_mask, target=future_frames, dim=1
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
