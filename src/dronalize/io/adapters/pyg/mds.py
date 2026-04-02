"""PyTorch Geometric adapter for Dronalize MDS exports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize._internal.optional import raise_missing_optional_dependency

try:
    from torch_geometric.data import Dataset, HeteroData

    from dronalize.io.adapters.mds import MDSDataset, StreamingDatasetInitArgs
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The MDS PyG dataset adapter", extra="pyg")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from streaming import Stream

    from dronalize.io.adapters.sample import RawSceneSample as _RawSceneSample


class MDSHeteroDataset(Dataset):
    """PyG compatible dataset for trajectory (scenes) data."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        split: str | None = None,
        streams: Sequence[Stream] | None = None,
        transform: Callable[[HeteroData], HeteroData] | None = None,
        backend_args: StreamingDatasetInitArgs | None = None,
    ) -> None:
        super().__init__(transform=transform)  # pyright: ignore[reportUnknownMemberType]
        self.backend: MDSDataset = MDSDataset(
            path=path, split=split, streams=streams, **(backend_args or {})
        )

    @override
    def __iter__(self) -> Iterator[HeteroData]:
        """Iterate over samples converted to ``torch_geometric`` hetero graphs."""
        for raw_sample in self.backend:
            yield _convert_to_hetero(raw_sample)

    @override
    def len(self) -> int:
        """Return the number of samples visible through the backend dataset."""
        return len(self.backend)

    @override
    def get(self, idx: int) -> HeteroData:
        """Return one sample converted to ``HeteroData``."""
        return _convert_to_hetero(self.backend[idx])


def _convert_to_hetero(sample: _RawSceneSample) -> HeteroData:
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
    data.global_origin = sample.global_origin
    return data
