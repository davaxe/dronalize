from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast, overload

from typing_extensions import Unpack, override

from dronalize._internal._optional import raise_missing_optional_dependency
from dronalize.io.adapters.sample import RawSceneSample as _RawSceneSample

try:
    import torch
    from streaming import Stream, StreamingDataset
    from torch_geometric.data import Dataset, HeteroData
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(
        error,
        feature="The MDS PyG dataset adapter",
        extra="pyg",
    )

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

# Rexport streaming.Stream for users of this module, so they don't need to
# import streaming directly.
__all__ = ["Stream"]


class _StreamingDatasetInitArgs(TypedDict, total=False):
    """All optional arguments to `StreamingDataset` initializer."""

    download_retry: int
    download_timeout: float
    validate_hash: str
    keep_zip: bool
    epoch_size: int | str
    predownload: int
    cache_limit: int | str
    sampling_method: str
    sampling_granularity: int
    partition_algo: str
    num_canonical_nodes: int
    batch_size: int
    shuffle: bool
    shuffle_algo: str
    shuffle_seed: int
    shuffle_block_size: int
    batching_method: str
    allow_unsafe_types: bool
    replication: int
    stream_name: str
    stream_config: dict[str, Any]


class _MDSDatasetBackend(StreamingDataset):
    """Read raw scene samples from an MDS dataset split."""

    @override
    def __init__(
        self,
        *,
        path: Path | None = None,
        split: str | None = None,
        streams: Sequence[Stream] | None = None,
        **args: Unpack[_StreamingDatasetInitArgs],
    ) -> None:
        if path is None and streams is None:
            msg = "Either `path` or `streams` must be provided."
            raise ValueError(msg)

        if path is not None:
            super().__init__(local=path.as_posix(), split=split, **args)
        elif streams is not None:
            super().__init__(streams=streams, **args)

    @override
    def __iter__(self) -> Iterator[_RawSceneSample]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # This implcity calls __get_item__ that converts the raw dict samples
        # into MDSRawSceneSample instances.
        return cast("Iterator[_RawSceneSample]", super().__iter__())

    @overload
    def __getitem__(self, at: int) -> _RawSceneSample: ...

    @overload
    def __getitem__(
        self, at: list[int] | npt.NDArray[np.int64] | slice
    ) -> list[_RawSceneSample]: ...

    @override
    def __getitem__(
        self,
        at: int | slice | list[int] | npt.NDArray[np.int64],
    ) -> _RawSceneSample | list[_RawSceneSample]:
        out: dict[str, Any] | list[dict[str, Any]] = super().__getitem__(at)
        if isinstance(out, list):
            return [self._convert_sample(sample) for sample in out]
        return self._convert_sample(out)

    @staticmethod
    def _convert_sample(sample: dict[str, Any]) -> _RawSceneSample:
        input_len = int(sample["input_len"])

        map_node_positions = torch.tensor(sample["map_node_positions"])
        map_edge_indices = torch.tensor(sample["map_edge_indices"])
        map_node_types = torch.tensor(sample["map_node_types"])
        map_edge_types = torch.tensor(sample["map_edge_types"])

        (
            map_node_positions,
            map_edge_indices,
            map_node_types,
            map_edge_types,
        ) = _MDSDatasetBackend._normalize_map_tensors(
            map_node_positions,
            map_edge_indices,
            map_node_types,
            map_edge_types,
        )

        return _RawSceneSample(
            scene_number=int(sample["scene_number"]),
            global_origin=torch.tensor(sample["global_origin"]),
            agent_types=torch.tensor(sample["agent_types"]),
            input_features=torch.tensor(sample["features"][:, :input_len]),
            input_mask=torch.tensor(sample["mask"][:, :input_len]),
            output_features=torch.tensor(sample["features"][:, input_len:]),
            output_mask=torch.tensor(sample["mask"][:, input_len:]),
            map_node_positions=map_node_positions,
            map_edge_indices=map_edge_indices,
            map_node_types=map_node_types,
            map_edge_types=map_edge_types,
        )

    @staticmethod
    def _normalize_map_tensors(
        map_node_positions: torch.Tensor,
        map_edge_indices: torch.Tensor,
        map_node_types: torch.Tensor,
        map_edge_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        is_empty_map = (
            map_node_positions.shape == (1, 2)
            and torch.isnan(map_node_positions).all()
            and map_edge_indices.shape == (2, 1)
            and (map_edge_indices == -1).all()
            and map_node_types.shape == (1,)
            and (map_node_types == -1).all()
            and map_edge_types.shape == (1,)
            and (map_edge_types == -1).all()
        )

        if is_empty_map:
            return (
                torch.empty((0, 2), dtype=map_node_positions.dtype),
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0,), dtype=map_node_types.dtype),
                torch.empty((0,), dtype=map_edge_types.dtype),
            )

        return map_node_positions, map_edge_indices, map_node_types, map_edge_types


class MDSHeteroDataset(Dataset):
    """PyG compatible dataset for trajectory (scenes) data."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        split: str | None = None,
        streams: Sequence[Stream] | None = None,
        transform: Callable[[HeteroData], HeteroData] | None = None,
        backend_args: _StreamingDatasetInitArgs | None = None,
    ) -> None:
        super().__init__(transform=transform)
        self.backend: _MDSDatasetBackend = _MDSDatasetBackend(
            path=path, split=split, streams=streams, **(backend_args or {})
        )

    @override
    def __iter__(self) -> Iterator[HeteroData]:
        for raw_sample in self.backend:
            yield _convert_to_hetero(raw_sample)

    @override
    def len(self) -> int:
        return len(self.backend)

    @override
    def get(self, idx: int) -> HeteroData:
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
