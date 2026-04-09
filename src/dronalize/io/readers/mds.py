"""Framework-neutral readers for Dronalize MDS exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, overload

import numpy as np
from typing_extensions import Unpack

from dronalize._internal.optional import raise_missing_optional_dependency
from dronalize.io.records import RawSceneRecord

try:
    from streaming import Stream, StreamingDataset
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The MDS storage reader", extra="storage-mds")

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    import numpy.typing as npt

__all__ = ["MDSReader", "MDSReaderInitArgs"]


class MDSReaderInitArgs(TypedDict, total=False):
    """Optional arguments forwarded to `streaming.StreamingDataset`."""

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


class MDSReader:
    """Read raw scene records from an MDS dataset split."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        split: str | None = None,
        streams: Sequence[Stream] | None = None,
        **reader_args: Unpack[MDSReaderInitArgs],
    ) -> None:
        if path is None and streams is None:
            msg = "Either `path` or `streams` must be provided."
            raise ValueError(msg)

        if path is not None:
            self._backend: StreamingDataset = StreamingDataset(
                local=path.as_posix(), split=split, **reader_args
            )
        else:
            self._backend = StreamingDataset(streams=streams, **reader_args)

    def __len__(self) -> int:
        """Return the number of scene records visible through the backend."""
        return len(self._backend)

    def __iter__(self) -> Iterator[RawSceneRecord]:
        """Iterate over decoded scene records."""
        yield from (self._convert_record(record) for record in self._backend)

    @overload
    def __getitem__(self, at: int) -> RawSceneRecord: ...

    @overload
    def __getitem__(
        self, at: list[int] | npt.NDArray[np.int64] | slice
    ) -> list[RawSceneRecord]: ...

    def __getitem__(
        self, at: int | slice | list[int] | npt.NDArray[np.int64]
    ) -> RawSceneRecord | list[RawSceneRecord]:
        """Return one or more decoded scene records."""
        out: dict[str, Any] | list[dict[str, Any]] = self._backend[at]
        if isinstance(out, list):
            return [self._convert_record(record) for record in out]
        return self._convert_record(out)

    @property
    def backend(self) -> StreamingDataset:
        """Return the underlying Mosaic Streaming dataset."""
        return self._backend

    @staticmethod
    def _convert_record(record: dict[str, Any]) -> RawSceneRecord:
        input_len = int(record["input_len"])
        features = np.asarray(record["features"])
        mask = np.asarray(record["mask"], dtype=bool)

        map_node_positions = np.asarray(record["map_node_positions"])
        map_edge_indices = np.asarray(record["map_edge_indices"])
        map_node_types = np.asarray(record["map_node_types"])
        map_edge_types = np.asarray(record["map_edge_types"])

        (map_node_positions, map_edge_indices, map_node_types, map_edge_types) = (
            MDSReader._normalize_map_arrays(
                map_node_positions, map_edge_indices, map_node_types, map_edge_types
            )
        )

        return RawSceneRecord(
            scene_number=int(record["scene_number"]),
            position_offset=np.asarray(record["position_offset"], dtype=np.float64),
            agent_types=np.asarray(record["agent_types"], dtype=np.int32),
            input_features=features[:, :input_len],
            input_mask=mask[:, :input_len],
            output_features=features[:, input_len:],
            output_mask=mask[:, input_len:],
            map_node_positions=map_node_positions,
            map_edge_indices=map_edge_indices,
            map_node_types=map_node_types,
            map_edge_types=map_edge_types,
        )

    @staticmethod
    def _normalize_map_arrays(
        map_node_positions: npt.NDArray[Any],
        map_edge_indices: npt.NDArray[Any],
        map_node_types: npt.NDArray[Any],
        map_edge_types: npt.NDArray[Any],
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        is_empty_map = (
            map_node_positions.shape == (1, 2)
            and np.isnan(map_node_positions).all()
            and map_edge_indices.shape == (2, 1)
            and (map_edge_indices == -1).all()
            and map_node_types.shape == (1,)
            and (map_node_types == -1).all()
            and map_edge_types.shape == (1,)
            and (map_edge_types == -1).all()
        )

        if is_empty_map:
            return (
                np.empty((0, 2), dtype=map_node_positions.dtype),
                np.empty((2, 0), dtype=map_edge_indices.dtype),
                np.empty((0,), dtype=map_node_types.dtype),
                np.empty((0,), dtype=map_edge_types.dtype),
            )

        return map_node_positions, map_edge_indices, map_node_types, map_edge_types
