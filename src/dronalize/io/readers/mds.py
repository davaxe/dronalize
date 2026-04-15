"""Framework-neutral readers for Dronalize MDS exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from typing_extensions import TypedDict, Unpack, override

from dronalize.core.optional import raise_missing_optional_dependency
from dronalize.io.base import DatasetReader, split_directory_name
from dronalize.io.encoding.mds import decode_mds_sample

try:
    from streaming import Stream, StreamingDataset
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The MDS storage reader", extra="mds")

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from dronalize.core.categories import DatasetSplit
    from dronalize.io.records import SceneRecord

__all__ = ["MDSReader", "MDSReaderInitArgs", "Stream"]


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


class MDSReader(DatasetReader):
    """Read raw scene records from an MDS dataset split."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        split: DatasetSplit | str | None = None,
        streams: Sequence[Stream] | None = None,
        **reader_args: Unpack[MDSReaderInitArgs],
    ) -> None:
        super().__init__()
        if path is None and streams is None:
            msg = "Either `path` or `streams` must be provided."
            raise ValueError(msg)

        if path is not None:
            self._backend: StreamingDataset = StreamingDataset(
                local=path.as_posix(), split=split_directory_name(split), **reader_args
            )
        else:
            self._backend = StreamingDataset(streams=streams, **reader_args)

    @override
    def __len__(self) -> int:
        return len(self._backend)

    @override
    def __iter__(self) -> Iterator[SceneRecord]:
        yield from (self._convert_record(record) for record in self._backend)

    @overload
    def __getitem__(self, at: int) -> SceneRecord: ...

    @overload
    def __getitem__(self, at: list[int] | npt.NDArray[np.int64] | slice) -> list[SceneRecord]: ...

    @override
    def __getitem__(
        self, at: int | slice | list[int] | npt.NDArray[np.int64]
    ) -> SceneRecord | list[SceneRecord]:
        """Return one or more decoded scene records."""
        out: dict[str, Any] | list[dict[str, Any]] = self._backend[at]
        if isinstance(out, list):
            return [self._convert_record(record) for record in out]
        return self._convert_record(out)

    @staticmethod
    def _convert_record(record: dict[str, Any]) -> SceneRecord:
        return decode_mds_sample(record)
