"""Framework-neutral readers for Dronalize MDS exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from typing_extensions import TypedDict, Unpack, override

from dronalize.core.optional import raise_missing_optional_dependency
from dronalize.io.base import DatasetReader, SampleT, split_directory_name
from dronalize.io.encoding.mds import decode_mds_sample

try:
    from streaming import Stream, StreamingDataset
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The MDS storage reader", extra="mds")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from dronalize.core.categories import DatasetSplit

__all__ = ["MDSReader", "MDSReaderInitArgs", "Stream"]


class MDSReaderInitArgs(TypedDict, total=False):
    """Optional arguments forwarded to `streaming.StreamingDataset`."""

    download_retry: int
    download_timeout: float
    validate_hash: str
    keep_zip: bool
    epoch_size: int | str | None
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


class MDSReader(DatasetReader[SampleT]):
    """Read raw scene records from an MDS dataset split.

    Parameters
    ----------
    path : Path, optional
        Path to the local MDS dataset directory. Must contain subdirectories for
        the split to read, e.g. `train` or `unsplit`.
    split : DatasetSplit or str, optional
        Dataset split to read, e.g. `train` or `unsplit`. If not provided, the
        reader will attempt to read from the "unsplit" subdirectory by default.
    streams : Sequence[Stream], optional
        Pre-configured MDS `Stream` objects to read from. If not provided, the
        path and split arguments will be used to construct a `StreamingDataset`.
    convert_raw : Callable[[dict[str, Any]], SampleT], optional
        Function to convert raw MDS sample payloads into the desired output
        format. Only useful when customizing the output format; by default, this
        decodes raw MDS samples into `SceneRecord` objects using the standard
        Dronalize MDS encoding scheme.
    reader_args : MDSReaderInitArgs, optional
        Additional keyword arguments forwarded to the `StreamingDataset`
        constructor.
    """

    def __init__(
        self,
        *,
        path: Path | None = None,
        split: DatasetSplit | str | None = None,
        streams: Sequence[Stream] | None = None,
        convert_raw: Callable[[Mapping[str, Any]], SampleT] = decode_mds_sample,
        **reader_args: Unpack[MDSReaderInitArgs],
    ) -> None:
        super().__init__()
        self._convert_record: Callable[[Mapping[str, Any]], SampleT] = convert_raw
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
    def __iter__(self) -> Iterator[SampleT]:
        yield from (self._convert_record(record) for record in self._backend)

    @overload
    def __getitem__(self, at: int) -> SampleT: ...

    @overload
    def __getitem__(self, at: list[int] | npt.NDArray[np.int64] | slice) -> list[SampleT]: ...

    @override
    def __getitem__(
        self, at: int | slice | list[int] | npt.NDArray[np.int64]
    ) -> SampleT | list[SampleT]:
        """Return one or more decoded scene records."""
        out: Mapping[str, Any] | list[Mapping[str, Any]] = self._backend[at]
        if isinstance(out, list):
            return [self._convert_record(record) for record in out]
        return self._convert_record(out)
