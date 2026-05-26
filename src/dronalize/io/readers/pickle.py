"""Framework-neutral readers for pickled raw scene-record exports."""

from __future__ import annotations

import pickle  # noqa: S403
from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.io.base import DatasetReader, SampleT, split_directory_name
from dronalize.io.records import SceneRecord

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.core.categories import DatasetSplit


class PickleReader(DatasetReader[SampleT]):
    """Read `SceneRecord` objects written by the pickle backend."""

    def __init__(
        self,
        path: Path,
        split: DatasetSplit | str | None = None,
        sample_type: type[SampleT] = SceneRecord,
    ) -> None:
        self._sample_type: type[SampleT] = sample_type
        self._path: Path = path / split_directory_name(split)
        self._files: tuple[Path, ...] = tuple(sorted(self._path.glob("*.pkl")))

    @override
    def __len__(self) -> int:
        return len(self._files)

    @override
    def __getitem__(self, at: int) -> SampleT:
        with self._files[at].open("rb") as file:
            record = pickle.load(file)  # noqa: S301
        if not isinstance(record, self._sample_type):
            msg = f"Expected pickled SceneRecord, but got {type(record).__name__}."
            raise TypeError(msg)
        return record
