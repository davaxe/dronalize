"""Mosaic Streaming writer backend.

This backend writes processed scenes as MosaicML Streaming shards. It is the
shard-based option intended for training pipelines that benefit from streaming,
compression, and the upstream `StreamingDataset` reader surface.

Notes
-----
- requires the optional `dronalize[mds]` extra
- writes data into split subdirectories such as `train` or `unsplit`
- when running in parallel, each worker writes to its own temporary subfolder
  and the backend merges the shard indexes at the end of the run
"""

from __future__ import annotations

import functools
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

from typing_extensions import override

from dronalize.core.errors import ConfigurationError
from dronalize.core.optional import raise_missing_optional_dependency
from dronalize.io.base import DatasetWriter, split_directory_name
from dronalize.io.encoding import encode_unsplit_scene_record
from dronalize.io.encoding.mds import encode_mds_sample, mds_columns

try:
    from streaming import MDSWriter
    from streaming.base.util import merge_index
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The MDS scene writer", extra="mds")


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from dronalize.core.categories import DatasetSplit
    from dronalize.core.scene import Scene
    from dronalize.runtime.types import OutputPlan


def _create_writer(
    parallel_group: int | None,
    *,
    output_dir: Path,
    config: OutputPlan,
    splits: Iterable[DatasetSplit] | None,
    parallel: bool,
) -> MDSDatasetWriter:
    return MDSDatasetWriter(
        output_dir=output_dir,
        config=config,
        splits=splits,
        parallel=parallel,
        parallel_group=parallel_group,
    )


class MDSDatasetWriter(DatasetWriter):
    """Write processed scenes to MosaicML Streaming shards."""

    def __init__(
        self,
        output_dir: Path,
        *,
        config: OutputPlan,
        splits: Iterable[DatasetSplit] | None,
        parallel: bool,
        parallel_group: int | str | None = None,
    ) -> None:
        self._base_output_dir: Path = Path(output_dir)
        self._config: OutputPlan = config
        self._splits: tuple[DatasetSplit, ...] | None = (
            tuple(dict.fromkeys(splits)) if splits is not None else None
        )
        self._parallel: bool = parallel
        self._parallel_group: str | int | None = parallel_group
        self._writers: dict[DatasetSplit | None, MDSWriter] | None = None

    @override
    @classmethod
    def as_factory(
        cls,
        output_dir: Path,
        config: OutputPlan,
        splits: Iterable[DatasetSplit] | None,
        parallel: bool,
    ) -> Callable[[int | None], MDSDatasetWriter]:
        """Create a worker-local factory for MDS scene writers."""
        return functools.partial(
            _create_writer, output_dir=output_dir, config=config, splits=splits, parallel=parallel
        )

    @staticmethod
    def _init_writers(
        output_dir: Path,
        *,
        splits: tuple[DatasetSplit, ...] | None,
        parallel: bool,
        parallel_group: int | str | None,
        config: OutputPlan,
    ) -> dict[DatasetSplit | None, MDSWriter]:
        writers: dict[DatasetSplit | None, MDSWriter] = {}
        for split in splits or [None]:
            split_dir = output_dir / split_directory_name(split)
            group_name = (
                parallel_group if parallel_group not in {None, ""} else mp.current_process().name
            )
            final_dir = split_dir if not parallel else split_dir / str(group_name)
            if sys.platform == "win32":
                # MDSWriter cannot handle the C:\ part of a windows path. Below is
                # a workaround that might mess up cloud based paths.
                path_str = "/" + final_dir.relative_to(final_dir.anchor).as_posix()
            else:
                path_str = final_dir.as_posix()
            writers[split] = MDSWriter(
                out=path_str,
                columns=mds_columns(config.config.precision),
                compression=config.mds.compression,
                hashes=(list(config.mds.hashes) if config.mds.hashes is not None else None),
                size_limit=config.mds.size_limit,
                exist_ok=config.mds.exist_ok,
            )
        return writers

    @override
    def write(self, scene: Scene) -> None:
        """Encode and write one scene to the split-specific shard."""
        if self._writers is None:
            self._writers = self._init_writers(
                self._base_output_dir,
                splits=self._splits,
                parallel=self._parallel,
                parallel_group=self._parallel_group,
                config=self._config,
            )

        split: DatasetSplit | None = scene.split_assignment
        if split not in self._writers:
            msg = (
                f"Scene {scene.scene_number} belongs to split {split}, "
                "but no writer is configured for this split."
            )
            raise ConfigurationError(msg)

        encoded_scene = encode_unsplit_scene_record(
            scene.with_split_assignment(split) if split is not None else scene,
            dtype=self._config.precision(),
            recenter_position=self._config.recenter_positions,
            trajectory_schema=self._config.trajectory_schema,
        )
        self._writers[split].write(
            dict(encode_mds_sample(encoded_scene, observation_length=scene.history_frames))
        )

    @override
    def finish_local(self) -> None:
        """Finish the local shard writers for the current worker."""
        if self._writers is None:
            return
        for writer in self._writers.values():
            writer.finish()
        self._writers = None

    @override
    def finish_final(self) -> None:
        """Merge per-worker MDS indices when running in parallel."""
        if not self._parallel:
            return
        if self._splits:
            for split in self._splits:
                with _suppress_output():
                    merge_index(
                        str(self._base_output_dir / split_directory_name(split)), keep_local=True
                    )
            return
        with _suppress_output():
            merge_index(str(self._base_output_dir / split_directory_name(None)), keep_local=True)


@contextmanager
def _suppress_output() -> Generator[None, None, None]:
    with (
        Path(os.devnull).open("w", encoding="utf-8") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):
        yield
