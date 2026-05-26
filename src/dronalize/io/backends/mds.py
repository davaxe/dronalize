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
import logging
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from dronalize.core.errors import ConfigurationError
from dronalize.core.optional import raise_missing_optional_dependency
from dronalize.io.base import (
    DatasetWriter,
    RecordTransform,
    SceneTransform,
    split_directory_name,
    validate_transform_choice,
)
from dronalize.io.encoding import encode_scene_record
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
    record_transform: RecordTransform[dict[str, Any]] | None,
    scene_transform: SceneTransform[dict[str, Any]] | None,
    sample_columns: dict[str, str] | None,
) -> MDSDatasetWriter:
    return MDSDatasetWriter(
        output_dir=output_dir,
        config=config,
        splits=splits,
        parallel=parallel,
        parallel_group=parallel_group,
        record_transform=record_transform,
        scene_transform=scene_transform,
        sample_columns=sample_columns,
    )


class MDSDatasetWriter(DatasetWriter):
    """Write processed scene samples to MosaicML Streaming shards.

    By default each shard sample contains the standard Dronalize `SceneRecord`
    payload encoded as MDS columns. Advanced callers may provide
    `record_transform` plus `sample_columns` to write custom MDS-compatible
    dictionaries derived from the encoded record, or `scene_transform` plus
    `sample_columns` to bypass record encoding and derive samples directly from
    the runtime `Scene`.

    Parameters
    ----------
    output_dir : Path
        The base output directory for the dataset. The writer will create split
        subdirectories such as `train` or `unsplit` as needed.
    config : OutputPlan
        The output configuration for the dataset, which controls encoding and
        MDS writer options.
    splits : Iterable[DatasetSplit], optional
        The dataset splits to write, e.g. `train` or `unsplit`. If not provided,
        the writer will write to the "unsplit" subdirectory by default.
    parallel : bool
        Whether the writer will be used in a parallel execution context. If True,
        the writer will write to worker-local temporary subdirectories and merge
        the shard indexes at the end of the run.
    parallel_group : int or str, optional
        An optional identifier for the parallel worker group. If not provided,
        the writer will use the current process name as the group identifier for
        parallel execution. This is only relevant if `parallel` is True.
    record_transform : RecordTransform[dict[str, Any]], optional
        A callable that transforms the encoded `SceneRecord` into a dictionary
        of MDS column values to be written as a sample. This is the preferred
        customization hook for users who want to write custom MDS-compatible
        sample.
    scene_transform : SceneTransform[dict[str, Any]], optional
        A callable that transforms the runtime `Scene` directly into a
        dictionary of MDS column values to be written as a sample.
    sample_columns : dict[str, str], optional
        A mapping from sample field names to MDS column names. This is required
        if either `record_transform` or `scene_transform` is provided, and is
        ignored otherwise since the writer will use the default Dronalize MDS
        encoding scheme. See [MosaicML docs](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/basic_dataset_conversion.html)
        for more details on the expected column layout.

    """

    def __init__(
        self,
        output_dir: Path,
        *,
        config: OutputPlan,
        splits: Iterable[DatasetSplit] | None,
        parallel: bool,
        parallel_group: int | str | None = None,
        record_transform: RecordTransform[dict[str, Any]] | None = None,
        scene_transform: SceneTransform[dict[str, Any]] | None = None,
        sample_columns: dict[str, str] | None = None,
    ) -> None:
        validate_transform_choice(
            record_transform=record_transform, scene_transform=scene_transform
        )
        if (record_transform is not None or scene_transform is not None) and sample_columns is None:
            msg = "Custom MDS transforms require `sample_columns`."
            raise ValueError(msg)
        self._base_output_dir: Path = Path(output_dir)
        self._config: OutputPlan = config
        self._splits: tuple[DatasetSplit, ...] | None = (
            tuple(dict.fromkeys(splits)) if splits is not None else None
        )
        self._parallel: bool = parallel
        self._parallel_group: str | int | None = parallel_group
        self._record_transform: RecordTransform[dict[str, Any]] | None = record_transform
        self._scene_transform: SceneTransform[dict[str, Any]] | None = scene_transform
        self._sample_columns: dict[str, str] = (
            sample_columns if sample_columns is not None else mds_columns(config.config.precision)
        )
        self._writers: dict[DatasetSplit | None, MDSWriter] | None = None

    @override
    @classmethod
    def as_factory(
        cls,
        output_dir: Path,
        config: OutputPlan,
        splits: Iterable[DatasetSplit] | None,
        parallel: bool,
        record_transform: RecordTransform[dict[str, Any]] | None = None,
        scene_transform: SceneTransform[dict[str, Any]] | None = None,
        sample_columns: dict[str, str] | None = None,
    ) -> Callable[[int | None], MDSDatasetWriter]:
        """Create a worker-local factory for MDS scene writers."""
        return functools.partial(
            _create_writer,
            output_dir=output_dir,
            config=config,
            splits=splits,
            parallel=parallel,
            record_transform=record_transform,
            scene_transform=scene_transform,
            sample_columns=sample_columns,
        )

    @classmethod
    def _init_writers(
        cls,
        output_dir: Path,
        *,
        splits: tuple[DatasetSplit, ...] | None,
        parallel: bool,
        parallel_group: int | str | None,
        config: OutputPlan,
        sample_columns: dict[str, str],
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
                columns=sample_columns,
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
                sample_columns=self._sample_columns,
            )

        split: DatasetSplit | None = scene.split_assignment
        if split not in self._writers:
            msg = (
                f"Scene {scene.scene_number} belongs to split {split}, "
                "but no writer is configured for this split."
            )
            raise ConfigurationError(msg)

        effective_scene = scene.with_split_assignment(split) if split is not None else scene
        self._writers[split].write(self._make_sample(effective_scene))

    def _make_sample(self, scene: Scene) -> dict[str, Any]:
        if self._scene_transform is not None:
            return dict(self._scene_transform(scene))

        encoded_scene = encode_scene_record(
            scene,
            dtype=self._config.precision(),
            recenter_position=self._config.recenter_positions,
            trajectory_schema=self._config.trajectory_schema,
            default_observation_length=self._config.default_observation_length,
        )
        if self._record_transform is not None:
            return dict(self._record_transform(encoded_scene))
        return dict(encode_mds_sample(encoded_scene))

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
    logger = logging.getLogger("streaming.base.storage.upload")
    old_level = logger.level
    old_disabled = logger.disabled
    try:
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
        with (
            Path(os.devnull).open("w", encoding="utf-8") as devnull,
            redirect_stdout(devnull),
            redirect_stderr(devnull),
        ):
            yield
    finally:
        logger.setLevel(old_level)
        logger.disabled = old_disabled
