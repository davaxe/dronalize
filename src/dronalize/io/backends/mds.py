"""MosaicML Streaming writer backend for persisted scene datasets."""

from __future__ import annotations

import functools
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from dronalize.core.errors import ConfigurationError
from dronalize.core.optional import raise_missing_optional_dependency
from dronalize.io.backends.base import DatasetWriter
from dronalize.io.encoding import encode_map_from_scene, encode_scene_record

try:
    from streaming import MDSWriter
    from streaming.base.util import merge_index
except ModuleNotFoundError as error:
    raise_missing_optional_dependency(error, feature="The MDS scene writer", extra="storage-mds")


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from dronalize.core.categories import DatasetSplit
    from dronalize.core.scene import Scene
    from dronalize.runtime.plans import OutputPlan


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
            split_dir = output_dir / split.value if split else output_dir / "all"
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
                columns=_mds_columns(config.inner.precision),
                compression=config.mds.compression,
                hashes=(list(config.mds.hashes) if config.mds.hashes is not None else None),
                size_limit=config.mds.size_limit,
                exist_ok=config.mds.exist_ok,
            )
        return writers

    @override
    def write(self, scene: Scene) -> bool:
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

        scene = scene.with_split_assignment(split) if split is not None else scene
        scene_sample = encode_scene_record(
            scene,
            dtype=self._config.precision(),
            recenter_position=self._config.recenter_positions,
            trajectory_schema=self._config.trajectory_schema,
        )
        map_sample = encode_map_from_scene(
            scene,
            dtype=self._config.precision(),
            offset=scene_sample["position_offset"] if self._config.recenter_positions else None,
            return_empty=True,
        )

        self._writers[split].write({
            "scene_number": int(scene_sample["scene_number"]),
            "history_frames": scene.history_frames,
            "future_frames": scene.future_frames,
            "position_offset": scene_sample["position_offset"],
            "agent_types": scene_sample["agent_types"],
            "features": scene_sample["features"],
            "mask": scene_sample["mask"].astype(np.uint8),
            **map_sample,
        })
        return True

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
                    merge_index(str(self._base_output_dir / split.value), keep_local=True)
            return
        with _suppress_output():
            merge_index(str(self._base_output_dir / "all"), keep_local=True)


def _mds_columns(dtype: str) -> dict[str, str]:
    """Return the MDS column schema for one encoded scene sample."""
    return {
        "scene_number": "int",
        "history_frames": "int",
        "future_frames": "int",
        "position_offset": "ndarray:float64:2",
        "agent_types": "ndarray:int32",
        "features": f"ndarray:{dtype}",
        "mask": "ndarray:uint8",
        "map_node_positions": f"ndarray:{dtype}",
        "map_edge_indices": "ndarray:int32",
        "map_node_types": "ndarray:int32",
        "map_edge_types": "ndarray:int32",
    }


@contextmanager
def _suppress_output() -> Generator[None, None, None]:
    with (
        Path(os.devnull).open("w", encoding="utf-8") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
    ):
        yield
