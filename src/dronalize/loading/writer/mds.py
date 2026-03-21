from __future__ import annotations

import functools
import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from streaming import MDSWriter
from streaming.base.util import merge_index
from typing_extensions import Unpack, override

from dronalize.exceptions import ConfigurationError
from dronalize.loading import SceneWriter
from dronalize.loading.writer.common import (
    encode_map_from_scene,
    scene_to_numpy_dict,
)

FLOAT_MDS_DTYPE = "float64"

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.categories import DatasetSplit
    from dronalize.config.loader import LoaderConfig
    from dronalize.config.writer import WriterConfig, WriterPrecision
    from dronalize.scene import Scene


class _MDSWriterArgs(TypedDict, total=False):
    compression: str | None
    hashes: list[str] | None
    size_limit: str | int
    exist_ok: bool


def _create_writer(
    parallel_group: int | None,
    *,
    output_dir: Path,
    config: WriterConfig,
    loader_config: LoaderConfig,
    splits: tuple[DatasetSplit, ...] | None,
    parallel: bool,
    **inner_args: Unpack[_MDSWriterArgs],
) -> MDSSceneWriter:
    return MDSSceneWriter(
        output_dir=output_dir,
        config=config,
        loader_config=loader_config,
        splits=splits,
        parallel=parallel,
        parallel_group=parallel_group,
        **inner_args,
    )


class MDSSceneWriter(SceneWriter):
    """Write processed scenes to MosaicML Streaming (MDS) shards."""

    def __init__(
        self,
        output_dir: Path,
        *,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: tuple[DatasetSplit, ...] | None,
        parallel: bool,
        parallel_group: int | str | None = None,
        **inner_args: Unpack[_MDSWriterArgs],
    ) -> None:
        """Configure a lazily initialized MDS scene writer."""
        self._base_output_dir: Path = Path(output_dir)
        self._config: WriterConfig = config
        self._loader_config: LoaderConfig = loader_config
        self._splits: tuple[DatasetSplit, ...] | None = splits
        self._parallel: bool = parallel
        self._parallel_group: int | str | None = parallel_group
        # Defer writer initialization until the first write call.
        self._writers: dict[DatasetSplit | None, MDSWriter] | None = None
        # Save inner writer args for later use during deferred initialization.
        self._inner_args: _MDSWriterArgs = inner_args

    @override
    @classmethod
    def as_factory(
        cls,
        output_dir: Path,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: tuple[DatasetSplit, ...] | None,
        parallel: bool,
        **inner_args: Unpack[_MDSWriterArgs],
    ) -> Callable[[int | None], MDSSceneWriter]:
        return functools.partial(
            _create_writer,
            output_dir=output_dir,
            config=config,
            loader_config=loader_config,
            splits=splits,
            parallel=parallel,
            **inner_args,
        )

    @staticmethod
    def _init_writers(
        output_dir: Path,
        *,
        splits: tuple[DatasetSplit, ...] | None,
        parallel: bool,
        parallel_group: int | str | None,
        precision: WriterPrecision,
        **inner_args: Unpack[_MDSWriterArgs],
    ) -> dict[DatasetSplit | None, MDSWriter]:
        """Initialize MDSWriters for the given splits and output directory."""
        writers: dict[DatasetSplit | None, MDSWriter] = {}
        for split in splits or [None]:
            split_dir = output_dir / split.value if split else output_dir
            final_dir = (
                split_dir
                if not parallel
                else split_dir
                / str(parallel_group if parallel_group is not None else mp.current_process().name)
            )
            writers[split] = MDSWriter(
                out=str(final_dir), columns=_mds_columns(precision), **inner_args
            )

        return writers

    @override
    def write(
        self,
        scene: Scene,
        split: DatasetSplit | None = None,
    ) -> bool:
        # Resolve the map once per scene (shared across target-agent samples)
        if self._writers is None:
            self._writers = self._init_writers(
                self._base_output_dir,
                splits=self._splits,
                parallel=self._parallel,
                parallel_group=self._parallel_group,
                precision=self._config.precision,
                **self._inner_args,
            )

        np_dtype = self._config.float_dtype
        effective_split = (
            split
            if split is not None
            else scene.split_assignment
            if self._splits is not None
            else None
        )
        if effective_split not in self._writers:
            msg = (
                f"Scene {scene.number} belongs to split {effective_split}, "
                "but no writer is configured for this split."
            )
            raise ConfigurationError(msg)

        scene = (
            scene.override_split_assignment(effective_split)
            if effective_split is not None
            else scene
        )
        numpy_dict = scene_to_numpy_dict(
            scene,
            dtype=np_dtype,
            offset_position=self._config.offset_positions,
            scene_schema=self._config.scene_schema,
        )
        map_sample = encode_map_from_scene(
            scene,
            dtype=np_dtype,
            offset=numpy_dict["global_origin"] if self._config.offset_positions else None,
            return_empty=True,
        )

        self._writers[effective_split].write({
            # -- scalar metadata --
            "scene_number": int(numpy_dict["scene_number"]),
            "num_nodes": int(numpy_dict["num_nodes"]),
            "input_len": self._loader_config.resampled_input_len,
            "output_len": self._loader_config.resampled_output_len,
            "global_origin": numpy_dict["global_origin"],
            # -- agent arrays --
            "type": numpy_dict["type"],
            "input_features": numpy_dict["input_features"],
            "target_features": numpy_dict["target_features"],
            # Cast bool → uint8 (MDS does not support bool arrays)
            "input_mask": numpy_dict["input_mask"].astype(np.uint8),
            "target_mask": numpy_dict["target_mask"].astype(np.uint8),
            "map_num_nodes": map_sample["map_num_nodes"],
            "map_num_edges": map_sample["map_num_edges"],
            "map_node_positions": map_sample["map_node_positions"],
            "map_edge_indices": map_sample["map_edge_indices"],
            "map_node_types": map_sample["map_node_types"],
            "map_edge_types": map_sample["map_edge_types"],
        })
        return True

    @override
    def finish_local(self) -> None:
        if self._writers is None:
            return
        for writer in self._writers.values():
            writer.finish()
        self._writers = None

    @override
    def finish_final(self) -> None:
        if not self._parallel:
            return
        if self._splits:
            for split in self._splits:
                merge_index(str(self._base_output_dir / split.value), keep_local=True)
            return

        merge_index(str(self._base_output_dir), keep_local=True)


def _mds_columns(dtype: WriterPrecision = FLOAT_MDS_DTYPE) -> dict[str, str]:
    """Define the column schema for MDS samples."""
    return {
        # -- scalar metadata --
        "scene_number": "int",
        "num_nodes": "int",
        "input_len": "int",
        "output_len": "int",
        # -- agent arrays (dynamic shape, fixed dtype) --
        "global_origin": "ndarray:float64:2",
        "type": "ndarray:int32",
        "input_features": f"ndarray:{dtype}",
        "target_features": f"ndarray:{dtype}",
        # MDS does not support ndarray:bool — store masks as uint8 (0/1)
        "input_mask": "ndarray:uint8",
        "target_mask": "ndarray:uint8",
        # -- map arrays (dynamic shape, fixed dtype) --
        "map_num_nodes": "int",
        "map_num_edges": "int",
        "map_node_positions": f"ndarray:{dtype}",
        "map_edge_indices": "ndarray:int32",
        "map_node_types": "ndarray:int32",
        "map_edge_types": "ndarray:int32",
    }
