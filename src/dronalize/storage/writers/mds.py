from __future__ import annotations

import functools
import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from streaming import MDSWriter
from streaming.base.util import merge_index
from typing_extensions import override

from dronalize.exceptions import ConfigurationError
from dronalize.storage.encoding import encode_map_from_scene, scene_to_numpy_dict
from dronalize.storage.spec import StorageManifest, write_manifest
from dronalize.storage.writers.protocol import SceneWriter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from dronalize.categories import DatasetSplit
    from dronalize.config.loader import LoaderConfig
    from dronalize.config.writer import WriterConfig
    from dronalize.scene import Scene


def _create_writer(
    parallel_group: int | None,
    *,
    output_dir: Path,
    config: WriterConfig,
    loader_config: LoaderConfig,
    splits: Iterable[DatasetSplit] | None,
    parallel: bool,
    has_map: bool,
) -> MDSSceneWriter:
    return MDSSceneWriter(
        output_dir=output_dir,
        config=config,
        loader_config=loader_config,
        splits=splits,
        parallel=parallel,
        parallel_group=parallel_group,
        has_map=has_map,
    )


class MDSSceneWriter(SceneWriter):
    """Write processed scenes to MosaicML Streaming shards."""

    def __init__(
        self,
        output_dir: Path,
        *,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: Iterable[DatasetSplit] | None,
        parallel: bool,
        has_map: bool,
        parallel_group: int | str | None = None,
    ) -> None:
        self._base_output_dir: Path = Path(output_dir)
        self._config: WriterConfig = config
        self._loader_config: LoaderConfig = loader_config
        self._splits: tuple[DatasetSplit, ...] | None = (
            tuple(dict.fromkeys(splits)) if splits is not None else None
        )
        self._parallel: bool = parallel
        self._parallel_group: str | int | None = parallel_group
        self._writers: dict[DatasetSplit | None, MDSWriter] | None = None
        self.manifest: StorageManifest = StorageManifest.from_configs(
            loader_config=loader_config,
            writer_config=config,
            has_map=has_map,
        )

    @override
    @classmethod
    def as_factory(
        cls,
        output_dir: Path,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: Iterable[DatasetSplit] | None,
        parallel: bool,
        has_map: bool,
    ) -> Callable[[int | None], MDSSceneWriter]:
        return functools.partial(
            _create_writer,
            output_dir=output_dir,
            config=config,
            loader_config=loader_config,
            splits=splits,
            parallel=parallel,
            has_map=has_map,
        )

    @staticmethod
    def _init_writers(
        output_dir: Path,
        *,
        splits: tuple[DatasetSplit, ...] | None,
        parallel: bool,
        parallel_group: int | str | None,
        manifest: StorageManifest,
        config: WriterConfig,
    ) -> dict[DatasetSplit | None, MDSWriter]:
        writers: dict[DatasetSplit | None, MDSWriter] = {}
        for split in splits or [None]:
            split_dir = output_dir / split.value if split else output_dir / "all"
            final_dir = (
                split_dir
                if not parallel
                else split_dir
                / str(parallel_group if parallel_group is not None else mp.current_process().name)
            )
            writers[split] = MDSWriter(
                out=(str(final_dir), ""),
                columns=_mds_columns(config.precision),
                compression=config.mds.compression,
                hashes=list(config.mds.hashes) if config.mds.hashes is not None else None,
                size_limit=config.mds.size_limit,
                exist_ok=config.mds.exist_ok,
            )
            write_manifest(split_dir, manifest)
        return writers

    @override
    def write(
        self,
        scene: Scene,
        split: DatasetSplit | None = None,
    ) -> bool:
        if self._writers is None:
            self._writers = self._init_writers(
                self._base_output_dir,
                splits=self._splits,
                parallel=self._parallel,
                parallel_group=self._parallel_group,
                manifest=self.manifest,
                config=self._config,
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
        scene_sample = scene_to_numpy_dict(
            scene,
            dtype=np_dtype,
            offset_position=self._config.offset_positions,
            scene_schema=self._config.scene_schema,
        )
        map_sample = encode_map_from_scene(
            scene,
            dtype=np_dtype,
            offset=scene_sample["global_origin"] if self._config.offset_positions else None,
            return_empty=True,
        )

        self._writers[effective_split].write({
            "scene_number": int(scene_sample["scene_number"]),
            "num_agents": int(scene_sample["num_agents"]),
            "input_len": self._loader_config.resampled_input_len,
            "output_len": self._loader_config.resampled_output_len,
            "global_origin": scene_sample["global_origin"],
            "agent_types": scene_sample["agent_types"],
            "input_features": scene_sample["input_features"],
            "target_features": scene_sample["target_features"],
            "input_mask": scene_sample["input_mask"].astype(np.uint8),
            "target_mask": scene_sample["target_mask"].astype(np.uint8),
            **map_sample,
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
                merge_index((str(self._base_output_dir / split.value), ""), keep_local=True)
            return
        merge_index((str(self._base_output_dir), ""), keep_local=True)


def _mds_columns(dtype: str) -> dict[str, str]:
    """Define the MDS sample schema."""
    return {
        "scene_number": "int",
        "num_agents": "int",
        "input_len": "int",
        "output_len": "int",
        "global_origin": "ndarray:float64:2",
        "agent_types": "ndarray:int32",
        "input_features": f"ndarray:{dtype}",
        "target_features": f"ndarray:{dtype}",
        "input_mask": "ndarray:uint8",
        "target_mask": "ndarray:uint8",
        "map_num_nodes": "int",
        "map_num_edges": "int",
        "map_node_positions": f"ndarray:{dtype}",
        "map_edge_indices": "ndarray:int32",
        "map_node_types": "ndarray:int32",
        "map_edge_types": "ndarray:int32",
    }
