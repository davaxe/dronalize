from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from streaming import MDSWriter
from streaming.base.util import merge_index
from typing_extensions import Self, Unpack, override

from dronalize.converters.numpy import map_graph_to_numpy, scene_to_numpy_dict
from dronalize.exceptions import ConfigurationError
from dronalize.loading import SceneWriter

FLOAT_MDS_DTYPE = "float64"

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.categories import DatasetSplit
    from dronalize.converters.numpy import NumpyMapGraphDict
    from dronalize.scene import Scene


class _MDSWriterArgs(TypedDict, total=False):
    compression: str | None
    hashes: list[str] | None
    size_limit: str | int
    exist_ok: bool


class MDSSceneWriter(SceneWriter):
    """Write processed scenes to MosaicML Streaming (MDS) shards."""

    def __init__(
        self,
        output_dir: Path,
        *,
        splits: list[DatasetSplit] | None = None,
        parallel: bool = False,
        multiple_targets: int | bool = False,
        parallel_group: int | str | None = None,
        **inner_args: Unpack[_MDSWriterArgs],
    ) -> None:
        """Configure a lazily initialized MDS scene writer."""
        self._base_output_dir: Path = Path(output_dir)
        self._multiple_targets: int | bool = multiple_targets
        self._parallel: bool = parallel
        self._parallel_group: int | str | None = parallel_group
        self._splits: list[DatasetSplit] | None = splits
        # Defer writer initialization until the first write call.
        self._writers: dict[DatasetSplit | None, MDSWriter] | None = None
        # Save inner writer args for later use during deferred initialization.
        self._inner_args: _MDSWriterArgs = inner_args

    @override
    @classmethod
    def as_factory(
        cls,
        output_dir: Path,
        *,
        splits: list[DatasetSplit] | None = None,
        parallel: bool = True,
        multiple_targets: int | bool = False,
        **inner_args: Unpack[_MDSWriterArgs],
    ) -> Callable[[int | None], Self]:
        def _factory(worker_id: int | None) -> Self:
            parallel_group = worker_id
            return cls(
                output_dir,
                splits=splits,
                parallel=parallel,
                multiple_targets=multiple_targets,
                parallel_group=parallel_group,
                **inner_args,
            )

        return _factory

    @staticmethod
    def _init_writers(
        output_dir: Path,
        *,
        splits: list[DatasetSplit] | None,
        parallel: bool,
        parallel_group: int | str | None,
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
            writers[split] = MDSWriter(out=str(final_dir), columns=_MDS_COLUMNS, **inner_args)

        return writers

    @override
    def set_output_dir(self, output_dir: Path) -> None:
        self._base_output_dir = output_dir
        if self._parallel:
            self._base_output_dir /= mp.current_process().name

    @override
    def get_output_dir(self) -> Path:
        return self._base_output_dir

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
                **self._inner_args,
            )
        map_sample = self._encode_map(scene)

        effective_split = (
            split
            if split is not None
            else scene.split_assignment
            if self._splits is not None
            else None
        )
        scene = (
            scene.override_split_assignment(effective_split)
            if effective_split is not None
            else scene
        )
        samples = scene_to_numpy_dict(scene, multiple_targets=self._multiple_targets)
        for target_id, numpy_dict in samples.items():
            if effective_split not in self._writers:
                msg = (
                    f"Scene {scene.scene_number} belongs to split {effective_split}, "
                    "but no writer is configured for this split."
                )
                raise ConfigurationError(msg)

            sample: dict[str, Any] = {
                # -- scalar metadata --
                "scene_number": scene.scene_number,
                "target_agent_id": int(target_id),
                "num_nodes": int(numpy_dict["num_nodes"]),
                "ta_index": int(numpy_dict["ta_index"]),
                "input_len": scene.input_len,
                "output_len": scene.output_len,
                # -- agent arrays --
                "type": numpy_dict["type"],
                "inp_pos": numpy_dict["inp_pos"],
                "inp_vel": numpy_dict["inp_vel"],
                "inp_acc": numpy_dict["inp_acc"],
                "inp_yaw": numpy_dict["inp_yaw"],
                "trg_pos": numpy_dict["trg_pos"],
                "trg_vel": numpy_dict["trg_vel"],
                "trg_acc": numpy_dict["trg_acc"],
                "trg_yaw": numpy_dict["trg_yaw"],
                # Cast bool → uint8 (MDS does not support bool arrays)
                "input_mask": numpy_dict["input_mask"].astype(np.uint8),
                "valid_mask": numpy_dict["valid_mask"].astype(np.uint8),
                "ma_mask": numpy_dict["ma_mask"].astype(np.uint8),
                "sa_mask": numpy_dict["sa_mask"].astype(np.uint8),
                # -- map --
                **map_sample,
            }
            self._writers[effective_split].write(sample)

        return len(samples) > 0

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

    @staticmethod
    def _encode_map(scene: Scene) -> NumpyMapGraphDict:
        """Encode the scene's map graph into flat MDS-compatible columns."""
        graph = scene.resolve_map()
        if graph is None:
            return {
                "map_num_nodes": 0,
                "map_num_edges": 0,
                "map_node_positions": _PLACEHOLDER_FLOAT,
                "map_edge_indices": _PLACEHOLDER_I32,
                "map_node_types": _PLACEHOLDER_I32,
                "map_edge_types": _PLACEHOLDER_I32,
            }

        return map_graph_to_numpy(graph)


# -- Column schema ----------------------------------------------------------
# Every field is stored as a native MDS type: int scalars, or ndarray with a
# fixed dtype (but dynamic shape, since agent count N varies per scene).
#
# The trajectory arrays have shape (N, T, D) where T is input_len or
# output_len and D is the feature dimension (2 for pos/vel/acc, 1 for yaw).
# Masks have shape (N, T).  MDS does not support bool arrays, so masks are
# stored as uint8 (0/1).
#
# Map arrays have shape (num_map_nodes, 2), (2, num_map_edges), etc.
# When no map is available, 1-element placeholder arrays are written
# (MDS rejects 0-element arrays) and ``has_map`` is set to 0.

_MDS_COLUMNS = {
    # -- scalar metadata --
    "scene_number": "int",
    "target_agent_id": "int",
    "num_nodes": "int",
    "ta_index": "int",
    "input_len": "int",
    "output_len": "int",
    # -- agent arrays (dynamic shape, fixed dtype) --
    "type": "ndarray:int32",
    "inp_pos": f"ndarray:{FLOAT_MDS_DTYPE}",
    "inp_vel": f"ndarray:{FLOAT_MDS_DTYPE}",
    "inp_acc": f"ndarray:{FLOAT_MDS_DTYPE}",
    "inp_yaw": f"ndarray:{FLOAT_MDS_DTYPE}",
    "trg_pos": f"ndarray:{FLOAT_MDS_DTYPE}",
    "trg_vel": f"ndarray:{FLOAT_MDS_DTYPE}",
    "trg_acc": f"ndarray:{FLOAT_MDS_DTYPE}",
    "trg_yaw": f"ndarray:{FLOAT_MDS_DTYPE}",
    # MDS does not support ndarray:bool — store masks as uint8 (0/1)
    "input_mask": "ndarray:uint8",
    "valid_mask": "ndarray:uint8",
    "ma_mask": "ndarray:uint8",
    "sa_mask": "ndarray:uint8",
    # -- map arrays (dynamic shape, fixed dtype) --
    "map_num_nodes": "int",
    "map_num_edges": "int",
    "map_node_positions": f"ndarray:{FLOAT_MDS_DTYPE}",
    "map_edge_indices": "ndarray:int32",
    "map_node_types": "ndarray:int32",
    "map_edge_types": "ndarray:int32",
}

# 1-element placeholder arrays for map-less samples.
# MDS rejects 0-element arrays, so we use minimal placeholders
# and signal absence via ``has_map == 0``.
_PLACEHOLDER_FLOAT = np.zeros((1,), dtype=np.float64)
_PLACEHOLDER_I32 = np.zeros((1,), dtype=np.int32)
