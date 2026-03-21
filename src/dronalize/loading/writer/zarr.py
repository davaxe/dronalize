# pyright: standard
from __future__ import annotations

import atexit
import functools
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypedDict, cast

import numpy as np
import zarr
from typing_extensions import override

from dronalize.loading import SceneWriter
from dronalize.loading.writer.common import (
    FloatDType,
    NumpyMapGraphDict,
    NumpySceneDict,
    encode_map_from_scene,
    scene_to_numpy_dict,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from zarr.hierarchy import Group

    from dronalize.categories import DatasetSplit
    from dronalize.config.loader import LoaderConfig
    from dronalize.config.writer import WriterConfig
    from dronalize.scene import Scene


_SCENE_CHUNK: Final[int] = 4096
_AGENT_CHUNK: Final[int] = 512
_MAP_NODE_CHUNK: Final[int] = 4096
_MAP_EDGE_CHUNK: Final[int] = 4096

_SceneArrayData = NumpySceneDict[np.float32] | NumpySceneDict[np.float64]
_MapArrayData = NumpyMapGraphDict[np.float32] | NumpyMapGraphDict[np.float64]


def _create_writer(
    _: int | None,  # added to match expected signature, even if it is not used
    *,
    output_dir: Path,
    config: WriterConfig,
    loader_config: LoaderConfig,
    splits: tuple[DatasetSplit, ...] | None,
    parallel: bool,
    queue: mp.Queue[_QueueItem] | None = None,
) -> ZarrSceneWriter:
    return ZarrSceneWriter(
        output_dir=output_dir,
        config=config,
        loader_config=loader_config,
        splits=splits,
        parallel=parallel,
        queue=queue,
    )


class ZarrSceneWriter(SceneWriter):
    _main_queue: ClassVar[mp.Queue[_QueueItem] | None] = None
    _zarr_root: ClassVar[dict[DatasetSplit | None, _ZarrRoot]] | None = None
    _writer_process: ClassVar[mp.Process | None] = None

    def __init__(
        self,
        output_dir: Path,
        *,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: tuple[DatasetSplit, ...] | None,
        parallel: bool,
        queue: mp.Queue[_QueueItem] | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._config = config
        self._loader_config = loader_config
        self._splits = splits
        self._parallel = parallel
        self._queue = queue

    @classmethod
    def _setup_parallel(
        cls,
        output_dir: Path,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: tuple[DatasetSplit, ...] | None,
    ) -> None:
        if cls._writer_process is not None:
            return

        cls._main_queue = mp.Queue(maxsize=256)
        args = (cls._main_queue, output_dir, config, loader_config, splits)
        cls._writer_process = mp.Process(target=cls._write_worker, daemon=True, args=args)
        cls._writer_process.start()
        _ = atexit.register(cls._teardown_parallel)

    @classmethod
    def _teardown_parallel(cls) -> None:
        if cls._writer_process is not None and cls._main_queue is not None:
            cls._main_queue.put(None)
            cls._writer_process.join()
            cls._writer_process = None

        if cls._main_queue is not None:
            cls._main_queue.close()
            cls._main_queue.join_thread()
            cls._main_queue = None

        cls._zarr_root = None

    @staticmethod
    def _write_worker(
        queue: mp.Queue[_QueueItem],
        output_dir: Path,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: tuple[DatasetSplit, ...] | None,
    ) -> None:
        zarr_root = ZarrSceneWriter._init_zarr_root(output_dir, config, loader_config, splits)
        pending = {split: _PendingBatch() for split in zarr_root}
        totals = (0, 0, 0)

        while True:
            message = queue.get()
            if message is None:
                break

            batch = pending[message["split"]]
            batch.add(message)
            if batch.should_flush():
                totals = _write_batch(zarr_root[message["split"]], batch.drain(), *totals)

        for split, batch in pending.items():
            if batch.items:
                totals = _write_batch(zarr_root[split], batch.drain(), *totals)

    @staticmethod
    def _init_zarr_root(
        output_dir: Path,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: tuple[DatasetSplit, ...] | None,
    ) -> dict[DatasetSplit | None, _ZarrRoot]:
        zarr_root: dict[DatasetSplit | None, _ZarrRoot] = {}
        for split in splits or [None]:
            file_name = f"{split.value}.zarr" if split is not None else "all.zarr"
            zarr_root[split] = _ZarrRoot(
                zarr.open_group(output_dir / file_name, mode="w"),
                input_len=loader_config.resampled_input_len,
                output_len=loader_config.resampled_output_len,
                feature_dim=config.feature_dim,
                feature_columns=config.feature_columns,
                scene_schema_name=config.scene_schema.name,
                float_type=config.float_dtype,
            )
        return zarr_root

    @classmethod
    @override
    def as_factory(
        cls,
        output_dir: Path,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: tuple[DatasetSplit, ...] | None,
        parallel: bool,
    ) -> Callable[[int | None], ZarrSceneWriter]:
        if parallel:
            cls._setup_parallel(output_dir, config, loader_config, splits)

        captured_queue = cls._main_queue if parallel else None
        # partial is used to support multiprocessing (pickleable), specifically
        # for Windows.
        return functools.partial(
            _create_writer,
            output_dir=output_dir,
            config=config,
            loader_config=loader_config,
            splits=splits,
            parallel=parallel,
            queue=captured_queue,
        )

    @override
    def write(self, scene: Scene, split: DatasetSplit | None = None) -> bool:
        effective_split = (
            split if split is not None else (scene.split_assignment if self._splits else None)
        )
        np_dtype = self._config.float_dtype

        if effective_split is not None:
            scene = scene.override_split_assignment(effective_split)

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
        )

        sample: _SceneData = {
            "split": effective_split,
            "scene_data": numpy_dict,
            "map_data": map_sample,
        }

        if self._parallel:
            if self._queue is not None:
                self._queue.put(sample)
                return True
            msg = "Writer queue is not initialized. Call setup() before writing scenes."
            raise RuntimeError(msg)

        if self._zarr_root is None:
            self._zarr_root = self._init_zarr_root(
                self._output_dir, self._config, self._loader_config, self._splits
            )

        _write_batch(self._zarr_root[effective_split], [sample])
        return True

    @override
    def finish_local(self) -> None: ...

    @override
    def finish_final(self) -> None:
        if self._queue is not None:
            self._queue.put(None)


class _ArrayArgs(TypedDict):
    shape: tuple[int, ...]
    dtype: Any
    chunks: tuple[int, ...]


@dataclass(slots=True, frozen=True)
class _ZarrGroupWrapper:
    """Unified wrapper for appending to any Zarr group."""

    group: Group

    def append(self, data: dict[str, Any]) -> None:
        for key, array in data.items():
            cast("zarr.Array", self.group[key]).append(array, axis=0)


def _meta_schema() -> dict[str, _ArrayArgs]:
    return {
        "scene_number": {"shape": (0,), "dtype": np.int32, "chunks": (_SCENE_CHUNK,)},
        "global_origin": {"shape": (0, 2), "dtype": np.float64, "chunks": (_SCENE_CHUNK, 2)},
        "agent_ptr": {"shape": (0,), "dtype": np.int32, "chunks": (_SCENE_CHUNK,)},
        "map_node_ptr": {"shape": (0,), "dtype": np.int32, "chunks": (_SCENE_CHUNK,)},
        "map_edge_ptr": {"shape": (0,), "dtype": np.int32, "chunks": (_SCENE_CHUNK,)},
        "num_agents": {"shape": (0,), "dtype": np.int32, "chunks": (_SCENE_CHUNK,)},
    }


def _agent_schema(
    input_len: int, output_len: int, feature_dim: int, float_type: FloatDType
) -> dict[str, _ArrayArgs]:
    return {
        "type": {"shape": (0,), "dtype": np.int32, "chunks": (_AGENT_CHUNK,)},
        "input_features": {
            "shape": (0, input_len, feature_dim),
            "dtype": float_type,
            "chunks": (_AGENT_CHUNK, input_len, feature_dim),
        },
        "target_features": {
            "shape": (0, output_len, feature_dim),
            "dtype": float_type,
            "chunks": (_AGENT_CHUNK, output_len, feature_dim),
        },
        "input_mask": {
            "shape": (0, input_len),
            "dtype": np.bool_,
            "chunks": (_AGENT_CHUNK, input_len),
        },
        "target_mask": {
            "shape": (0, output_len),
            "dtype": np.bool_,
            "chunks": (_AGENT_CHUNK, output_len),
        },
    }


def _map_schema(float_type: FloatDType) -> dict[str, _ArrayArgs]:
    return {
        "node_positions": {"shape": (0, 2), "dtype": float_type, "chunks": (_MAP_NODE_CHUNK, 2)},
        "node_types": {"shape": (0,), "dtype": np.int32, "chunks": (_MAP_NODE_CHUNK,)},
        "edge_indices": {"shape": (0, 2), "dtype": np.int32, "chunks": (_MAP_EDGE_CHUNK, 2)},
        "edge_types": {"shape": (0,), "dtype": np.int32, "chunks": (_MAP_EDGE_CHUNK,)},
    }


@dataclass(slots=True)
class _ZarrRoot:
    inner: Group
    input_len: int
    output_len: int
    feature_dim: int
    feature_columns: tuple[str, ...]
    scene_schema_name: str
    float_type: FloatDType

    def __post_init__(self) -> None:
        def _init_group(name: str, schema: dict[str, _ArrayArgs]) -> None:
            group = self.inner.require_group(name)
            for d_name, args in schema.items():
                group.require_dataset(d_name, exact=True, **args)

        _init_group("meta", _meta_schema())
        _init_group(
            "agent",
            _agent_schema(self.input_len, self.output_len, self.feature_dim, self.float_type),
        )
        _init_group("map", _map_schema(self.float_type))

        self.inner.attrs.update(
            input_len=self.input_len,
            output_len=self.output_len,
            feature_dim=self.feature_dim,
            feature_columns=self.feature_columns,
            scene_schema=self.scene_schema_name,
            float_type=str(self.float_type),
        )

    def group(self, name: str) -> _ZarrGroupWrapper:
        return _ZarrGroupWrapper(self.inner.require_group(name))


class _SceneData(TypedDict):
    split: DatasetSplit | None
    map_data: _MapArrayData | None
    scene_data: _SceneArrayData


_QueueItem = _SceneData | None


@dataclass(slots=True)
class _PendingBatch:
    items: list[_SceneData] = field(default_factory=list)
    agents: int = 0
    map_nodes: int = 0
    map_edges: int = 0

    def add(self, item: _SceneData) -> None:
        self.items.append(item)
        self.agents += item["scene_data"]["input_features"].shape[0]
        if item["map_data"] is not None:
            self.map_nodes += item["map_data"]["map_num_nodes"]
            self.map_edges += item["map_data"]["map_num_edges"]

    def should_flush(self) -> bool:
        return (
            len(self.items) >= _SCENE_CHUNK
            or self.agents >= _AGENT_CHUNK
            or self.map_nodes >= _MAP_NODE_CHUNK
            or self.map_edges >= _MAP_EDGE_CHUNK
        )

    def drain(self) -> list[_SceneData]:
        items, self.items = self.items, []
        self.agents = self.map_nodes = self.map_edges = 0
        return items


def _write_batch(
    zarr_root: _ZarrRoot,
    batch: list[_SceneData],
    total_agents: int = 0,
    total_map_nodes: int = 0,
    total_map_edges: int = 0,
) -> tuple[int, int, int]:
    scene_data = [item["scene_data"] for item in batch]
    map_data = [item["map_data"] for item in batch if item["map_data"] is not None]

    num_agents = np.array(
        [scene["input_features"].shape[0] for scene in scene_data], dtype=np.int32
    )
    num_map_nodes = np.array(
        [item["map_data"]["map_num_nodes"] if item["map_data"] else 0 for item in batch],
        dtype=np.int32,
    )
    num_map_edges = np.array(
        [item["map_data"]["map_num_edges"] if item["map_data"] else 0 for item in batch],
        dtype=np.int32,
    )

    zarr_root.group("meta").append({
        "scene_number": np.array([scene["scene_number"] for scene in scene_data], dtype=np.int32),
        "global_origin": np.stack([scene["global_origin"] for scene in scene_data]),
        "agent_ptr": _offsets(num_agents, base=total_agents),
        "map_node_ptr": _offsets(num_map_nodes, base=total_map_nodes),
        "map_edge_ptr": _offsets(num_map_edges, base=total_map_edges),
        "num_agents": num_agents,
    })

    zarr_root.group("agent").append({
        "type": np.concatenate([scene["type"] for scene in scene_data], axis=0),
        "input_features": np.concatenate([scene["input_features"] for scene in scene_data], axis=0),
        "target_features": np.concatenate(
            [scene["target_features"] for scene in scene_data], axis=0
        ),
        "input_mask": np.concatenate([scene["input_mask"] for scene in scene_data], axis=0),
        "target_mask": np.concatenate([scene["target_mask"] for scene in scene_data], axis=0),
    })

    if map_data:
        zarr_root.group("map").append({
            "node_positions": np.concatenate(
                [data["map_node_positions"] for data in map_data], axis=0
            ),
            "node_types": np.concatenate([data["map_node_types"] for data in map_data], axis=0),
            "edge_indices": np.concatenate([data["map_edge_indices"] for data in map_data], axis=0),
            "edge_types": np.concatenate([data["map_edge_types"] for data in map_data], axis=0),
        })

    return (
        total_agents + num_agents.sum(),
        total_map_nodes + num_map_nodes.sum(),
        total_map_edges + num_map_edges.sum(),
    )


def _offsets(
    counts: np.ndarray[Any, np.dtype[np.int32]], *, base: int
) -> np.ndarray[Any, np.dtype[np.int32]]:
    if len(counts) == 0:
        return np.array([], dtype=np.int32)
    return base + np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int32)))
