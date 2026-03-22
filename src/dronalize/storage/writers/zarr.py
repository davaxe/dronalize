from __future__ import annotations

import atexit
import functools
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import zarr
from typing_extensions import override

from dronalize.storage.encoding import encode_map_from_scene, scene_to_numpy_dict
from dronalize.storage.spec import (
    StorageManifest,
    StorageMapSampleF32,
    StorageMapSampleF64,
    StorageSceneSampleF32,
    StorageSceneSampleF64,
    write_manifest,
)
from dronalize.storage.writers.protocol import SceneWriter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from zarr.hierarchy import Group

    from dronalize.categories import DatasetSplit
    from dronalize.config.loader import LoaderConfig
    from dronalize.config.writer import WriterConfig
    from dronalize.scene import Scene


_SceneArrayData = StorageSceneSampleF32 | StorageSceneSampleF64
_MapArrayData = StorageMapSampleF32 | StorageMapSampleF64


def _create_writer(
    _: int | None,
    *,
    output_dir: Path,
    config: WriterConfig,
    loader_config: LoaderConfig,
    splits: Iterable[DatasetSplit] | None,
    parallel: bool,
    has_map: bool,
    queue: mp.Queue[_QueueItem] | None = None,
) -> ZarrSceneWriter:
    return ZarrSceneWriter(
        output_dir=output_dir,
        config=config,
        loader_config=loader_config,
        splits=splits,
        parallel=parallel,
        has_map=has_map,
        queue=queue,
    )


class ZarrSceneWriter(SceneWriter):
    """Write processed scenes to Zarr with append-friendly array layouts."""

    _main_queue: mp.Queue[_QueueItem] | None = None
    _writer_process: mp.Process | None = None

    def __init__(
        self,
        output_dir: Path,
        *,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: Iterable[DatasetSplit] | None,
        parallel: bool,
        has_map: bool,
        queue: mp.Queue[_QueueItem] | None = None,
    ) -> None:
        self._output_dir: Path = Path(output_dir)
        self._config: WriterConfig = config
        self._loader_config: LoaderConfig = loader_config
        self._splits: tuple[DatasetSplit, ...] | None = (
            tuple(dict.fromkeys(splits)) if splits is not None else None
        )
        self._parallel: bool = parallel
        self._queue: mp.Queue[_QueueItem] | None = queue
        self._roots: dict[DatasetSplit | None, _ZarrRoot] | None = None
        self.manifest: StorageManifest = StorageManifest.from_configs(
            loader_config=loader_config,
            writer_config=config,
            has_map=has_map,
        )

    @classmethod
    def _setup_parallel(
        cls,
        output_dir: Path,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: tuple[DatasetSplit, ...] | None,
        manifest: StorageManifest,
    ) -> None:
        if cls._writer_process is not None:
            return

        cls._main_queue = mp.Queue(maxsize=256)
        args = (cls._main_queue, output_dir, config, loader_config, splits, manifest)
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

    @staticmethod
    def _write_worker(
        queue: mp.Queue[_QueueItem],
        output_dir: Path,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: Iterable[DatasetSplit] | None,
        manifest: StorageManifest,
    ) -> None:
        splits_tuple = tuple(dict.fromkeys(splits)) if splits is not None else None
        roots = _init_zarr_roots(output_dir, config, loader_config, splits_tuple, manifest)
        pending = {split: _PendingBatch() for split in roots}
        totals = {split: _WriteTotals() for split in roots}

        while True:
            message = queue.get()
            if message is None:
                break
            split = message["split"]
            batch = pending[split]
            batch.add(message)
            if batch.should_flush(config):
                totals[split] = _write_batch(roots[split], batch.drain(), totals[split])

        for split, batch in pending.items():
            if batch.items:
                totals[split] = _write_batch(roots[split], batch.drain(), totals[split])

    @classmethod
    @override
    def as_factory(
        cls,
        output_dir: Path,
        config: WriterConfig,
        loader_config: LoaderConfig,
        splits: tuple[DatasetSplit, ...] | None,
        parallel: bool,
        has_map: bool,
    ) -> Callable[[int | None], ZarrSceneWriter]:
        manifest = StorageManifest.from_configs(
            loader_config=loader_config,
            writer_config=config,
            has_map=has_map,
        )
        if parallel:
            cls._setup_parallel(output_dir, config, loader_config, splits, manifest)

        captured_queue = cls._main_queue if parallel else None
        return functools.partial(
            _create_writer,
            output_dir=output_dir,
            config=config,
            loader_config=loader_config,
            splits=splits,
            parallel=parallel,
            has_map=has_map,
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
        sample: _SceneData = {
            "split": effective_split,
            "scene_data": scene_sample,
            "map_data": map_sample,
        }

        if self._parallel:
            if self._queue is None:
                msg = "Writer queue is not initialized. Call setup() before writing scenes."
                raise RuntimeError(msg)
            self._queue.put(sample)
            return True

        if self._roots is None:
            self._roots = _init_zarr_roots(
                self._output_dir,
                self._config,
                self._loader_config,
                self._splits,
                self.manifest,
            )

        root = self._roots[effective_split]
        root.totals = _write_batch(root, [sample], root.totals)
        return True

    @override
    def finish_local(self) -> None:
        self._roots = None

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
    group: Group

    def append(self, data: dict[str, Any]) -> None:
        for key, array in data.items():
            cast("zarr.Array", self.group[key]).append(array, axis=0)


def _meta_schema(config: WriterConfig) -> dict[str, _ArrayArgs]:
    scene_chunk = config.zarr.scene_chunk
    return {
        "scene_number": {"shape": (0,), "dtype": np.int32, "chunks": (scene_chunk,)},
        "global_origin": {"shape": (0, 2), "dtype": np.float64, "chunks": (scene_chunk, 2)},
        "agent_ptr": {"shape": (0,), "dtype": np.int64, "chunks": (scene_chunk,)},
        "map_node_ptr": {"shape": (0,), "dtype": np.int64, "chunks": (scene_chunk,)},
        "map_edge_ptr": {"shape": (0,), "dtype": np.int64, "chunks": (scene_chunk,)},
        "num_agents": {"shape": (0,), "dtype": np.int32, "chunks": (scene_chunk,)},
        "num_map_nodes": {"shape": (0,), "dtype": np.int32, "chunks": (scene_chunk,)},
        "num_map_edges": {"shape": (0,), "dtype": np.int32, "chunks": (scene_chunk,)},
    }


def _agent_schema(
    config: WriterConfig,
    *,
    input_len: int,
    output_len: int,
) -> dict[str, _ArrayArgs]:
    agent_chunk = config.zarr.agent_chunk
    feature_dim = config.feature_dim
    float_type = config.float_dtype
    return {
        "agent_types": {"shape": (0,), "dtype": np.int32, "chunks": (agent_chunk,)},
        "input_features": {
            "shape": (0, input_len, feature_dim),
            "dtype": float_type,
            "chunks": (agent_chunk, input_len, feature_dim),
        },
        "target_features": {
            "shape": (0, output_len, feature_dim),
            "dtype": float_type,
            "chunks": (agent_chunk, output_len, feature_dim),
        },
        "input_mask": {
            "shape": (0, input_len),
            "dtype": np.bool_,
            "chunks": (agent_chunk, input_len),
        },
        "target_mask": {
            "shape": (0, output_len),
            "dtype": np.bool_,
            "chunks": (agent_chunk, output_len),
        },
    }


def _map_schema(config: WriterConfig) -> dict[str, _ArrayArgs]:
    return {
        "node_positions": {
            "shape": (0, 2),
            "dtype": config.float_dtype,
            "chunks": (config.zarr.map_node_chunk, 2),
        },
        "node_types": {
            "shape": (0,),
            "dtype": np.int32,
            "chunks": (config.zarr.map_node_chunk,),
        },
        "edge_indices": {
            "shape": (0, 2),
            "dtype": np.int32,
            "chunks": (config.zarr.map_edge_chunk, 2),
        },
        "edge_types": {
            "shape": (0,),
            "dtype": np.int32,
            "chunks": (config.zarr.map_edge_chunk,),
        },
    }


@dataclass(slots=True)
class _WriteTotals:
    total_agents: int = 0
    total_map_nodes: int = 0
    total_map_edges: int = 0


@dataclass(slots=True)
class _ZarrRoot:
    inner: Group
    manifest: StorageManifest
    config: WriterConfig
    totals: _WriteTotals = field(default_factory=_WriteTotals)

    def __post_init__(self) -> None:
        def _init_group(name: str, schema: dict[str, _ArrayArgs]) -> None:
            group = self.inner.require_group(name)
            for dataset_name, args in schema.items():
                group.require_dataset(dataset_name, exact=True, **args)

        _init_group("meta", _meta_schema(self.config))
        _init_group(
            "agent",
            _agent_schema(
                self.config,
                input_len=self.manifest.input_len,
                output_len=self.manifest.output_len,
            ),
        )
        _init_group("map", _map_schema(self.config))
        self.inner.attrs.update(self.manifest.as_dict())

    def group(self, name: str) -> _ZarrGroupWrapper:
        return _ZarrGroupWrapper(self.inner.require_group(name))


class _SceneData(TypedDict):
    split: DatasetSplit | None
    scene_data: _SceneArrayData
    map_data: _MapArrayData


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
        self.map_nodes += item["map_data"]["map_num_nodes"]
        self.map_edges += item["map_data"]["map_num_edges"]

    def should_flush(self, config: WriterConfig) -> bool:
        return (
            len(self.items) >= config.zarr.scene_chunk
            or self.agents >= config.zarr.agent_chunk
            or self.map_nodes >= config.zarr.map_node_chunk
            or self.map_edges >= config.zarr.map_edge_chunk
        )

    def drain(self) -> list[_SceneData]:
        items, self.items = self.items, []
        self.agents = self.map_nodes = self.map_edges = 0
        return items


def _init_zarr_roots(
    output_dir: Path,
    config: WriterConfig,
    loader_config: LoaderConfig,
    splits: tuple[DatasetSplit, ...] | None,
    manifest: StorageManifest,
) -> dict[DatasetSplit | None, _ZarrRoot]:
    _ = loader_config
    roots: dict[DatasetSplit | None, _ZarrRoot] = {}
    for split in splits or [None]:
        file_name = f"{split.value}.zarr" if split is not None else "all.zarr"
        root_dir = output_dir / file_name
        roots[split] = _ZarrRoot(
            zarr.open_group(root_dir, mode="w"),
            manifest=manifest,
            config=config,
        )
        write_manifest(root_dir, manifest)
    return roots


def _write_batch(
    zarr_root: _ZarrRoot,
    batch: list[_SceneData],
    totals: _WriteTotals,
) -> _WriteTotals:
    scene_data = [item["scene_data"] for item in batch]
    map_data = [item["map_data"] for item in batch]

    num_agents = np.array([scene["num_agents"] for scene in scene_data], dtype=np.int32)
    num_map_nodes = np.array([scene["map_num_nodes"] for scene in map_data], dtype=np.int32)
    num_map_edges = np.array([scene["map_num_edges"] for scene in map_data], dtype=np.int32)

    zarr_root.group("meta").append({
        "scene_number": np.array([scene["scene_number"] for scene in scene_data], dtype=np.int32),
        "global_origin": np.stack([scene["global_origin"] for scene in scene_data]),
        "agent_ptr": _offsets(num_agents, base=totals.total_agents),
        "map_node_ptr": _offsets(num_map_nodes, base=totals.total_map_nodes),
        "map_edge_ptr": _offsets(num_map_edges, base=totals.total_map_edges),
        "num_agents": num_agents,
        "num_map_nodes": num_map_nodes,
        "num_map_edges": num_map_edges,
    })

    zarr_root.group("agent").append({
        "agent_types": np.concatenate([scene["agent_types"] for scene in scene_data], axis=0),
        "input_features": np.concatenate([scene["input_features"] for scene in scene_data], axis=0),
        "target_features": np.concatenate(
            [scene["target_features"] for scene in scene_data],
            axis=0,
        ),
        "input_mask": np.concatenate([scene["input_mask"] for scene in scene_data], axis=0),
        "target_mask": np.concatenate([scene["target_mask"] for scene in scene_data], axis=0),
    })

    zarr_root.group("map").append({
        "node_positions": np.concatenate(
            [scene["map_node_positions"] for scene in map_data],
            axis=0,
        ),
        "node_types": np.concatenate([scene["map_node_types"] for scene in map_data], axis=0),
        "edge_indices": np.concatenate([scene["map_edge_indices"] for scene in map_data], axis=0),
        "edge_types": np.concatenate([scene["map_edge_types"] for scene in map_data], axis=0),
    })

    return _WriteTotals(
        total_agents=totals.total_agents + int(num_agents.sum()),
        total_map_nodes=totals.total_map_nodes + int(num_map_nodes.sum()),
        total_map_edges=totals.total_map_edges + int(num_map_edges.sum()),
    )


def _offsets(
    counts: np.ndarray[Any, np.dtype[np.int32]],
    *,
    base: int,
) -> np.ndarray[Any, np.dtype[np.int64]]:
    if len(counts) == 0:
        return np.array([], dtype=np.int64)
    return base + np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
