from __future__ import annotations

import atexit
import functools
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NotRequired, TypedDict

import numpy as np
import numpy.typing as npt
import zarr
from typing_extensions import override

from dronalize.storage.encoding import encode_map_from_scene, scene_to_numpy_dict
from dronalize.storage.spec import AnyMapSample, AnySceneSample, StorageManifest, write_manifest
from dronalize.storage.writers.protocol import SceneWriter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from zarr import Group
    from zarr.types import AnyArray

    from dronalize.categories import DatasetSplit
    from dronalize.config.loader import LoaderConfig
    from dronalize.config.writer import WriterConfig
    from dronalize.scene import Scene


def _create_writer(
    _: int | None,
    *,
    output_dir: Path,
    config: WriterConfig,
    loader_config: LoaderConfig,
    splits: Iterable[DatasetSplit] | None,
    parallel: bool,
    has_map: bool,
    queue: mp.Queue[_SceneData | None] | None = None,
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


def _init_zarr_roots(
    output_dir: Path,
    config: WriterConfig,
    splits: tuple[DatasetSplit, ...] | None,
    manifest: StorageManifest,
) -> dict[DatasetSplit | None, _ZarrRoot]:
    roots: dict[DatasetSplit | None, _ZarrRoot] = {}
    for split in splits or (None,):
        root_dir = output_dir / (f"{split.value}.zarr" if split is not None else "all.zarr")
        roots[split] = _ZarrRoot(
            zarr.open_group(root_dir, mode="w"),
            manifest=manifest,
            config=config,
        )
        write_manifest(root_dir, manifest)
    return roots


class ZarrSceneWriter(SceneWriter):
    """Write processed scenes to Zarr with append-friendly array layouts."""

    _main_queue: mp.Queue[_SceneData | None] | None = None
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
        queue: mp.Queue[_SceneData | None] | None = None,
    ) -> None:
        self._output_dir: Path = Path(output_dir)
        self._config: WriterConfig = config
        self._splits: tuple[DatasetSplit, ...] | None = (
            tuple(dict.fromkeys(splits)) if splits is not None else None
        )
        self._parallel: bool = parallel
        self._queue: mp.Queue[_SceneData | None] | None = queue
        self._roots: dict[DatasetSplit | None, _ZarrRoot] | None = None
        self._pending: dict[DatasetSplit | None, _PendingBuffers] | None = None
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

        _ = loader_config
        cls._main_queue = mp.Queue(maxsize=256)
        cls._writer_process = mp.Process(
            target=cls._write_worker,
            daemon=True,
            args=(cls._main_queue, output_dir, config, splits, manifest),
        )
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
        queue: mp.Queue[_SceneData | None],
        output_dir: Path,
        config: WriterConfig,
        splits: tuple[DatasetSplit, ...] | None,
        manifest: StorageManifest,
    ) -> None:
        roots = _init_zarr_roots(output_dir, config, splits, manifest)
        pending = {split: _PendingBuffers(config) for split in roots}

        while True:
            item = queue.get()
            if item is None:
                break
            split = item["split"]
            pending[split].add(item)
            pending[split].flush(roots[split], all_data=False)

        for split, buffers in pending.items():
            buffers.flush(roots[split], all_data=True)

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

        return functools.partial(
            _create_writer,
            output_dir=output_dir,
            config=config,
            loader_config=loader_config,
            splits=splits,
            parallel=parallel,
            has_map=has_map,
            queue=cls._main_queue if parallel else None,
        )

    @override
    def write(self, scene: Scene, split: DatasetSplit | None = None) -> bool:
        effective_split = (
            split if split is not None else (scene.split_assignment if self._splits else None)
        )
        if effective_split is not None:
            scene = scene.override_split_assignment(effective_split)

        scene_sample = scene_to_numpy_dict(
            scene,
            dtype=self._config.float_dtype,
            offset_position=self._config.offset_positions,
            scene_schema=self._config.scene_schema,
        )
        map_sample = encode_map_from_scene(
            scene,
            dtype=self._config.float_dtype,
            offset=scene_sample["global_origin"] if self._config.offset_positions else None,
            return_empty=True,
        )
        item: _SceneData = {
            "split": effective_split,
            "scene_data": scene_sample,
            "map_data": map_sample,
        }

        if self._parallel:
            if self._queue is None:
                msg = "Writer queue is not initialized. Call setup() before writing scenes."
                raise RuntimeError(msg)
            self._queue.put(item)
            return True

        if self._roots is None or self._pending is None:
            self._roots = _init_zarr_roots(
                self._output_dir,
                self._config,
                self._splits,
                self.manifest,
            )
            self._pending = {split_key: _PendingBuffers(self._config) for split_key in self._roots}

        self._pending[effective_split].add(item)
        self._pending[effective_split].flush(self._roots[effective_split], all_data=False)
        return True

    @override
    def finish_local(self) -> None:
        if self._roots is not None and self._pending is not None:
            for split, buffers in self._pending.items():
                buffers.flush(self._roots[split], all_data=True)
        self._roots = None
        self._pending = None

    @override
    def finish_final(self) -> None:
        if self._parallel:
            if self._queue is not None:
                self._queue.put(None)
            return
        self.finish_local()


class _ArrayBuffer:
    def __init__(self, chunk_size: int) -> None:
        self.chunk_size: int = chunk_size
        self.data: dict[str, list[npt.NDArray[Any]]] = {}
        self.length: int = 0

    def add(self, items: dict[str, npt.NDArray[Any]], length: int) -> None:
        if not self.data:
            self.data = {key: [] for key in items}
        for key, value in items.items():
            self.data[key].append(value)
        self.length += length

    def drain(self, *, all_data: bool) -> dict[str, npt.NDArray[Any]] | None:
        if self.length == 0 or (not all_data and self.length < self.chunk_size):
            return None

        flush_len = self.length if all_data else (self.length // self.chunk_size) * self.chunk_size
        merged = {key: np.concatenate(values, axis=0) for key, values in self.data.items()}
        out = {key: value[:flush_len] for key, value in merged.items()}

        if flush_len == self.length:
            self.data = {key: [] for key in self.data}
        else:
            self.data = {key: [value[flush_len:]] for key, value in merged.items()}
        self.length -= flush_len
        return out


class _ArrayArgs(TypedDict):
    shape: tuple[int, ...]
    dtype: Any
    chunks: tuple[int, ...]
    compressor: NotRequired[Any]


class _SceneData(TypedDict):
    split: DatasetSplit | None
    scene_data: AnySceneSample
    map_data: AnyMapSample


@dataclass(slots=True)
class _ZarrRoot:
    inner: Group
    manifest: StorageManifest
    config: WriterConfig
    arrays: dict[str, dict[str, AnyArray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for group_name, schema in _schemas(self.config, self.manifest).items():
            group = self.inner.require_group(group_name)
            group_arrays = self.arrays.setdefault(group_name, {})
            for dataset_name, args in schema.items():
                group_arrays[dataset_name] = group.require_array(
                    dataset_name,
                    exact=True,
                    **args,
                )
        self.inner.attrs.update(self.manifest.json_dict())

    def append(self, group_name: str, data: dict[str, npt.NDArray[Any]]) -> None:
        for name, values in data.items():
            _ = self.arrays[group_name][name].append(values)


class _PendingBuffers:
    _BUFFER_TO_GROUP: Final[dict[str, str]] = {
        "meta": "meta",
        "agent": "agent",
        "map_node": "map",
        "map_edge": "map",
    }

    def __init__(self, config: WriterConfig) -> None:
        self.buffers: dict[str, _ArrayBuffer] = {
            "meta": _ArrayBuffer(config.zarr.scene_chunk),
            "agent": _ArrayBuffer(config.zarr.agent_chunk),
            "map_node": _ArrayBuffer(config.zarr.map_node_chunk),
            "map_edge": _ArrayBuffer(config.zarr.map_edge_chunk),
        }
        self.total_agents: int = 0
        self.total_map_nodes: int = 0
        self.total_map_edges: int = 0

    def add(self, item: _SceneData) -> None:
        scene = item["scene_data"]
        map_data = item["map_data"]

        n_agents = int(scene["num_agents"])
        n_nodes = int(map_data["map_num_nodes"])
        n_edges = int(map_data["map_num_edges"])

        self.buffers["meta"].add(
            {
                "pointers": np.array(
                    [
                        [
                            scene["scene_number"],
                            self.total_agents,
                            self.total_map_nodes,
                            self.total_map_edges,
                        ]
                    ],
                    dtype=np.int64,
                ),
                "counts": np.array([[n_agents, n_nodes, n_edges]], dtype=np.int32),
                "global_origin": np.expand_dims(scene["global_origin"], axis=0),
            },
            length=1,
        )

        self.total_agents += n_agents
        self.total_map_nodes += n_nodes
        self.total_map_edges += n_edges

        if n_agents:
            self.buffers["agent"].add(
                {
                    "agent_types": scene["agent_types"],
                    "features": scene["features"],
                    "mask": scene["mask"],
                },
                length=n_agents,
            )

        if n_nodes:
            self.buffers["map_node"].add(
                {
                    "node_positions": map_data["map_node_positions"],
                    "node_types": map_data["map_node_types"],
                },
                length=n_nodes,
            )

        if n_edges:
            self.buffers["map_edge"].add(
                {
                    "edge_indices": map_data["map_edge_indices"],
                    "edge_types": map_data["map_edge_types"],
                },
                length=n_edges,
            )

    def flush(self, root: _ZarrRoot, *, all_data: bool) -> None:
        for buffer_name, group_name in self._BUFFER_TO_GROUP.items():
            data = self.buffers[buffer_name].drain(all_data=all_data)
            if data is not None:
                root.append(group_name, data)


def _schemas(config: WriterConfig, manifest: StorageManifest) -> dict[str, dict[str, _ArrayArgs]]:
    sequence_length = manifest.input_len + manifest.output_len
    return {
        "meta": {
            "pointers": {
                "shape": (0, 4),
                "dtype": np.int64,
                "chunks": (config.zarr.scene_chunk, 4),
            },
            "counts": {
                "shape": (0, 3),
                "dtype": np.int32,
                "chunks": (config.zarr.scene_chunk, 3),
            },
            "global_origin": {
                "shape": (0, 2),
                "dtype": np.float64,
                "chunks": (config.zarr.scene_chunk, 2),
            },
        },
        "agent": {
            "agent_types": {
                "shape": (0,),
                "dtype": np.int32,
                "chunks": (config.zarr.agent_chunk,),
            },
            "features": {
                "shape": (0, sequence_length, config.feature_dim),
                "dtype": config.float_dtype,
                "chunks": (config.zarr.agent_chunk, manifest.input_len, config.feature_dim),
            },
            "mask": {
                "shape": (0, sequence_length),
                "dtype": np.bool_,
                "chunks": (config.zarr.agent_chunk, manifest.input_len),
            },
        },
        "map": {
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
        },
    }
