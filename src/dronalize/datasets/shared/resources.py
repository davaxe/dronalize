"""Shared helpers for dataset resource factories."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.config.sections import MapConfig
from dronalize.core.map_graph import MapGraph
from dronalize.processing.loading.resources import EMPTY_DATASET_RESOURCES, DatasetResources

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable
    from multiprocessing.shared_memory import SharedMemory


MapBuilder = Callable[[Path, MapConfig], MapGraph]


@contextmanager
def open_named_shared_map_resources(
    *,
    map_config: MapConfig | None,
    named_paths: Iterable[tuple[str | None, Path]],
    build_map: MapBuilder,
) -> Generator[DatasetResources, None, None]:
    """Open shared-memory map resources keyed by explicit names."""
    if map_config is None:
        yield EMPTY_DATASET_RESOURCES
        return

    handles: list[SharedMemory] = []
    mappings: dict[str | None, str] = {}
    for key, path in named_paths:
        handle = build_map(path, map_config).to_shared()
        handles.append(handle)
        mappings[key] = handle.name

    try:
        yield DatasetResources(shared_maps=mappings)
    finally:
        for handle in handles:
            handle.close()
            handle.unlink()


@contextmanager
def open_single_shared_map_resource(
    *,
    map_config: MapConfig | None,
    map_path: Path,
    build_map: MapBuilder,
) -> Generator[DatasetResources, None, None]:
    """Open a single shared-memory map resource."""
    if map_config is None:
        yield EMPTY_DATASET_RESOURCES
        return

    handle = build_map(map_path, map_config).to_shared()
    try:
        yield DatasetResources(shared_maps=handle.name)
    finally:
        handle.close()
        handle.unlink()
