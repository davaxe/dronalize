"""Shared helpers for dataset resource factories."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dronalize.config.models import MapConfig, ScenesConfig
from dronalize.core.maps import MapGraph
from dronalize.datasets.shared import utils
from dronalize.processing.loading.resources import DatasetResources

if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.shared_memory import SharedMemory


MapBuilder = Callable[[Path, MapConfig], MapGraph]
NamedPathsFactory = Callable[[Path], Iterable[tuple[str | None, Path]]]
SinglePathFactory = Callable[[Path], Path]
ResourcesFactory = Callable[
    [Path, ScenesConfig, MapConfig | None], AbstractContextManager[DatasetResources]
]


@contextmanager
def open_named_shared_map_resources(
    *,
    map_config: MapConfig | None,
    named_paths: Iterable[tuple[str | None, Path]],
    build_map: MapBuilder,
) -> Generator[DatasetResources, None, None]:
    """Open shared-memory map resources keyed by explicit names."""
    if map_config is None:
        yield DatasetResources()
        return

    handles: list[SharedMemory] = []
    mappings: dict[str | None, str] = {}
    for key, path in named_paths:
        handle = utils.apply_map_config(build_map(path, map_config), map_config).to_shared()
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
    *, map_config: MapConfig | None, map_path: Path, build_map: MapBuilder
) -> Generator[DatasetResources, None, None]:
    """Open a single shared-memory map resource."""
    if map_config is None:
        yield DatasetResources()
        return

    handle = utils.apply_map_config(build_map(map_path, map_config), map_config).to_shared()
    try:
        yield DatasetResources(shared_maps=handle.name)
    finally:
        handle.close()
        handle.unlink()


def named_shared_map_resources_factory(
    *, named_paths: NamedPathsFactory, build_map: MapBuilder
) -> ResourcesFactory:
    """Return a registry resource factory for named shared maps."""

    @contextmanager
    def _factory(
        root: Path, scenes: ScenesConfig, map_config: MapConfig | None
    ) -> Generator[DatasetResources, None, None]:
        _ = scenes
        with open_named_shared_map_resources(
            map_config=map_config, named_paths=named_paths(root), build_map=build_map
        ) as resources:
            yield resources

    return _factory


def single_shared_map_resource_factory(
    *, map_path: SinglePathFactory, build_map: MapBuilder
) -> ResourcesFactory:
    """Return a registry resource factory for one shared map."""

    @contextmanager
    def _factory(
        root: Path, scenes: ScenesConfig, map_config: MapConfig | None
    ) -> Generator[DatasetResources, None, None]:
        _ = scenes
        with open_single_shared_map_resource(
            map_config=map_config, map_path=map_path(root), build_map=build_map
        ) as resources:
            yield resources

    return _factory
