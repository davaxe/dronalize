"""Orchestration functions for dataset processing.

This module is the central entrypoint that ties together:

- **Registry** — dataset discovery via `DatasetDescriptor`
- **Config** — TOML-based per-dataset configuration overrides
- **Loaders** — `BaseSceneLoader` instances that produce `Scene` objects
- **Writers** — any `SceneWriter`-compatible sink

The two public functions are:

- `process_dataset` — process a single dataset end-to-end.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dronalize.core.protocols.writer import SceneWriter
from dronalize.datasets.registry import DatasetDescriptor, get
from dronalize.processing.config import (
    ConfigDict,
    load_config,
    resolve_loader_config,
    resolve_map_config,
)

if TYPE_CHECKING:
    from dronalize.core.datatypes.loader_config import LoaderConfig
    from dronalize.core.datatypes.split import DatasetSplit
    from dronalize.core.protocols.loader import BaseSceneLoader


WriterFactory = Callable[[str, Path], SceneWriter]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_dataset(
    descriptor: DatasetDescriptor | str,
    *,
    data_root: Path,
    writer: SceneWriter,
    config_overrides: ConfigDict | Path | None = None,
    split: DatasetSplit | None = None,
    parallel: bool = False,
    num_workers: int | None = None,
) -> None:
    """Process a single dataset end-to-end and write scenes via *writer*.

    Parameters
    ----------
    descriptor : DatasetDescriptor or str
        Either a `DatasetDescriptor` instance or the canonical dataset name
        (e.g. `"a43"`, `"ind"`). When a string is given the descriptor is looked
        up in the global registry.
    data_root : Path
        Root directory that contains the raw data for this dataset. Passed as
        the first positional argument to the loader factory.
    writer : SceneWriter
        Any object satisfying the `SceneWriter` protocol. Each produced `Scene`
        is handed to `writer.write()`; after all scenes have been processed
        `writer.finalize()` is called.
    config_overrides : ConfigDict or Path, optional
        Per-dataset configuration overrides (same shape as one section of the
        TOML config file). Merged on top of the loader's `default_config()`.
        When `None` the default config is used unchanged. If it is a `Path`, the
        entire TOML config file is loaded and the section corresponding to this
        dataset's name is extracted and used as the overrides.
    split : DatasetSplit, optional
        Dataset split to process. Forwarded to the loader constructor. `None`
        means *all data*.
    parallel : bool
        If `True`, wrap the loader in a `ParallelSceneLoader` for multi-process
        execution.
    num_workers : int, optional
        Number of worker processes when *parallel* is `True`. `None` lets
        `ParallelSceneLoader` pick its own default.
    """
    if isinstance(descriptor, str):
        descriptor = get(descriptor)

    if isinstance(config_overrides, Path):
        all_overrides = load_config(config_overrides)
        config_overrides = all_overrides.get(descriptor.name)

    config_overrides = config_overrides or {}

    loader_config = resolve_loader_config(
        descriptor.default_config, config_overrides.get("loader", {})
    )
    map_config = resolve_map_config(descriptor.default_map_config, config_overrides.get("map", {}))

    extra_kwargs = config_overrides.get("loader", {}).get("extra_kwargs", None)
    with descriptor.execute_lifecycle_context(data_root, loader_config, map_config):
        loader = _build_loader(
            descriptor,
            data_root=data_root,
            loader_config=loader_config,
            split=split,
            extra_kwargs=extra_kwargs,
        )


def _build_loader(
    descriptor: DatasetDescriptor,
    *,
    data_root: Path,
    loader_config: LoaderConfig,
    split: DatasetSplit | None,
    extra_kwargs: dict[str, Any] | None,
) -> BaseSceneLoader:
    """Instantiate a loader from a descriptor and its resolved config."""
    kwargs: dict[str, Any] = {}
    if loader_config is not None:
        kwargs["loader_config"] = loader_config
    if split is not None:
        kwargs["split"] = split
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    return descriptor.loader_factory(data_root, **kwargs)


if __name__ == "__main__":
    ...
