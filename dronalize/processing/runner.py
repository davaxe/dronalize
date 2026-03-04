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
from typing import TYPE_CHECKING, Any, Literal, overload

from dronalize.core.protocols.writer import SceneWriter
from dronalize.datasets.registry import DatasetDescriptor, get
from dronalize.processing.config import ConfigDict, load_overrides, resolve_config
from dronalize.processing.parallel import ParallelProcessor, ProgressBar
from dronalize.processing.sequential import SequentialProcessor

if TYPE_CHECKING:
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
    loader_kwargs: dict[str, Any] | None = None,
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
    config_overrides : ConfigDict, optional
        Per-dataset configuration overrides (same shape as one section of the
        TOML config file). Merged on top of the loader's `default_config()`.
        When `None` the default config is used unchanged.
    split : DatasetSplit, optional
        Dataset split to process. Forwarded to the loader constructor. `None`
        means *all data*.
    parallel : bool
        If `True`, wrap the loader in a `ParallelSceneLoader` for multi-process
        execution.
    num_workers : int, optional
        Number of worker processes when *parallel* is `True`. `None` lets
        `ParallelSceneLoader` pick its own default.
    loader_kwargs : dict[str, Any], optional
        Extra keyword arguments forwarded verbatim to the loader factory. Used
        for dataset-specific options such as `file_batch_size` on Argoverse 2.

    """
    if isinstance(descriptor, str):
        descriptor = get(descriptor)

    if isinstance(config_overrides, Path):
        all_overrides = load_overrides(config_overrides)
        config_overrides = all_overrides.get(descriptor.name)

    loader = _build_loader(
        descriptor,
        data_root=data_root,
        config_overrides=config_overrides,
        split=split,
        extra_kwargs=loader_kwargs,
    )

    scene_loader = _maybe_parallelize(loader, parallel=parallel, num_workers=num_workers)


def _build_loader(
    descriptor: DatasetDescriptor,
    *,
    data_root: Path,
    config_overrides: ConfigDict | None,
    split: DatasetSplit | None,
    extra_kwargs: dict[str, Any] | None,
) -> BaseSceneLoader:
    """Instantiate a loader from a descriptor and its resolved config."""
    base_config = descriptor.default_config
    loader_config = resolve_config(base_config, config_overrides or {})

    kwargs: dict[str, Any] = {}
    if loader_config is not None:
        kwargs["loader_config"] = loader_config
    if split is not None:
        kwargs["split"] = split
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    return descriptor.loader_factory(data_root, **kwargs)


@overload
def _maybe_parallelize(
    loader: BaseSceneLoader,
    *,
    parallel: Literal[False],
    num_workers: int | None = None,
) -> SequentialProcessor: ...


@overload
def _maybe_parallelize(
    loader: BaseSceneLoader,
    *,
    parallel: Literal[True],
    num_workers: int | None = None,
) -> ParallelProcessor: ...


def _maybe_parallelize(
    loader: BaseSceneLoader,
    *,
    parallel: bool,
    num_workers: int | None = None,
) -> SequentialProcessor | ParallelProcessor:
    """Optionally wrap a loader in `ParallelSceneLoader`."""
    if not parallel:
        return SequentialProcessor(loader, progress_bar=ProgressBar.SOURCES)

    parallel_kwargs: dict[str, Any] = {"progress_bar": ProgressBar.SOURCES}
    if num_workers is not None:
        parallel_kwargs["processes"] = num_workers

    return ParallelProcessor(loader, **parallel_kwargs)
