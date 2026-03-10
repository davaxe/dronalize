from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dronalize.core.protocols.writer import SceneWriter
from dronalize.datasets.registry import DatasetDescriptor, get
from dronalize.processing.config import Config, ConfigDict, load_config, resolve_config
from dronalize.processing.parallel import ParallelProcessor
from dronalize.processing.sequential import SequentialProcessor

if TYPE_CHECKING:
    from dronalize.core.datatypes.loader_config import LoaderConfig
    from dronalize.core.datatypes.split import DatasetSplit
    from dronalize.core.protocols.loader import BaseSceneLoader


WriterFactory = Callable[[int | None], SceneWriter]


def process_dataset(
    descriptor: DatasetDescriptor | str,
    *,
    data_root: Path,
    writer: SceneWriter | WriterFactory,
    config_overrides: ConfigDict | Path | None = None,
    split: DatasetSplit | None = None,
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
        is handed to `writer.write()`. When processing completes, writers are
        finalized via `finish_local()` and `finish_final()`, unless a custom
        finalize callback is used by the underlying processor.
    config_overrides : ConfigDict or Path, optional
        Per-dataset configuration overrides (same shape as one section of the
        TOML config file). Merged on top of the loader's `default_config()`.
        When `None` the default config is used unchanged. If it is a `Path`, the
        entire TOML config file is loaded and the section corresponding to this
        dataset's name is extracted and used as the overrides.
    split : DatasetSplit, optional
        Dataset split to process. Forwarded to the loader constructor. `None`
        means *all data*.
    """
    if isinstance(descriptor, str):
        descriptor = get(descriptor)

    if isinstance(config_overrides, Path):
        all_overrides = load_config(config_overrides)
        config_overrides = all_overrides.get(descriptor.name)

    config_overrides = config_overrides or {}
    config = Config(loader=descriptor.default_config, map=descriptor.default_map_config)
    config = resolve_config(config, config_overrides)

    if config.execution.parallel and isinstance(writer, SceneWriter):
        msg = (
            "When `parallel=True`, `writer` must be a factory function that takes a "
            "process index and returns a `SceneWriter` instance. Use "
            "`SceneWriter.as_factory` classmethod to create a factory function."
        )
        raise ValueError(msg)

    with descriptor.execute_lifecycle_context(data_root, config.loader, config.map):
        loader = _build_loader(
            descriptor,
            data_root=data_root,
            loader_config=config.loader,
            split=split,
        )
        if config.execution.parallel:
            writer = cast("WriterFactory", writer)
            ParallelProcessor(
                loader,
                processes=config.execution.workers,
                chunksize=config.execution.chunksize,
                progress_bar=True,
            ).write_scenes(
                writer_factory=writer,
            )
            return

        if isinstance(writer, Callable):
            writer = writer(None)

        SequentialProcessor(loader, progress_bar=True).write_scenes(writer_factory=lambda: writer)


def _build_loader(
    descriptor: DatasetDescriptor,
    *,
    data_root: Path,
    loader_config: LoaderConfig,
    split: DatasetSplit | None,
) -> BaseSceneLoader:
    """Instantiate a loader from a descriptor and its resolved config."""
    kwargs: dict[str, Any] = dict(loader_config.extra_kwargs)
    kwargs["loader_config"] = loader_config
    if split is not None:
        kwargs["split"] = split

    return descriptor.loader_factory(data_root, **kwargs)
