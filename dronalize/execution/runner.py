from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from dronalize.categories import DatasetSplit
from dronalize.config.config import Config, ConfigSection, load_config, resolve_config
from dronalize.datasets._registry import DatasetDescriptor, get
from dronalize.execution.common import SplitMode, resolve_split_mode
from dronalize.execution.executor import WriterFactory
from dronalize.execution.parallel import ParallelExecutor
from dronalize.execution.progress import ProgressReportingExecutor
from dronalize.execution.sequential import SequentialExecutor
from dronalize.loading.writer.mds import MDSSceneWriter

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.config.loader import LoaderConfig
    from dronalize.config.map import MapConfig
    from dronalize.loading import BaseSceneLoader


Split = Literal["train", "val", "test", "all"]
OutputFormat = Literal["mds"]


def process_data_entry(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    split: Split | None = None,
    output_format: OutputFormat = "mds",
    config_path: Path | None = None,
    jobs: int | None = None,
    progress: bool = True,
    limit: int | None = None,
    custom_split: tuple[float, float, float] | None = None,
    seed: int | None = None,
) -> None:
    """Entrypoint for CLI.

    Parameters
    ----------
    dataset : str
        The name of the dataset to process. Looked up in the global registry.
    input_dir : Path
        Directory containing the raw dataset.
    output_dir : Path
        Directory to save the processed dataset.
    split : Split, optional
        The split of the dataset to process. `None` means *all data*.
    output_format : OutputFormat, optional
        Output format for processed data. Default is `"mds"`.
    config_path : Path, optional
        Path to the optional configuration file. If `None`, the default config from
        the dataset descriptor is used without overrides.
    jobs : int or None, optional
        Number of parallel jobs to run. If `None`, the value from the config is
        used. 1 means no parallelism, -1 means using all available cores.
    progress : bool, optional
        Show progress bar during processing. Default is `True`.
    limit : int or None, optional
        Limit the number of samples to process. Can be useful to test.
    custom_split : tuple[float, float, float], optional
        Ratio to used for train/val/test splits, optional. If provided,
        overrides the predefined splits of the dataset. Should be a tuple of
        three floats that sum to are positive (will be normalized to sum to 1)
        representing the ratios for train, val, and test splits. Putting a ratio
        to zero will exclude that split from the dataset
        completely.
    seed : int, optional
        Seed for non-deterministic operations, such as random splitting. If
        `None`, no seed is set and the behavior is non-deterministic.

    """
    split_mode = resolve_split_mode(split, custom_split)
    descriptor = get(dataset)

    config_section: ConfigSection = {}
    if config_path is not None:
        all_overrides = load_config(config_path)
        config_section = all_overrides.get(descriptor.name, {})

    config: Config = Config(loader=descriptor.default_config, map=descriptor.default_map_config)
    config = resolve_config(Config, default=config, overrides=config_section)
    parallel = jobs > 1 if jobs is not None else config.execution.parallel

    writer_factory = _get_writer(
        output_format,
        output_dir,
        parallel=parallel,
        splits=split_mode.splits(descriptor.predefined_splits),
    )

    process_dataset(
        descriptor,
        data_root=input_dir,
        writer=writer_factory,
        config=config,
        split_mode=split_mode,
        parallel=parallel,
        progress_bar=progress,
    )


def process_dataset(
    descriptor: DatasetDescriptor,
    *,
    data_root: Path,
    writer: WriterFactory,
    config: Config,
    split_mode: SplitMode,
    parallel: bool,
    progress_bar: bool,
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
        finalize callback is used by the underlying execution wrapper.
    config : Config
        The resolved configuration for this dataset. Contains all necessary
        information to instantiate the loader and writer, as well as execution
        parameters.
    """
    with descriptor.execution_scope(data_root, config.loader, config.map):
        loader = _build_loader(
            descriptor,
            data_root=data_root,
            loader_config=config.loader,
            map_config=config.map,
            splits=None,
        )
        if parallel:
            ProgressReportingExecutor(
                ParallelExecutor(
                    loader, processes=config.execution.workers, chunksize=config.execution.chunksize
                ),
                enable=progress_bar,
            ).execute(writer_factory=writer)
        else:
            ProgressReportingExecutor(SequentialExecutor(loader), enable=progress_bar).execute(
                writer_factory=writer
            )


def _build_loader(
    descriptor: DatasetDescriptor,
    *,
    data_root: Path,
    loader_config: LoaderConfig,
    map_config: MapConfig,
    splits: list[DatasetSplit] | None,
) -> BaseSceneLoader[object]:
    """Instantiate a loader from a descriptor and its resolved config."""
    kwargs: dict[str, object] = dict(loader_config.extra_kwargs)
    unsupported_splits = [
        split
        for split in splits or []
        if split is not DatasetSplit.ALL and split not in descriptor.predefined_splits
    ]
    if unsupported_splits:
        msg: str = (
            f"Splits {unsupported_splits} not supported for dataset {descriptor.name}"
            f" (supported splits: {descriptor.predefined_splits})"
        )
        raise ValueError(msg)
    if splits is not None:
        kwargs["splits"] = splits

    return descriptor.loader_factory(data_root, loader_config, map_config, **kwargs)


def _get_writer(
    output_format: OutputFormat,
    output_dir: Path,
    *,
    parallel: bool,
    splits: list[DatasetSplit] | None,
) -> WriterFactory:
    match output_format:
        case "mds":
            return MDSSceneWriter.as_factory(
                output_dir,
                parallel=parallel,
                splits=splits,
            )
