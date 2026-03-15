from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

from dronalize.config.config import Config, ConfigSection, load_config, resolve_config
from dronalize.datasets._registry import DatasetDescriptor, get
from dronalize.execution.common import CustomSplit, NoSplit, SplitMode, resolve_split_mode
from dronalize.execution.progress import ProgressReportingExecutor

if TYPE_CHECKING:
    from pathlib import Path

    from dronalize.categories import DatasetSplit
    from dronalize.config.loader import LoaderConfig
    from dronalize.config.map import MapConfig
    from dronalize.execution.assigner import SplitAssigner
    from dronalize.execution.executor import WriterFactory
    from dronalize.loading import BaseSceneLoader


Split = Literal["train", "val", "test"]
OutputFormat = Literal["mds", "dummy"]


@dataclass
class ProcessDatasetArgs:
    """Arguments required for processing a dataset."""

    descriptor: DatasetDescriptor
    data_root: Path
    config: Config
    split_mode: SplitMode
    parallel: bool
    progress_bar: bool
    limit: int | None
    seed: int | None
    writer: WriterFactory | None = None

    def loader_splits(self) -> list[DatasetSplit] | None:
        """Resolve which predefined splits, if any, the loader should read."""
        if isinstance(self.split_mode, (CustomSplit, NoSplit)):
            return None
        return self.split_mode.splits(self.descriptor.predefined_splits)

    def writer_splits(self) -> list[DatasetSplit] | None:
        """Resolve which split directories, if any, the writer should create."""
        if isinstance(self.split_mode, NoSplit):
            return None
        return self.split_mode.splits(self.descriptor.predefined_splits)

    def split_assigner(self) -> SplitAssigner | None:
        """Build a split assigner for modes that derive splits during writing."""
        if not isinstance(self.split_mode, CustomSplit):
            return None

        from dronalize.execution.assigner import StatelessWeightedAssigner  # noqa: PLC0415

        groups: list[DatasetSplit] | None = self.split_mode.splits()
        if groups is None or len(groups) == 0:
            return None

        return StatelessWeightedAssigner(
            groups=groups,
            weights=self.split_mode.weights(),
            seed=self.seed,
        )


@overload
def entrypoint(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    split: list[str] | None = None,
    output_format: OutputFormat = "mds",
    config_path: Path | None = None,
    jobs: int | None = None,
    progress: bool = True,
    limit: int | None = None,
    custom_split: tuple[float, float, float] | None = None,
    seed: int | None = None,
    run: Literal[True] = True,
) -> None: ...


@overload
def entrypoint(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    split: list[str] | None = None,
    output_format: OutputFormat = "mds",
    config_path: Path | None = None,
    jobs: int | None = None,
    progress: bool = True,
    limit: int | None = None,
    custom_split: tuple[float, float, float] | None = None,
    seed: int | None = None,
    run: Literal[False],
) -> ProcessDatasetArgs: ...


def entrypoint(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    split: list[str] | None = None,
    output_format: OutputFormat = "mds",
    config_path: Path | None = None,
    jobs: int | None = None,
    progress: bool = True,
    limit: int | None = None,
    custom_split: tuple[float, float, float] | None = None,
    seed: int | None = None,
    run: bool = True,
) -> ProcessDatasetArgs | None:
    """Resolve dataset configuration and run the preprocessing entry point.

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
        Explicit worker count override. Values greater than 1 enable parallel
        execution; `None` defers to the dataset configuration.
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
    if not input_dir.exists():
        msg = f"Input directory {input_dir} does not exist."
        raise ValueError(msg)

    split_mode = resolve_split_mode(split, custom_split)
    descriptor = get(dataset)

    config_section: ConfigSection = {}
    if config_path is not None:
        all_overrides = load_config(config_path)
        config_section = all_overrides.get(descriptor.name, {})

    config: Config = Config(loader=descriptor.default_config, map=descriptor.default_map_config)
    config = resolve_config(Config, default=config, overrides=config_section)
    parallel = jobs > 1 if jobs is not None else config.execution.parallel

    args = ProcessDatasetArgs(
        descriptor=descriptor,
        data_root=input_dir,
        config=config,
        split_mode=split_mode,
        parallel=parallel,
        progress_bar=progress,
        limit=limit,
        seed=seed,
    )

    args.writer = _get_writer(
        output_format,
        output_dir,
        parallel=args.parallel,
        splits=args.writer_splits(),
    )

    if not run:
        return args

    return execute(args)


def execute(args: ProcessDatasetArgs) -> None:
    """Process a single dataset end-to-end and write scenes via *writer*.

    Parameters
    ----------
    args : ProcessDatasetArgs
        Dataclass containing all required arguments to execute the processing.
    """
    if args.writer is None:
        msg = "WriterFactory must be initialized before processing the dataset."
        raise ValueError(msg)

    loader_splits = args.loader_splits()
    split_assigner = args.split_assigner()

    with args.descriptor.execution_scope(args.data_root, args.config.loader, args.config.map):
        loader = _build_loader(
            args.descriptor,
            data_root=args.data_root,
            loader_config=args.config.loader,
            map_config=args.config.map,
            splits=loader_splits,
        )
        if args.parallel:
            from dronalize.execution.parallel import ParallelExecutor  # noqa: PLC0415

            ProgressReportingExecutor(
                ParallelExecutor(
                    loader,
                    processes=args.config.execution.workers,
                    chunksize=args.config.execution.chunksize,
                    limit=args.limit,
                ),
                enable=args.progress_bar,
            ).execute(writer_factory=args.writer, split_assigner=split_assigner)
        else:
            from dronalize.execution.sequential import SequentialExecutor  # noqa: PLC0415

            ProgressReportingExecutor(
                SequentialExecutor(loader, limit=args.limit),
                enable=args.progress_bar,
            ).execute(writer_factory=args.writer, split_assigner=split_assigner)


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
        split for split in splits or [] if split not in descriptor.predefined_splits
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
            # mds imports are quite heavy, lazy import help initial startup time
            from dronalize.loading.writer.mds import MDSSceneWriter  # noqa: PLC0415

            return MDSSceneWriter.as_factory(
                output_dir,
                parallel=parallel,
                splits=splits,
            )
        case "dummy":
            from dronalize.loading.writer._dummy import DummyWriter  # noqa: PLC0415

            return DummyWriter.as_factory(log=True)
