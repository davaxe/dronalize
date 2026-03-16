from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import dronalize.exceptions as dronalize_exceptions
import dronalize.execution.common as ex_common
from dronalize.config.config import Config, ConfigSection, load_config, resolve_config
from dronalize.datasets import DatasetDescriptor, get

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from dronalize.categories import DatasetSplit
    from dronalize.config.loader import LoaderConfig
    from dronalize.config.map import MapConfig
    from dronalize.execution.assigner import SplitAssigner
    from dronalize.execution.executor import ObservableWritingExecutor, WriterFactory
    from dronalize.loading import BaseSceneLoader


@dataclass
class ProcessDatasetArgs:
    """Resolved arguments required to process a dataset.

    This dataclass is the prepared, library-facing form of the high-level
    preprocessing request. CLI code and other frontends should prefer
    `prepare_dataset()` / `process_dataset()` and only call `execute()` when
    they already have a fully resolved argument set.
    """

    descriptor: DatasetDescriptor
    data_root: Path
    config: Config
    split_mode: ex_common.SplitMode
    parallel: bool
    limit: int | None
    seed: int | None
    writer: WriterFactory | None = None

    def loader_splits(self) -> list[DatasetSplit] | None:
        """Resolve which predefined splits, if any, the loader should read."""
        if isinstance(self.split_mode, (ex_common.CustomSplit, ex_common.NoSplit)):
            return None
        return self.split_mode.splits(self.descriptor.predefined_splits)

    def writer_splits(self) -> list[DatasetSplit] | None:
        """Resolve which split directories, if any, the writer should create."""
        if isinstance(self.split_mode, ex_common.NoSplit):
            return None
        return self.split_mode.splits(self.descriptor.predefined_splits)

    def split_assigner(self) -> SplitAssigner | None:
        """Build a split assigner for modes that derive splits during writing."""
        if not isinstance(self.split_mode, ex_common.CustomSplit):
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


@dataclass(slots=True)
class ExecutionSession:
    """Executable processing session bound to a live execution scope."""

    executor: ObservableWritingExecutor
    writer_factory: WriterFactory
    split_assigner: SplitAssigner | None = None

    def run(self) -> None:
        """Execute the configured dataset processing run."""
        self.executor.execute(
            writer_factory=self.writer_factory,
            split_assigner=self.split_assigner,
        )


def prepare_dataset(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    split: ex_common.SplitType | Sequence[ex_common.SplitType] | None = None,
    output_format: str = "mds",
    config_path: Path | None = None,
    jobs: int | None = None,
    limit: int | None = None,
    custom_split: tuple[float, float, float] | None = None,
    seed: int | None = None,
) -> ProcessDatasetArgs:
    """Resolve user-facing options into a reusable processing request.

    Parameters
    ----------
    dataset : str
        The name of the dataset to process. Looked up in the global registry.
    input_dir : Path
        Directory containing the raw dataset.
    output_dir : Path
        Directory to save the processed dataset.
    split : DatasetSplit | str | list[DatasetSplit | str] | None, optional
        The predefined split or splits to process. `None` means *all data*.
    output_format : dronalize.execution.types.OutputFormat, optional
        Output format for processed data. `"dummy"` is reserved for internal
        debugging and tests. Default is `"mds"`.
    config_path : Path, optional
        Path to the optional configuration file. If `None`, the default config from
        the dataset descriptor is used without overrides.
    jobs : int or None, optional
        Explicit worker count override. Values greater than 1 enable parallel
        execution; `None` defers to the dataset configuration.
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

    Returns
    -------
    ProcessDatasetArgs
        Resolved dataset processing arguments ready for `execute()`.
    """
    if not input_dir.exists():
        msg = f"Input directory {input_dir} does not exist."
        raise FileNotFoundError(msg)

    split_mode = ex_common.resolve_split_mode(split, custom_split)
    descriptor = get(dataset)

    config_section: ConfigSection = {}
    if config_path is not None:
        all_overrides = load_config(config_path)
        config_section = all_overrides.get(descriptor.name, {})

    config: Config = Config(loader=descriptor.default_config, map=descriptor.default_map_config)
    config = resolve_config(Config, default=config, overrides=config_section)
    if jobs is not None:
        if jobs < 1:
            msg = "jobs must be at least 1."
            raise ValueError(msg)
        config = config.model_copy(
            update={
                "execution": config.execution.model_copy(
                    update={"parallel": jobs > 1, "workers": jobs}
                )
            }
        )

    args = ProcessDatasetArgs(
        descriptor=descriptor,
        data_root=input_dir,
        config=config,
        split_mode=split_mode,
        parallel=config.execution.parallel,
        limit=limit,
        seed=seed,
    )

    args.writer = _get_writer(
        output_format,
        output_dir,
        parallel=args.parallel,
        splits=args.writer_splits(),
    )
    return args


def process_dataset(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    split: ex_common.SplitType | Sequence[ex_common.SplitType] | None = None,
    output_format: str = "mds",
    config_path: Path | None = None,
    jobs: int | None = None,
    limit: int | None = None,
    custom_split: tuple[float, float, float] | None = None,
    seed: int | None = None,
) -> None:
    """Resolve dataset options and execute the processing run."""
    args = prepare_dataset(
        dataset=dataset,
        input_dir=input_dir,
        output_dir=output_dir,
        split=split,
        output_format=output_format,
        config_path=config_path,
        jobs=jobs,
        limit=limit,
        custom_split=custom_split,
        seed=seed,
    )
    execute(args)


@contextmanager
def open_execution(args: ProcessDatasetArgs) -> Iterator[ExecutionSession]:
    """Open an execution session and keep required resources alive for the run."""
    if args.writer is None:
        msg = "WriterFactory must be initialized before processing the dataset."
        raise dronalize_exceptions.ConfigurationError(msg)

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

            executor = ParallelExecutor(
                loader,
                processes=args.config.execution.workers,
                chunksize=args.config.execution.chunksize,
                limit=args.limit,
            )
        else:
            from dronalize.execution.sequential import SequentialExecutor  # noqa: PLC0415

            executor = SequentialExecutor(loader, limit=args.limit)

        yield ExecutionSession(
            executor=executor,
            writer_factory=args.writer,
            split_assigner=split_assigner,
        )


def execute(
    args: ProcessDatasetArgs,
) -> None:
    """Process a prepared dataset request."""
    with open_execution(args) as session:
        session.run()


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
    unsupported_splits: list[DatasetSplit] = [
        split for split in splits or [] if split not in descriptor.predefined_splits
    ]
    if unsupported_splits:
        raise dronalize_exceptions.SplitNotSupportedError(descriptor.name, unsupported_splits)
    if splits is not None:
        kwargs["splits"] = splits
    return descriptor.loader_factory(data_root, loader_config, map_config, **kwargs)


def _get_writer(
    output_format: str,
    output_dir: Path,
    *,
    parallel: bool,
    splits: list[DatasetSplit] | None,
) -> WriterFactory:
    """Resolve an output writer factory from a public format identifier."""
    supported_formats = ("mds", "dummy")

    match output_format:
        case "mds":
            # `streaming` imports are heavy, so the writer stays lazy.
            from dronalize.loading.writer.mds import MDSSceneWriter  # noqa: PLC0415

            return MDSSceneWriter.as_factory(
                output_dir,
                parallel=parallel,
                splits=splits,
            )
        case "dummy":
            from dronalize.loading.writer._dummy import DummyWriter  # noqa: PLC0415

            return DummyWriter.as_factory(log=False)
        case _:
            raise dronalize_exceptions.UnsupportedOutputFormatError(
                output_format,
                supported_formats,
            )
