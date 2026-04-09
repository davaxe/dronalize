"""Planning helpers that resolve user-facing options into dataset runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dronalize.core.errors as dronalize_exceptions
from dronalize.datasets import get
from dronalize.io.formats import StorageBackend
from dronalize.runtime.config import ConfigResolver, PlanOverrides, load_project_config
from dronalize.runtime.models import DatasetPlan

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from dronalize.core.categories import DatasetSplit
    from dronalize.datasets.registry import DatasetSpec
    from dronalize.io.config import TrajectorySchemaLike
    from dronalize.processing.loading.splits import SplitStrategyName
    from dronalize.runtime.config import ResolvedConfig


def plan_dataset(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    storage_backend: StorageBackend | str = StorageBackend.MDS,
    trajectory_schema: TrajectorySchemaLike | None = None,
    config_path: Path | None = None,
    jobs: int | None = None,
    limit: int | None = None,
    split: SplitStrategyName | None = None,
    read_split: Sequence[DatasetSplit | str] | DatasetSplit | str | None = None,
    ratio: tuple[float, float, float] | None = None,
    gap: int | None = None,
    segments: int | None = None,
    seed: int | None = None,
    include_map: bool | None = None,
    input_dir_exists: bool = True,
) -> DatasetPlan:
    """Resolve user-facing options into a reusable dataset plan.

    Parameters
    ----------
    dataset : str
        The name of the dataset to plan, as registered in the dataset registry.
        For example, `"argoverse1"`, `"interact"`, or `"lyft"`.
    input_dir : Path
        The root directory containing the raw dataset files.
    output_dir : Path
        The directory where processed dataset files should be written.
    storage_backend : StorageBackend, optional
        The storage backend to write the processed dataset in.
    trajectory_schema : TrajectorySchemaLike, optional
        An optional trajectory schema that written scenes should conform to. Some datasets
        might not natively support a given schema. In those cases, the additional
        fields will be derived whenever possible and fail with error when not.
    config_path : Path, optional
        An optional path to TOML configuration file, which will result in
        configuration overrides.
    jobs : int, optional
        The number of parallel jobs to use for processing.
    split : SplitStrategyName, optional
        An optional split strategy to apply when planning the dataset.
    read_split : Sequence[DatasetSplit | str] or DatasetSplit or str, optional
        An optional dataset-defined partition(s) to read when natives splits
        are requested.
    ratio : tuple[float, float, float], optional
        An optional tuple of three floats specifying the train/val/test split.
    seed : int, optional
        An optional random seed to use for split assignment and other randomized
        operations.
    include_map : bool, optional
        Override default behavior of whether to include map data when supported
        by the dataset.

    Other Parameters
    ----------------
    input_dir_exists : bool
        Whether to check for the existence of the input directory. This can be set
        to `False` to defer this check.
    segments : int, optional
        An optional number of segments to divide each source into when using
        shuffled-time split strategy.
    gap : int, optional
        An optional gap (in frames) to apply between split segments when using
        time-based split strategies.
    limit : int, optional
        An optional limit on the number of scenes to process. Mainly useful
        for testing and debugging.

    """
    if not input_dir.exists() and input_dir_exists:
        msg = f"Input directory {input_dir} does not exist."
        raise FileNotFoundError(msg)

    descriptor = get(dataset)
    config = _resolve_plan_config(
        descriptor,
        config_path=config_path,
        trajectory_schema=trajectory_schema,
        jobs=jobs,
        split=split,
        read_split=read_split,
        ratio=ratio,
        gap=gap,
        segments=segments,
        include_map=include_map,
    )

    return DatasetPlan(
        descriptor=descriptor,
        data_root=input_dir,
        output_dir=output_dir,
        storage_backend=_resolve_storage_backend(storage_backend),
        config=config,
        split_request=config.split.with_seed(seed),
        limit=limit,
        seed=seed,
    )


def _resolve_plan_config(
    descriptor: DatasetSpec,
    *,
    config_path: Path | None,
    trajectory_schema: TrajectorySchemaLike | None,
    jobs: int | None,
    split: SplitStrategyName | None,
    read_split: Sequence[DatasetSplit | str] | DatasetSplit | str | None,
    ratio: tuple[float, float, float] | None,
    gap: int | None,
    segments: int | None,
    include_map: bool | None,
) -> ResolvedConfig:
    file_config = None if config_path is None else load_project_config(config_path)
    plan_overrides = PlanOverrides(
        trajectory_schema=trajectory_schema,
        jobs=jobs,
        split=split,
        read_split=read_split,
        ratio=ratio,
        gap=gap,
        segments=segments,
        include_map=include_map,
    )
    return ConfigResolver().resolve(
        descriptor=descriptor, file_config=file_config, cli_overrides=plan_overrides
    )


def _resolve_storage_backend(storage_backend: str) -> StorageBackend:
    try:
        return StorageBackend(storage_backend)
    except ValueError as exc:
        raise dronalize_exceptions.UnsupportedStorageBackendError(
            storage_backend, tuple(f.value for f in StorageBackend)
        ) from exc
