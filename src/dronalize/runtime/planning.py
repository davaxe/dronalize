from __future__ import annotations

from typing import TYPE_CHECKING

import dronalize.core.errors as dronalize_exceptions
from dronalize.datasets import get
from dronalize.io.formats import OutputFormat
from dronalize.runtime.config import ConfigResolver, PlanOverrides, load_project_config
from dronalize.runtime.models import DatasetPlan

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from dronalize.core.categories import DatasetSplit
    from dronalize.datasets.registry import DatasetDescriptor
    from dronalize.io.config import SceneSchemaLike
    from dronalize.processing.ingest.splits import SplitModeName
    from dronalize.runtime.config import ResolvedConfig


def plan_dataset(
    *,
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    output_format: str = "mds",
    scene_schema: SceneSchemaLike | None = None,
    config_path: Path | None = None,
    jobs: int | None = None,
    limit: int | None = None,
    split: SplitModeName | None = None,
    read_split: Sequence[DatasetSplit | str] | DatasetSplit | str | None = None,
    ratio: tuple[float, float, float] | None = None,
    gap: int | None = None,
    segments: int | None = None,
    seed: int | None = None,
    include_map: bool | None = None,
    input_dir_exists: bool = True,
) -> DatasetPlan:
    """Resolve user-facing options into a reusable dataset plan."""
    if not input_dir.exists() and input_dir_exists:
        msg = f"Input directory {input_dir} does not exist."
        raise FileNotFoundError(msg)

    descriptor = get(dataset)
    config = _resolve_plan_config(
        descriptor,
        config_path=config_path,
        scene_schema=scene_schema,
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
        output_format=_resolve_output_format(output_format),
        config=config,
        split_request=config.split.with_seed(seed),
        limit=limit,
        seed=seed,
    )


def _resolve_plan_config(
    descriptor: DatasetDescriptor,
    *,
    config_path: Path | None,
    scene_schema: SceneSchemaLike | None,
    jobs: int | None,
    split: SplitModeName | None,
    read_split: Sequence[DatasetSplit | str] | DatasetSplit | str | None,
    ratio: tuple[float, float, float] | None,
    gap: int | None,
    segments: int | None,
    include_map: bool | None,
) -> ResolvedConfig:
    file_config = None if config_path is None else load_project_config(config_path)
    plan_overrides = PlanOverrides(
        scene_schema=scene_schema,
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


def _resolve_output_format(output_format: str) -> OutputFormat:
    try:
        return OutputFormat(output_format)
    except ValueError as exc:
        msg = f"Unsupported output format: {output_format}"
        raise dronalize_exceptions.ConfigurationError(msg) from exc
