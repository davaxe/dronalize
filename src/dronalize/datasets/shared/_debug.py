from __future__ import annotations

import os
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

import altair as alt

from dronalize.config.models import (
    DatasetConfig,
    OutputConfig,
    SplitConfig,
    effective_scene_window,
)
from dronalize.core.scene.schema import CANONICAL
from dronalize.plot import plot_trajectories, plot_trajectories_on_map
from dronalize.processing.loading.base import NoLoaderOptions
from dronalize.runtime._internal.scene import SceneBuilder

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from dronalize.core.scene import Scene
    from dronalize.datasets.registry import DatasetSpec
    from dronalize.processing.loading.base import BaseSceneLoader
    from dronalize.processing.models import LoaderRequest


def debug_visualize_scenes(
    source: BaseSceneLoader[Any, Any],
    *,
    max_scenes: int = 1,
    skip_scenes: int = 0,
    step: int = 1,
    group_by: str = "id",
    n_groups: int | None = None,
    group_sample_seed: int | None = None,
    highlight_frame: int | Sequence[int] | None = None,
    include_map: bool = True,
    include_map_nodes: bool = False,
    show: bool = True,
    title_prefix: str | None = None,
    trajectory_kwargs: dict[str, Any] | None = None,
    overlay_kwargs: dict[str, Any] | None = None,
) -> list[Scene]:
    """Render a small sample of scenes produced by a loader."""
    _ = alt.renderers.enable("browser")

    trajectory_kwargs = {} if trajectory_kwargs is None else dict(trajectory_kwargs)
    overlay_kwargs = {} if overlay_kwargs is None else dict(overlay_kwargs)
    scenes_list: list[Scene] = []
    history_frames, future_frames, sample_time = effective_scene_window(source.scenes_config)
    builder = SceneBuilder(
        spec=_debug_spec(loader=source, target_schema=CANONICAL.name),
        split_request=source.split_config,
        source_schema=source.native_trajectory_schema(),
        target_schema=CANONICAL,
        history_frames=history_frames,
        future_frames=future_frames,
        sample_time=sample_time,
    )
    scenes: Iterator[Scene] = _iter_debug_scenes(loader=source, builder=builder)
    for scene in islice(scenes, skip_scenes, skip_scenes + max_scenes * step, step):
        title_parts: list[str] = []
        if title_prefix:
            title_parts.append(title_prefix)
        title_parts.append(f"scene {scene.scene_number}")
        if scene.map_key is not None:
            title_parts.append(f"map={scene.map_key}")
        if scene.split_assignment:
            title_parts.append(f"split assignment={scene.split_assignment.value}")
        title = " | ".join(title_parts)
        map_graph = None
        if include_map:
            map_graph = scene.resolve_map()

        if map_graph is not None:
            chart = plot_trajectories_on_map(
                scene.frame,
                map_graph,
                group_by=group_by,
                n_groups=n_groups,
                group_sample_seed=group_sample_seed,
                highlight_frame=highlight_frame,
                include_map_nodes=include_map_nodes,
                title=title,
                **overlay_kwargs,
            )
        else:
            chart = plot_trajectories(
                scene.frame,
                group_by=group_by,
                n_groups=n_groups,
                group_sample_seed=group_sample_seed,
                highlight_frame=highlight_frame,
                title=title,
                **trajectory_kwargs,
            )

        if show:
            chart.show()
        scenes_list.append(scene)
    return scenes_list


def resolve_dataset_root_from_env(
    *default_parts: str, alternatives: Sequence[Sequence[str]] = ()
) -> Path:
    """Resolve a dataset root from `TRAJ_DATA`, preferring existing candidates."""
    path_str = os.environ.get("TRAJ_DATA")
    base = Path() if path_str is None else Path(path_str)

    for parts in (default_parts, *alternatives):
        candidate = base.joinpath(*parts)
        if candidate.exists():
            return candidate

    return base.joinpath(*default_parts)


def debug_descriptor(
    descriptor: DatasetSpec,
    root: Path,
    *,
    max_scenes: int = 3,
    skip_scenes: int = 0,
    step: int = 1,
    request: LoaderRequest | None = None,
) -> list[Scene]:
    """Build a loader from its descriptor and visualize a scene sample."""
    loader_request = request or descriptor.default_loader_request()
    with descriptor.open_resources(root, loader_request) as resources:
        loader = descriptor.build_loader(root=root, request=loader_request, resources=resources)
        return debug_visualize_scenes(
            loader, max_scenes=max_scenes, skip_scenes=skip_scenes, step=step
        )


def _iter_debug_scenes(
    *, loader: BaseSceneLoader[Any, Any], builder: SceneBuilder
) -> Iterator[Scene]:
    scene_number = 0
    for source in loader.all_sources():
        for prepared in builder.prepare_source(loader, source):
            yield builder.create_scene(loader, prepared, source, scene_number)
            scene_number += 1


def _debug_spec(loader: BaseSceneLoader[Any, Any], *, target_schema: str) -> DatasetSpec:
    from dronalize.datasets.registry import DatasetSpec  # noqa: PLC0415

    dataset_block = loader.loader_options
    dataset_payload = (
        None
        if isinstance(dataset_block, NoLoaderOptions)
        else dataset_block.model_dump(exclude_defaults=False)
    )
    return DatasetSpec(
        name=f"debug::{type(loader).__name__}",
        loader_factory=type(loader).unified_factory,
        default_config=DatasetConfig(
            scenes=loader.scenes_config,
            screening=loader.screening_config,
            output=OutputConfig(trajectory_schema=target_schema),
            map=loader.map_config or DatasetConfig.model_fields["map"].get_default(),
            split=SplitConfig(loader.split_config.config)
            if loader.split_config is not None
            else SplitConfig.model_validate({"strategy": "none"}),
            dataset=dataset_payload,
        ),
        native_schema=loader.native_trajectory_schema(),
        dataset_options_model=type(loader).loader_options_model(),
        has_map=loader.map_config is not None,
    )
