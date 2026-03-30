from __future__ import annotations

import os
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

import altair as alt

from dronalize.plot import plot_trajectories, plot_trajectories_on_map

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from dronalize.core.scene import Scene
    from dronalize.datasets.registry import DatasetDescriptor
    from dronalize.processing.ingest.loader import SceneLoader


def debug_visualize_scenes(
    source: SceneLoader,
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
) -> list[alt.TopLevelMixin]:
    _ = alt.renderers.enable("browser")

    trajectory_kwargs = {} if trajectory_kwargs is None else dict(trajectory_kwargs)
    overlay_kwargs = {} if overlay_kwargs is None else dict(overlay_kwargs)
    charts: list[alt.TopLevelMixin] = []

    scenes: Iterator[Scene] = iter(source.scenes())
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
        charts.append(chart)
    return charts


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
    descriptor: DatasetDescriptor,
    root: Path,
    *,
    max_scenes: int = 3,
    skip_scenes: int = 0,
    step: int = 1,
) -> list[alt.TopLevelMixin]:
    """Build a loader from its descriptor and visualize a scene sample."""
    loader_config = descriptor.default_config
    map_config = descriptor.default_map_config
    loader = descriptor.build_loader(
        root, loader_config=loader_config, map_config=map_config, output_schema=None
    )

    with descriptor.execution_scope(root, loader_config, map_config):
        return debug_visualize_scenes(
            loader, max_scenes=max_scenes, skip_scenes=skip_scenes, step=step
        )
