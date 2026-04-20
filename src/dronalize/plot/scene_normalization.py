from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import polars as pl

from dronalize.core.categories import AgentCategory, EdgeType
from dronalize.core.maps import MapGraph
from dronalize.core.scene import Scene

if TYPE_CHECKING:
    from dronalize.io import SceneRecord

SCREENING_PASSED: Final[str] = "Passed"
SCREENING_FAILED: Final[str] = "Failed"
PHASE_HISTORY: Final[str] = "History"
PHASE_FUTURE: Final[str] = "Future"

EDGE_TYPE_LABELS: Final[dict[int, str]] = {
    item.value: item.name.replace("_", " ").title() for item in EdgeType
}
_AGENT_CATEGORY_LABELS: Final[dict[int, str]] = {
    item.value: item.name.replace("_", " ").title() for item in AgentCategory
}


@dataclass(slots=True, frozen=True)
class PlotSceneData:
    """Normalized plotting payload shared across scene-like inputs."""

    scene_number: int
    trajectories: pl.DataFrame
    map_edges: pl.DataFrame | None


def normalize_plot_scene_data(
    data: Scene | SceneRecord,
    *,
    max_agents: int | None = None,
    agent_sample_seed: int | None = None,
    ignore_map_types: set[EdgeType] | None = None,
    ignore_agent_categories: set[AgentCategory] | None = None,
) -> PlotSceneData:
    """Normalize a Scene or SceneRecord into the shared plotting payload."""
    ignored_agents: set[int] = (
        {c.value for c in ignore_agent_categories} if ignore_agent_categories else set()
    )
    ignored_maps: set[int] = {m.value for m in ignore_map_types} if ignore_map_types else set()

    if isinstance(data, Scene):
        return _normalize_scene(data, max_agents, agent_sample_seed, ignored_maps, ignored_agents)

    return _normalize_scene_record(
        data, max_agents, agent_sample_seed, ignored_maps, ignored_agents
    )


def _normalize_scene(
    scene: Scene,
    max_agents: int | None,
    seed: int | None,
    ignored_maps: set[int],
    ignored_agents: set[int],
) -> PlotSceneData:
    """Convert a `Scene` into the normalized plotting payload."""
    traj = scene.frame.select(
        pl.col("frame").cast(pl.Int64()),
        pl.col("id").cast(pl.Int64()).alias("agent"),
        pl.col("x").cast(pl.Float64()),
        pl.col("y").cast(pl.Float64()),
        pl.col("agent_category").cast(pl.Int64()),
    )

    if ignored_agents:
        traj = traj.filter(~pl.col("agent_category").is_in(list(ignored_agents)))

    screening_expr = (
        (
            pl
            .when(pl.col("agent").is_in(list(scene.passed_agent_ids)))
            .then(pl.lit(SCREENING_PASSED))
            .otherwise(pl.lit(SCREENING_FAILED))
        )
        if scene.passed_agent_ids is not None
        else pl.lit(SCREENING_PASSED)
    )

    traj = traj.with_columns(
        screening_status=screening_expr,
        agent_type=pl.col("agent_category").replace_strict(
            _AGENT_CATEGORY_LABELS, default="Unknown"
        ),
    ).drop("agent_category")

    traj = _finalize_trajectories(traj, scene.history_frames, max_agents, seed)
    map_edges = _map_graph_to_edge_frame(scene.resolve_map(), ignored_maps)

    return PlotSceneData(scene.scene_number, traj, map_edges)


def _normalize_scene_record(
    record: SceneRecord,
    max_agents: int | None,
    seed: int | None,
    ignored_maps: set[int],
    ignored_agents: set[int],
) -> PlotSceneData:
    """Convert a `SceneRecord` into the normalized plotting payload."""
    features = np.concatenate((record.input_features, record.output_features), axis=1)
    mask = np.concatenate((record.input_mask, record.output_mask), axis=1)

    if features.shape[-1] < 2:
        msg = "SceneRecord plotting requires x/y feature columns in positions 0 and 1."
        raise ValueError(msg)

    offset = np.asarray(record.position_offset, dtype=np.float64).reshape(-1)
    if offset.shape != (2,):
        msg = f"SceneRecord position_offset must resolve to shape (2,), got {offset.shape!r}."
        raise ValueError(msg)

    num_agents, num_frames = mask.shape
    valid_mask = mask.flatten()

    agent_ids = np.repeat(np.arange(num_agents, dtype=np.int32), num_frames)[valid_mask]
    agent_types = record.agent_types[agent_ids]

    if ignored_agents:
        allowed_mask = ~np.isin(agent_types, list(ignored_agents))
        valid_indices = np.where(valid_mask)[0][allowed_mask]
        agent_ids = agent_ids[allowed_mask]
        agent_types = agent_types[allowed_mask]
    else:
        valid_indices = np.where(valid_mask)[0]

    frame_ids = np.tile(np.arange(num_frames, dtype=np.int32), num_agents)[valid_indices]

    passed_screening = np.where(
        record.passed_agent_mask[agent_ids], SCREENING_PASSED, SCREENING_FAILED
    )
    mapped_types = [_AGENT_CATEGORY_LABELS.get(int(t), "Unknown") for t in agent_types]

    traj = pl.DataFrame({
        "frame": frame_ids,
        "agent": agent_ids,
        "x": features[..., 0].flatten()[valid_indices] + offset[0],
        "y": features[..., 1].flatten()[valid_indices] + offset[1],
        "agent_type": mapped_types,
        "screening_status": passed_screening,
    }).cast({"frame": pl.Int64(), "agent": pl.Int64()})

    traj = _finalize_trajectories(traj, record.input_mask.shape[1], max_agents, seed)

    graph = MapGraph(
        node_positions=np.asarray(record.map_node_positions, dtype=np.float64) + offset,
        edge_indices=np.asarray(record.map_edge_indices, dtype=np.int32),
        node_types=np.asarray(record.map_node_types, dtype=np.int32),
        edge_types=np.asarray(record.map_edge_types, dtype=np.int32),
    )

    return PlotSceneData(record.scene_number, traj, _map_graph_to_edge_frame(graph, ignored_maps))


def _finalize_trajectories(
    traj: pl.DataFrame, history_frames: int, max_agents: int | None, seed: int | None
) -> pl.DataFrame:
    """Apply the shared trajectory post-processing used by all plot inputs."""
    if traj.is_empty():
        return traj.with_columns(
            phase=pl.lit(PHASE_HISTORY),
            segment=pl.lit(0, dtype=pl.Int64),
            total_frames=pl.lit(0, dtype=pl.Int64),
            missing_frames=pl.lit(0, dtype=pl.Int64),
        )

    traj = traj.with_columns(
        phase=pl
        .when(pl.col("frame").rank(method="dense") <= history_frames)
        .then(pl.lit(PHASE_HISTORY))
        .otherwise(pl.lit(PHASE_FUTURE))
    )

    if max_agents is not None and traj.select("agent").n_unique() > max_agents:
        sampled = traj.select("agent").unique().sample(max_agents, seed=seed)
        traj = traj.join(sampled, on="agent", how="semi")

    traj = traj.sort(["agent", "frame"])
    frame_delta = pl.col("frame").diff().fill_null(1).over("agent")

    return traj.with_columns(
        segment=frame_delta.gt(1).cast(pl.Int64).cum_sum().over("agent"),
        total_frames=pl.len().over("agent"),
        missing_frames=(frame_delta - 1).clip(lower_bound=0).sum().over("agent").cast(pl.Int64),
    )


def _map_graph_to_edge_frame(graph: MapGraph | None, ignored_maps: set[int]) -> pl.DataFrame | None:
    """Convert a map graph to the minimal edge dataframe needed by Altair."""
    if not graph or graph.num_edges == 0:
        return None

    edge_mask = (
        ~np.isin(graph.edge_types, list(ignored_maps))
        if ignored_maps
        else np.ones(graph.num_edges, dtype=bool)
    )

    if not np.any(edge_mask):
        return None

    starts, ends = graph.edge_indices[0][edge_mask], graph.edge_indices[1][edge_mask]
    start_pos, end_pos = graph.node_positions[starts], graph.node_positions[ends]

    return pl.DataFrame({
        "x1": start_pos[:, 0],
        "y1": start_pos[:, 1],
        "x2": end_pos[:, 0],
        "y2": end_pos[:, 1],
        "edge_type": graph.edge_types[edge_mask],
    }).with_columns(
        pl.col("edge_type").replace_strict(EDGE_TYPE_LABELS, default=EdgeType.VIRTUAL.name.title()),
        length=((pl.col("x1") - pl.col("x2")) ** 2 + (pl.col("y1") - pl.col("y2")) ** 2).sqrt(),
    )
