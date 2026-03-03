from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from dronalize.common.trajectory import T_DataFrame
    from dronalize.core.datatypes.categories import AgentCategory


# TODO: Possibly add filter option based on minimum required valid frames in
# input and output, respectively.


@dataclass(slots=True, frozen=True)
class FilteringConfig:
    """Configuration for filtering scenes based on agent validity and scene composition."""

    min_agents: int = 2
    """Minimum number of agents required in a scene to be valid."""

    require_all_valid: bool = False
    """If True, requires all agents in a scene to have valid positions for all
    time-steps (input and output)."""

    require_prediction_frame: bool = True
    """If True, requires all agents to have valid positions at the first prediction
    frame."""

    require_frames: frozenset[int] | None = None
    """Specific frames offset required for the scene to be considered valid."""

    filter_agent_category: frozenset[AgentCategory] | None = None
    """Set of agent categories to filter out from scenes."""

    filter_slow_agents: float | None = None
    """Filter out agents with an average speed below this threshold."""

    def __init__(
        self,
        min_agents: int = 2,
        *,
        require_all_valid: bool = False,
        require_prediction_frame: bool = True,
        require_frames: Collection[int] | None = None,
        filter_agent_category: Collection[AgentCategory] | None = None,
        filter_slow_agents: float | None = None,
    ) -> None:
        """Initialize and normalise collections to frozensets."""
        object.__setattr__(self, "min_agents", min_agents)
        object.__setattr__(self, "require_all_valid", require_all_valid)
        object.__setattr__(self, "require_prediction_frame", require_prediction_frame)
        object.__setattr__(
            self,
            "require_frames",
            frozenset(require_frames) if require_frames is not None else None,
        )
        object.__setattr__(
            self,
            "filter_agent_category",
            frozenset(filter_agent_category) if filter_agent_category is not None else None,
        )
        object.__setattr__(self, "filter_slow_agents", filter_slow_agents)


def filter_scene(
    data: T_DataFrame,
    config: FilteringConfig | None,
    input_len: int,
    output_len: int,
    sample_time: float = 1.0,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = None,
) -> T_DataFrame:
    """Filter scenes based on configuration.

    Parameters
    ----------
    data : T_DataFrame
        Input DataFrame containing trajectory data.
    config : FilteringConfig
        FilteringConfig object with criteria for scene filtering.
    group_by : str or Sequence[str], optional
        Column name to group by. Each group will be considered an independent scene.
    agent_id : str, optional
        Column name representing the agent ID.
    frame_column : str, optional
        Column name representing the frame index.
    category_column : str, optional
        Column name representing the agent category.

    Returns
    -------
    T_DataFrame
        Filtered DataFrame based on the configuration.

    """
    filter_expr = filter_scene_expr(
        config=config,
        input_len=input_len,
        output_len=output_len,
        sample_time=sample_time,
        group_by=group_by,
        agent_id=agent_id,
        frame_column=frame_column,
        category_column=category_column,
    )
    return data.filter(filter_expr)


def filter_scene_expr(
    config: FilteringConfig | None,
    input_len: int,
    output_len: int,
    sample_time: float = 1.0,
    group_by: str | Sequence[str] | None = None,
    agent_id: str = "id",
    frame_column: str = "frame",
    category_column: str | None = None,
    x_column: str = "x",
    y_column: str = "y",
) -> pl.Expr:
    """Express scene filtering logic as a Polars expression.

    Parameters
    ----------
    config : LoaderConfig
        LoaderConfig object with filtering criteria.
    group_by : str or Sequence[str], optional
        Column name to group by.
    agent_id : str, optional
        Column name representing the agent ID.
    frame_column : str, optional
        Column name representing the frame index.
    category_column : str, optional
        Column name representing the agent category.
    x_column : str, optional
        Column name representing the agent's X position.
    y_column : str, optional
        Column name representing the agent's Y position.

    Returns
    -------
    pl.Expr
        Expression for filtering scenes based on the configuration.

    """
    if config is None:
        return pl.lit(value=True)

    filtering: FilteringConfig = config
    group_by_list = [group_by] if isinstance(group_by, str) else list(group_by or [])

    # Define Windows
    scene_window = group_by_list or pl.lit(1)
    agent_window = [*group_by_list, agent_id] if group_by_list else [agent_id]

    conditions: list[pl.Expr] = []

    # --- 1. Global Scene Validations ---
    scene_start_frame = pl.col(frame_column).min().over(scene_window)

    if filtering.require_frames is not None:
        conditions.append(
            _check_required_frames(
                set(filtering.require_frames), frame_column, scene_start_frame, scene_window
            )
        )

    agent_validity = pl.lit(value=True)
    if filtering.filter_agent_category is not None and category_column is not None:
        agent_validity &= _check_agent_category(
            set(filtering.filter_agent_category), category_column
        )

    agent_validity &= _check_input_overlap(scene_start_frame, input_len, frame_column, agent_window)

    if filtering.require_prediction_frame:
        agent_validity &= _check_prediction_frame(
            scene_start_frame, input_len, frame_column, agent_window
        )

    if filtering.require_all_valid:
        total_len = input_len + output_len
        agent_validity &= _check_full_length(total_len, agent_window)

    if filtering.filter_slow_agents is not None:
        agent_validity &= _check_slow_agents(
            filtering.filter_slow_agents, sample_time, x_column, y_column, agent_window
        )

    conditions.append(agent_validity)
    if filtering.min_agents > 0:
        conditions.append(
            _check_min_agents(filtering.min_agents, agent_id, agent_validity, scene_window)
        )

    if not conditions:
        return pl.lit(value=True)

    return pl.all_horizontal(conditions)


def _check_required_frames(
    require_frames: set[int],
    frame_column: str,
    scene_start_frame: pl.Expr,
    scene_window: pl.Expr | list[str],
) -> pl.Expr:
    """Check if the scene contains all the required relative frame offsets."""
    relative_frame = pl.col(frame_column) - scene_start_frame
    return relative_frame.filter(relative_frame.is_in(require_frames)).n_unique().over(
        scene_window
    ) == len(require_frames)


def _check_agent_category(filter_categories: set[int], category_column: str) -> pl.Expr:
    """Filter out specific agent categories."""
    return ~pl.col(category_column).is_in(filter_categories)


def _check_input_overlap(
    scene_start_frame: pl.Expr, input_len: int, frame_column: str, agent_window: list[str]
) -> pl.Expr:
    """Ensure the agent appears at or before the last input frame."""
    last_input_frame = scene_start_frame + input_len - 1
    agent_start_frame = pl.col(frame_column).min().over(agent_window)
    return agent_start_frame <= last_input_frame


def _check_prediction_frame(
    scene_start_frame: pl.Expr, input_len: int, frame_column: str, agent_window: list[str]
) -> pl.Expr:
    """Ensure the agent exists at the exact first prediction frame."""
    target_frame = scene_start_frame + input_len - 1
    return (pl.col(frame_column) == target_frame).any().over(agent_window)


def _check_full_length(total_len: int, agent_window: list[str]) -> pl.Expr:
    """Ensure the agent's track length matches the total required length."""
    return pl.len().over(agent_window) == total_len


def _check_slow_agents(
    min_speed: float, sample_time: float, x_column: str, y_column: str, agent_window: list[str]
) -> pl.Expr:
    """Filter out agents moving slower than the minimum speed threshold."""
    dx = pl.col(x_column).diff().over(agent_window)
    dy = pl.col(y_column).diff().over(agent_window)
    step_distance = (dx.pow(2) + dy.pow(2)).sqrt().fill_null(0.0)

    total_distance = step_distance.sum().over(agent_window)

    agent_frame_count = pl.len().over(agent_window)
    total_time = (agent_frame_count - 1) * sample_time

    avg_speed = pl.when(total_time > 0).then(total_distance / total_time).otherwise(0.0)
    return avg_speed >= min_speed


def _check_min_agents(
    min_agents: int, agent_id: str, agent_validity: pl.Expr, scene_window: pl.Expr | list[str]
) -> pl.Expr:
    """Ensure the scene meets the minimum count of valid agents."""
    valid_agent_count = pl.col(agent_id).filter(agent_validity).n_unique().over(scene_window)
    return valid_agent_count >= min_agents
