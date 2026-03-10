from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from dronalize.core.datatypes.categories import AgentCategory

if TYPE_CHECKING:
    from collections.abc import Iterable


def _to_frozenset(v: Iterable | str | AgentCategory | None) -> frozenset | None:
    """Coerce single items, lists, or sets into a frozenset."""
    if v is None:
        return None

    if isinstance(v, str):
        return frozenset([AgentCategory.from_str(v)])

    if isinstance(v, AgentCategory):
        return frozenset([v])

    parsed_items = []
    for item in v:
        if isinstance(item, str):
            parsed_items.append(AgentCategory.from_str(item))
        else:
            parsed_items.append(item)

    return frozenset(parsed_items)


class FilteringConfig(BaseModel):
    """Configuration for filtering scenes based on agent validity and scene composition."""

    model_config = ConfigDict(frozen=True)

    min_agents: int = Field(
        default=2, ge=0, description="Minimum number of agents required in a scene to be valid."
    )

    require_all_valid: bool = Field(
        default=False,
        description=(
            "If True, requires all agents in a scene to have valid positions "
            "for all time-steps in the scene."
        ),
    )

    require_frames: Annotated[frozenset[int] | None, BeforeValidator(_to_frozenset)] = Field(
        default=None,
        description="Specific frames offset required for the agent to be considered valid.",
    )

    filter_agent_category: Annotated[
        frozenset[AgentCategory] | None, BeforeValidator(_to_frozenset)
    ] = Field(default=None, description="Set of agent categories to filter out from scenes.")

    filter_slow_agents: float | None = Field(
        default=None,
        ge=0.0,
        description="Filter out agents with an average distance per step below this threshold.",
    )

    min_samples_per_agent: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Minimum number of data points (rows) required per agent. "
            "Agents with fewer samples are removed before any other validity checks."
        ),
    )
