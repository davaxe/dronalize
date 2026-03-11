from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from dronalize.core.datatypes.categories import AgentCategory

if TYPE_CHECKING:
    from collections.abc import Iterable


def _ensure_frozenset(v: Any) -> frozenset[Any] | None:
    if v is None:
        return None

    # If it's a single item, wrap it in a list first
    if isinstance(v, (str, int, AgentCategory)):
        v = [v]

    normalized_items = []
    for item in v:
        if isinstance(item, str):
            normalized_items.append(item.lower())
        else:
            normalized_items.append(item)

    return frozenset(normalized_items)


# Type aliases for cleaner fields
FrozenIntSet = Annotated[frozenset[int] | None, BeforeValidator(_ensure_frozenset)]
FrozenAgentSet = Annotated[frozenset[AgentCategory] | None, BeforeValidator(_ensure_frozenset)]


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

    require_frames: FrozenIntSet = Field(
        default=None,
        description="Specific frames offset required for the agent to be considered valid.",
    )

    filter_agent_category: FrozenAgentSet = Field(
        default=None, description="Set of agent categories to filter out from scenes."
    )

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

    @classmethod
    def create(
        cls,
        *,
        min_agents: int = 2,
        require_all_valid: bool = False,
        require_frames: int | Iterable[int] | None = None,
        filter_agent_category: str | AgentCategory | Iterable[str | AgentCategory] | None = None,
        filter_slow_agents: float | None = None,
        min_samples_per_agent: int | None = None,
    ) -> FilteringConfig:
        # Pack arguments into a dictionary and pass to model_validate.
        # This safely triggers the BeforeValidator logic without static type errors.
        config_dict = {
            "min_agents": min_agents,
            "require_all_valid": require_all_valid,
            "require_frames": require_frames,
            "filter_agent_category": filter_agent_category,
            "filter_slow_agents": filter_slow_agents,
            "min_samples_per_agent": min_samples_per_agent,
        }

        return cls.model_validate(config_dict)
