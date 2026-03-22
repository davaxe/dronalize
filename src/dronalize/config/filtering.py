from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from dronalize.categories import AgentCategory

if TYPE_CHECKING:
    from collections.abc import Iterable


def _ensure_frozenset_agent(
    v: AgentTypeValue | Iterable[AgentTypeValue] | None,
) -> frozenset[AgentCategory] | None:
    if v is None:
        return None

    # If it's a single item, wrap it in a list first
    if isinstance(v, (str, int, AgentCategory)):
        v = [AgentCategory.from_value(v)]

    normalized_items = [AgentCategory.from_value(item) for item in v]

    return frozenset(normalized_items)


def _ensure_frozenset_int(v: int | Iterable[int] | None) -> frozenset[int] | None:
    if v is None:
        return None

    # If it's a single item, wrap it in a list first
    if isinstance(v, int):
        v = [v]

    normalized_items = [int(item) for item in v]

    return frozenset(normalized_items)


AgentTypeValue = int | str | AgentCategory
FrozenIntSet = Annotated[frozenset[int] | None, BeforeValidator(_ensure_frozenset_int)]
FrozenAgentSet = Annotated[
    frozenset[AgentCategory] | None, BeforeValidator(_ensure_frozenset_agent),
]


class FilteringConfig(BaseModel):
    """Configuration for filtering scenes based on agent validity and scene composition."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
    min_agents: int = Field(default=2, ge=0)
    require_all_valid: bool = Field(default=False)
    require_frames: FrozenIntSet = Field(default=None)
    exclude_agent_categories: FrozenAgentSet = Field(default=None)
    filter_slow_agents: float | None = Field(default=None, ge=0.0)
    min_samples_per_agent: int | None = Field(default=None, ge=1)

    @classmethod
    def create(
        cls,
        *,
        min_agents: int = 2,
        require_all_valid: bool = False,
        require_frames: int | Iterable[int] | None = None,
        exclude_agent_categories: AgentTypeValue | Iterable[AgentTypeValue] | None = None,
        filter_slow_agents: float | None = None,
        min_samples_per_agent: int | None = None,
    ) -> FilteringConfig:
        """Flexible constructor.

        The inputs mirror the field types, but allow for some further
        flexibility in how the user can specify the values. For example,
        `require_frames` can be a single integer or an iterable of integers and
        similarly for `exclude_agent_categories`.

        Parameters
        ----------
        min_agents : int, optional
            Minimum number of agents required in a scene to be valid. Default is
            2.
        require_all_valid : bool, optional
            If True, requires all agents in a scene to have valid positions for
            all time-steps in the scene. Default is False.
        require_frames : int or iterable of int, optional
            Specific frames offset required for the agent to be considered
            valid. Can be a single integer or an iterable of integers. Default
            is None (no specific frame requirement).
        exclude_agent_categories : str, AgentCategory, or iterable of these, optional
            Set of agent categories to filter out from scenes. Can be a single
            category or an iterable of categories, specified as either strings
            or AgentCategory enums. Default is None (no filtering).
        filter_slow_agents : float, optional
            Filter out agents with an average distance per step below this
            threshold. Default is None (no speed filtering).
        min_samples_per_agent : int, optional
            Minimum number of data points (rows) required per agent. Agents with
            fewer samples are removed before any other validity checks. Default
            is None (no minimum sample requirement).

        """
        # Pack arguments into a dictionary and pass to model_validate.
        # This safely triggers the BeforeValidator logic without static type errors.
        config_dict = {
            "min_agents": min_agents,
            "require_all_valid": require_all_valid,
            "require_frames": require_frames,
            "exclude_agent_categories": exclude_agent_categories,
            "filter_slow_agents": filter_slow_agents,
            "min_samples_per_agent": min_samples_per_agent,
        }

        return cls.model_validate(config_dict)
