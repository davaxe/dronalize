"""Public screening surface grouped by rule family.

## Import guide

``python
from prejectory.processing import screening
from prejectory.processing.screening import (
    AgentCategorySelector,
    PassingRequirement,
    ScreeningRuleSet,
    Tolerance,
)
``

This package is organized around three kinds of public symbols:

- root-level screen containers such as
  [`ScreeningRuleSet`][prejectory.processing.screening.ScreeningRuleSet]
- tolerance models for agent-rule aggregation
- grouped rule families exposed through the
  `cleanup`,
  `scene`, and
  `agent` submodules

This keeps the common screening API discoverable without flattening every rule
type into one package namespace.

## Related modules

- [`prejectory.processing`][] for higher-level processing config
- `prejectory.processing.screening.agent` for agent-rule definitions
"""

from prejectory.processing.screening import agent, cleanup, scene
from prejectory.processing.screening.base import (
    AgentCategorySelector,
    CountRange,
    PassingRequirement,
    Tolerance,
)
from prejectory.processing.screening.screen import ScreeningRuleSet

__all__ = [
    "AgentCategorySelector",
    "CountRange",
    "PassingRequirement",
    "ScreeningRuleSet",
    "Tolerance",
    "agent",
    "cleanup",
    "scene",
]
