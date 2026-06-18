"""Public screening surface grouped by rule family.

## Import guide

```python
from dronalize.processing import screening
from dronalize.processing.screening import (
    AgentCategorySelector,
    PassingRequirement,
    ScreeningRuleSet,
    Tolerance,
)
```

This package is organized around three kinds of public symbols:

- root-level screen containers such as
  [`ScreeningRuleSet`][dronalize.processing.screening.ScreeningRuleSet]
- helper functions such as
  [`screen_scene`][dronalize.processing.screening.screen_scene]
- tolerance models for agent-rule aggregation
- grouped rule families exposed through the
  [`cleanup`][dronalize.processing.screening.cleanup],
  [`scene`][dronalize.processing.screening.scene], and
  [`agent`][dronalize.processing.screening.agent] submodules

This keeps the common screening API discoverable without flattening every rule
type into one package namespace.

## Related modules

- [`dronalize.processing`][] for higher-level processing config
- [`dronalize.processing.screening.agent`][] for agent-rule definitions
"""

from dronalize.processing.screening import agent, cleanup, scene
from dronalize.processing.screening.base import (
    AgentCategorySelector,
    CountRange,
    PassingRequirement,
    Tolerance,
)
from dronalize.processing.screening.screen import ScreeningRuleSet

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
