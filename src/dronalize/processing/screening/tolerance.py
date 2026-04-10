"""Tolerance models for agent-level screening rules.

## Import guide

```python
from dronalize.processing.screening import Tolerance, tol
from dronalize.processing.screening import AbsoluteTolerance, RelativeTolerance
```

Use these models when an agent rule should allow some invalid agents without
failing the whole scene immediately.

- [`AbsoluteTolerance`][dronalize.processing.screening.AbsoluteTolerance] allows
  a fixed number of invalid agents
- [`RelativeTolerance`][dronalize.processing.screening.RelativeTolerance]
  allows a fraction of invalid agents
- [`CombinedTolerance`][dronalize.processing.screening.CombinedTolerance]
  requires both thresholds to be satisfied

[`tol`][dronalize.processing.screening.tol] is the compact constructor for
programmatic rule definitions.

## Related modules

- [`dronalize.processing.screening.agent`][] for agent rules that consume
  tolerance
- [`dronalize.processing.screening.base`][dronalize.processing.screening.base]
  for the base rule models
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class RelativeTolerance(BaseModel):
    """Relative tolerance for agent-rule failures."""

    kind: Literal["relative"] = Field("relative", repr=False, init=False)
    relative: float = Field(ge=0.0, le=1.0)
    """Fraction of invalid agents to tolerate, between 0 and 1 inclusive."""


class AbsoluteTolerance(BaseModel):
    """Absolute tolerance for agent-rule failures."""

    kind: Literal["absolute"] = Field("absolute", repr=False, init=False)
    absolute: int = Field(ge=0)
    """Number of invalid agents to tolerate, 0 or greater."""


class CombinedTolerance(BaseModel):
    """Combined tolerance for agent-rule failures.

    The stricter of the absolute and relative thresholds is used when evaluating
    a rule with this tolerance. For example, if the relative threshold allows
    10% invalid agents but the absolute threshold allows only 2 invalid agents,
    then at most 2 invalid agents would be tolerated regardless of the total
    number of agents in the scene.

    """

    kind: Literal["combined"] = Field("combined", repr=False, init=False)
    absolute: int = Field(ge=0)
    """Number of invalid agents to tolerate, 0 or greater."""
    relative: float = Field(ge=0.0, le=1.0)
    """Fraction of invalid agents to tolerate, between 0 and 1 inclusive."""


Tolerance = Annotated[
    RelativeTolerance | AbsoluteTolerance | CombinedTolerance, Field(discriminator="kind")
]
"""Discriminated union for screening tolerance configuration."""


def tol(*, relative: float | None = None, absolute: int | None = None) -> Tolerance:
    """Construct a tolerance model from compact numeric arguments.

    This is a convenience function for tolerance. It infers the appropriate
    tolerance model based on which arguments are provided.

    Parameters
    ----------
    relative : float | None, optional
        Maximum invalid-agent fraction to tolerate.
    absolute : int | None, optional
        Maximum number of invalid agents to tolerate.

    Returns
    -------
    Tolerance
        A concrete tolerance model matching the provided arguments.

    Raises
    ------
    ValueError
        If neither `relative` nor `absolute` is provided.
    """
    match (relative, absolute):
        case (None, None):
            msg = "At least one of relative or absolute tolerance must be specified."
            raise ValueError(msg)
        case (None, int(absolute)):
            return AbsoluteTolerance(absolute=absolute)
        case (float(relative) | int(relative), None):
            return RelativeTolerance(relative=relative)
        case (float(relative) | int(relative), int(absolute)):
            return CombinedTolerance(absolute=absolute, relative=relative)


__all__ = ["AbsoluteTolerance", "CombinedTolerance", "RelativeTolerance", "Tolerance", "tol"]
