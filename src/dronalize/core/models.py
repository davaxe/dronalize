from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic.config import ConfigDict


class Range(BaseModel):
    """Generic integer range with optional minimum and maximum."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    minimum: int | None = None
    maximum: int | None = None

    @model_validator(mode="after")
    def _val_range(self) -> Range:
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            msg = "maximum must be greater than or equal to minimum."
            raise ValueError(msg)
        return self

    def min_max(self) -> tuple[int | None, int | None]:
        """Return minimum and maximum as a tuple."""
        return self.minimum, self.maximum


class RelativeTolerance(BaseModel):
    """Relative tolerance, e.g. fraction of invalid agents to tolerate."""

    kind: Literal["relative"] = Field("relative", repr=False, init=False)
    relative: float = Field(ge=0.0, le=1.0)


class AbsoluteTolerance(BaseModel):
    """Absolute integer tolerance, e.g. number of invalid agents to tolerate."""

    kind: Literal["absolute"] = Field("absolute", repr=False, init=False)
    absolute: int = Field(ge=0)


class CombinedTolerance(BaseModel):
    """Combined tolerance where both must be satisfied."""

    kind: Literal["combined"] = Field("combined", repr=False, init=False)
    absolute: int = Field(ge=0)
    relative: float = Field(ge=0.0, le=1.0)


Tolerance = Annotated[
    RelativeTolerance | AbsoluteTolerance | CombinedTolerance, Field(discriminator="kind")
]


def tol(*, relative: float | None = None, absolute: int | None = None) -> Tolerance:
    """Construct tolerance from simple relative and/or absolute values."""
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
