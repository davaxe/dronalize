"""Small shared validation models.

## Import guide

```python
from dronalize.core.models import Range
```

`Range` is used in public configuration and validation models that need an
optional integer minimum and maximum.

## Related modules

- [`dronalize.processing.filtering`][] for filter-specific tolerance models
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, model_validator
from pydantic.config import ConfigDict


class Range(BaseModel):
    """Generic integer range with optional minimum and maximum.

    Parameters
    ----------
    minimum : int | None, optional
        Inclusive lower bound.
    maximum : int | None, optional
        Inclusive upper bound.
    """

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


__all__ = ["Range"]
