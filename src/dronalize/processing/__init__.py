"""Grouped processing-facing public API.

## Import guide

```python
from dronalize.processing import maps, pipeline, screening
```

Use this package as a lightweight namespace for the focused processing
subpackages:

- [`screening`][dronalize.processing.screening] for screen containers and
  grouped rule families
- [`maps`][dronalize.processing.maps] for map builders, extraction config, and
  resolver helpers
- [`pipeline`][dronalize.processing.pipeline] for pipeline and transform
  abstractions

The advanced custom dataset loader API lives one level deeper in
[`dronalize.processing.loading`][] instead of being flattened into this root
surface.

## Related modules

- [`dronalize.processing.loading`][] for the advanced loader extension API
- [`dronalize.runtime`][] for runtime config resolution and planning
"""

from dronalize.processing import maps, pipeline, screening

__all__ = ["maps", "pipeline", "screening"]
