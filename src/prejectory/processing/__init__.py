"""Grouped processing-facing public API.

## Import guide

```python
from prejectory.processing import maps, screening
```

Use this package as a lightweight namespace for the focused processing
subpackages:

- [`screening`][prejectory.processing.screening] for screen containers and
  grouped rule families
- [`maps`][prejectory.processing.maps] for map builders, extraction config, and
  resolver helpers

The advanced custom dataset loader API lives one level deeper in
[`prejectory.processing.loading`][] instead of being flattened into this root
surface.

## Related modules

- [`prejectory.processing.loading`][] for the advanced loader extension API
- [`prejectory.runtime`][] for runtime config resolution and planning
"""
