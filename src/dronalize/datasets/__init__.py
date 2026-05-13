"""Public dataset registry surface for built-in and custom integrations.

The dataset API is intentionally small:

- [`available`][dronalize.datasets.available] lists built-in and dynamically
  registered dataset keys
- [`get`][dronalize.datasets.get] resolves one key to a
  [`DatasetSpec`][dronalize.datasets.DatasetSpec]
- [`register`][dronalize.datasets.register] adds a custom dataset descriptor to
  the in-memory registry

Use [`ResourcesFactory`][dronalize.datasets.ResourcesFactory] when authoring
dataset integrations that need shared per-run state such as cached metadata or
map stores.
"""

from dronalize.config.models import MapConfig, ScenesConfig
from dronalize.datasets.registry import (
    DatasetFeatureSupport,
    DatasetSpec,
    ResourcesFactory,
    available,
    get,
    register,
)
from dronalize.processing.loading import DatasetResources

__all__ = [
    "DatasetResources",
    "DatasetFeatureSupport",
    "DatasetSpec",
    "MapConfig",
    "ResourcesFactory",
    "ScenesConfig",
    "available",
    "get",
    "register",
]
