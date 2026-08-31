"""Public dataset registry surface for built-in and custom integrations.

The dataset API is intentionally small:

- [`list_datasets`][prejectory.datasets.list_datasets] lists built-in and dynamically
  registered dataset keys
- [`get_dataset`][prejectory.datasets.get_dataset] resolves one key to a
  [`DatasetDescriptor`][prejectory.datasets.DatasetDescriptor]
- [`register_dataset`][prejectory.datasets.register_dataset] adds a custom dataset descriptor to
  the in-memory registry

Use
[`MapProviderFactory`][prejectory.datasets.shared.resources.MapProviderFactory]
when authoring dataset integrations that need run-scoped map-provider setup.
"""

from prejectory.config.models import MapConfig, PredictionTaskConfig, ScenesConfig
from prejectory.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    DatasetTemporalSupport,
    DatasetWindowingSupport,
    FrameBounds,
    get_dataset,
    list_datasets,
    register_dataset,
)

__all__ = [
    "DatasetDescriptor",
    "DatasetFeatureSupport",
    "DatasetTemporalSupport",
    "DatasetWindowingSupport",
    "FrameBounds",
    "MapConfig",
    "PredictionTaskConfig",
    "ScenesConfig",
    "get_dataset",
    "list_datasets",
    "register_dataset",
]
