"""Public dataset registry surface for built-in and custom integrations.

The dataset API is intentionally small:

- [`list_datasets`][dronalize.datasets.list_datasets] lists built-in and dynamically
  registered dataset keys
- [`get_dataset`][dronalize.datasets.get_dataset] resolves one key to a
  [`DatasetDescriptor`][dronalize.datasets.DatasetDescriptor]
- [`register_dataset`][dronalize.datasets.register_dataset] adds a custom dataset descriptor to
  the in-memory registry

Use [`ResourcesFactory`][dronalize.datasets.ResourcesFactory] when authoring
dataset integrations that need shared per-run state such as cached metadata or
map stores.
"""

from dronalize.config.models import MapConfig, ScenesConfig
from dronalize.datasets.registry import (
    DatasetDescriptor,
    DatasetFeatureSupport,
    ResourcesFactory,
    get_dataset,
    list_datasets,
    register_dataset,
)
from dronalize.processing.loading import DatasetRunResources

__all__ = [
    "DatasetDescriptor",
    "DatasetFeatureSupport",
    "DatasetRunResources",
    "MapConfig",
    "ResourcesFactory",
    "ScenesConfig",
    "get_dataset",
    "list_datasets",
    "register_dataset",
]
