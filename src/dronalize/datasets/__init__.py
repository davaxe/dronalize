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

from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from dronalize.config.models import MapConfig, ScenesConfig
from dronalize.datasets.registry import DatasetSpec, available, get, register
from dronalize.processing.loading import DatasetResources

ResourcesFactory = Callable[
    [Path, ScenesConfig, MapConfig | None], AbstractContextManager[DatasetResources]
]
"""Factory signature for dataset-scoped shared resources opened per execution run."""

__all__ = ["DatasetSpec", "ResourcesFactory", "available", "get", "register"]
