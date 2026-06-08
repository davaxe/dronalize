"""Thin loader API for dataset ingestion adapters."""

from dronalize.processing.loading.base import SceneLoader
from dronalize.processing.loading.models import (
    BoundMapResolver,
    DatasetRunResources,
    DatasetSource,
    LoadedSourceFrame,
    LoaderOptionsModel,
    MapProvider,
    MapReference,
    NoLoaderOptions,
    SharedMapProvider,
)

__all__ = [
    "BoundMapResolver",
    "DatasetRunResources",
    "DatasetSource",
    "LoadedSourceFrame",
    "LoaderOptionsModel",
    "MapProvider",
    "MapReference",
    "NoLoaderOptions",
    "SceneLoader",
    "SharedMapProvider",
]
