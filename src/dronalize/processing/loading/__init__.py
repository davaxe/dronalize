"""Thin loader API for dataset ingestion adapters."""

from dronalize.processing.loading.base import SceneLoader
from dronalize.processing.loading.models import (
    DatasetRunResources,
    DatasetSource,
    LoadedSourceFrame,
    LoaderOptionsModel,
    NoLoaderOptions,
)

__all__ = [
    "DatasetRunResources",
    "DatasetSource",
    "LoadedSourceFrame",
    "LoaderOptionsModel",
    "NoLoaderOptions",
    "SceneLoader",
]
