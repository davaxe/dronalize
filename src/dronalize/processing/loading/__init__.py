"""Thin loader API for dataset ingestion adapters."""

from dronalize.processing.loading.base import SceneLoader
from dronalize.processing.loading.models import (
    DatasetSource,
    LoadedSourceFrame,
    LoaderOptionsModel,
    NoLoaderOptions,
)

__all__ = [
    "DatasetSource",
    "LoadedSourceFrame",
    "LoaderOptionsModel",
    "NoLoaderOptions",
    "SceneLoader",
]
