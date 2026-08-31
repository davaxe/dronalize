"""Thin loader API for dataset ingestion adapters."""

from prejectory.processing.loading.base import SceneLoader
from prejectory.processing.loading.models import (
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
