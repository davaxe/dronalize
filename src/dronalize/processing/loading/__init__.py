"""Thin loader API for dataset ingestion adapters."""

from dronalize.processing.loading.base import SceneLoader
from dronalize.processing.loading.models import (
    DatasetOptionsModel,
    DatasetRunResources,
    DatasetSource,
    LoadedSourceFrame,
    MapReference,
    NoDatasetOptions,
)

__all__ = [
    "DatasetOptionsModel",
    "DatasetRunResources",
    "DatasetSource",
    "LoadedSourceFrame",
    "MapReference",
    "NoDatasetOptions",
    "SceneLoader",
]
