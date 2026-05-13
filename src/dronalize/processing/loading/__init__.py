"""Thin loader API for dataset ingestion adapters."""

from dronalize.processing.loading.base import BaseSceneLoader
from dronalize.processing.loading.models import (
    DatasetOptionsModel,
    DatasetResources,
    LoadedSourceData,
    MapBinding,
    NoDatasetOptions,
    Source,
)

__all__ = [
    "BaseSceneLoader",
    "DatasetOptionsModel",
    "DatasetResources",
    "LoadedSourceData",
    "MapBinding",
    "NoDatasetOptions",
    "Source",
]
