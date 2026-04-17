"""Thin loader API for dataset ingestion adapters."""

from dronalize.processing.loading.base import (
    BaseSceneLoader,
    DatasetOptionsModel,
)
from dronalize.processing.loading.loader import (
    LoadedSourceData,
    MapBinding,
    PreparedSceneData,
    Source,
)
from dronalize.processing.loading.resources import DatasetResources

__all__ = [
    "BaseSceneLoader",
    "DatasetOptionsModel",
    "DatasetResources",
    "LoadedSourceData",
    "MapBinding",
    "PreparedSceneData",
    "Source",
]
