"""Thin loader API for dataset ingestion adapters."""

from dronalize.processing.loading.base import BaseSceneLoader
from dronalize.processing.loading.loader import (
    LoadedSourceData,
    MapBinding,
    PreparedSceneData,
    Source,
)
from dronalize.processing.loading.resources import DatasetResources

__all__ = [
    "BaseSceneLoader",
    "DatasetResources",
    "LoadedSourceData",
    "MapBinding",
    "PreparedSceneData",
    "Source",
]
