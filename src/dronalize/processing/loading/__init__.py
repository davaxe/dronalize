"""Thin loader API for dataset ingestion adapters."""

from dronalize.processing.loading.base import ALL_SOURCES, BaseSceneLoader, SourceSelection
from dronalize.processing.loading.loader import LoadedSourceData, MapBinding, Source
from dronalize.processing.loading.options import DatasetOptionsModel, NoDatasetOptions
from dronalize.processing.loading.resources import DatasetResources

__all__ = [
    "ALL_SOURCES",
    "BaseSceneLoader",
    "DatasetOptionsModel",
    "DatasetResources",
    "LoadedSourceData",
    "MapBinding",
    "NoDatasetOptions",
    "Source",
    "SourceSelection",
]
