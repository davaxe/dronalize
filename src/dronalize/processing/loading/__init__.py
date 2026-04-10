"""Thin loader API for dataset ingestion adapters."""

from dronalize.processing.loading.base import (
    BaseSceneLoader,
    LoaderOptions,
    LoaderSplitCapabilities,
)
from dronalize.processing.loading.loader import (
    BlockSplitSupport,
    LoadedSourceData,
    MapBinding,
    PreparedSceneData,
    Source,
)
from dronalize.processing.loading.resources import DatasetResources

__all__ = [
    "BaseSceneLoader",
    "BlockSplitSupport",
    "DatasetResources",
    "LoadedSourceData",
    "LoaderOptions",
    "LoaderSplitCapabilities",
    "MapBinding",
    "PreparedSceneData",
    "Source",
]
