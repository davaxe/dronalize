from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor
from dronalize.datasets.vod import scope as _scope
from dronalize.datasets.vod.loader import VodLoader
from dronalize.datasets.vod.maps.builder import VODMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "vod",
    VodLoader,
    execution_scope_fn=_scope.vod_execution_scope,
    capabilities=DatasetCapabilities.MAP_AVAILABLE,
    infer_capabilities=True,
)

__all__ = ["DESCRIPTOR", "VODMapBuilder", "VodLoader"]
