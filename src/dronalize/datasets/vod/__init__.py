from dronalize.datasets.registry import DatasetDescriptor
from dronalize.datasets.vod import scope as _scope
from dronalize.datasets.vod.loader import VodLoader
from dronalize.datasets.vod.maps.builder import VODMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "vod",
    VodLoader,
    execution_scope_fn=_scope.vod_execution_scope,
    has_map=True,
)

__all__ = ["DESCRIPTOR", "VODMapBuilder", "VodLoader"]
