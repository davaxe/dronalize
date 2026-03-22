from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.vod import _scope
from dronalize.datasets.vod.loader import VodLoader
from dronalize.datasets.vod.map.builder import VODMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "vod",
    VodLoader,
    VodLoader,
    execution_scope_fn=_scope.vod_execution_scope,
    has_map=True,
)

__all__ = ["DESCRIPTOR", "VODMapBuilder", "VodLoader"]
