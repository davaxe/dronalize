from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.opendd.loader import OpenDDLoader
from dronalize.datasets.opendd.map.builder import OpenDDMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader("opendd", OpenDDLoader, OpenDDLoader, has_map=True)

__all__ = ["DESCRIPTOR", "OpenDDLoader", "OpenDDMapBuilder"]
