from dronalize.datasets.opendd.loader import OpenDDLoader
from dronalize.datasets.opendd.maps.builder import OpenDDMapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader("opendd", OpenDDLoader, has_map=True)

__all__ = ["DESCRIPTOR", "OpenDDLoader", "OpenDDMapBuilder"]
