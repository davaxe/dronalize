from dronalize.datasets.registry import DatasetDescriptor
from dronalize.datasets.waymo.loader import WaymoLoader
from dronalize.datasets.waymo.maps.builder import WaymoMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader("waymo", WaymoLoader, has_map=True)

__all__ = ["DESCRIPTOR", "WaymoLoader", "WaymoMapBuilder"]
