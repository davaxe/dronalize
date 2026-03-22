from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.waymo.loader import WaymoLoader
from dronalize.datasets.waymo.map.builder import WaymoMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader("waymo", WaymoLoader, WaymoLoader, has_map=True)

__all__ = ["DESCRIPTOR", "WaymoLoader", "WaymoMapBuilder"]
